import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.common.networks import Actor, MLP

class CQL:
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=1.0, # CQL weight
        cql_temp=1.0,
        use_amp: bool = False,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.cql_temp = cql_temp

        dev = device if isinstance(device, torch.device) else torch.device(device)
        self.use_amp = bool(use_amp) and dev.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.q1 = MLP(state_dim + action_dim, 1).to(device)
        self.q2 = MLP(state_dim + action_dim, 1).to(device)
        
        self.q1_target = MLP(state_dim + action_dim, 1).to(device)
        self.q2_target = MLP(state_dim + action_dim, 1).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.actor(state).cpu().numpy()[0]

    def update(self, batch):
        s = batch['observations'].to(self.device)
        a = batch['actions'].to(self.device)
        r = batch['rewards'].to(self.device)
        s_next = batch['next_observations'].to(self.device)
        d = batch['terminals'].to(self.device)

        autocast_ctx = torch.cuda.amp.autocast(enabled=self.use_amp)
        
        # --- 1. Q-Target Calculation ---
        with torch.no_grad():
            with autocast_ctx:
                a_next = self.actor(s_next)
                q1_next = self.q1_target(torch.cat([s_next, a_next], dim=-1))
                q2_next = self.q2_target(torch.cat([s_next, a_next], dim=-1))
                target_q = r + (1 - d) * self.gamma * torch.min(q1_next, q2_next)
            
        with autocast_ctx:
            current_q1 = self.q1(torch.cat([s, a], dim=-1))
            current_q2 = self.q2(torch.cat([s, a], dim=-1))

            # Standard Bellman Error
            q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            # --- 2. CQL Regularization (Simplified CORL version) ---
            random_actions = torch.rand_like(a).to(self.device)
            q1_rand = self.q1(torch.cat([s, random_actions], dim=-1))
            q2_rand = self.q2(torch.cat([s, random_actions], dim=-1))

            cql_loss1 = torch.logsumexp(q1_rand / self.cql_temp, dim=0) - current_q1.mean()
            cql_loss2 = torch.logsumexp(q2_rand / self.cql_temp, dim=0) - current_q2.mean()

            total_q_loss = q_loss + self.alpha * (cql_loss1 + cql_loss2)

        self.q_optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(total_q_loss).backward()
            self.scaler.step(self.q_optimizer)
            self.scaler.update()
        else:
            total_q_loss.backward()
            self.q_optimizer.step()
        
        # --- 3. Actor Update ---
        with autocast_ctx:
            new_a = self.actor(s)
            actor_loss = -self.q1(torch.cat([s, new_a], dim=-1)).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(actor_loss).backward()
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            actor_loss.backward()
            self.actor_optimizer.step()
        
        # --- 4. Target Soft Update ---
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
            
        return {
            "q_loss": q_loss.item(),
            "cql_loss": (cql_loss1 + cql_loss2).item(),
            "actor_loss": actor_loss.item(),
            "total_q_loss": total_q_loss.item()
        }

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path):
        def _sd(m):
            return m.module.state_dict() if hasattr(m, "module") else m.state_dict()
        torch.save({
            "actor": _sd(self.actor),
            "q1": _sd(self.q1),
            "q2": _sd(self.q2)
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        (self.actor.module if hasattr(self.actor, "module") else self.actor).load_state_dict(checkpoint["actor"])
        (self.q1.module if hasattr(self.q1, "module") else self.q1).load_state_dict(checkpoint["q1"])
        (self.q2.module if hasattr(self.q2, "module") else self.q2).load_state_dict(checkpoint["q2"])
