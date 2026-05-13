import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.common.networks import Actor, MLP, Value

class IQL:
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        expectile=0.7, # The secret sauce of IQL
        beta=3.0,     # Temperature for Advantage weighting
        use_amp: bool = False,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.beta = beta

        dev = device if isinstance(device, torch.device) else torch.device(device)
        self.use_amp = bool(use_amp) and dev.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.q1 = MLP(state_dim + action_dim, 1).to(device)
        self.q2 = MLP(state_dim + action_dim, 1).to(device)
        self.vf = Value(state_dim).to(device)
        
        self.q1_target = MLP(state_dim + action_dim, 1).to(device)
        self.q2_target = MLP(state_dim + action_dim, 1).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=lr)
        
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.actor(state).cpu().numpy()[0]

    def _expectile_loss(self, diff, expectile):
        weight = torch.where(diff < 0, 1 - expectile, expectile)
        return torch.mean(weight * (diff**2))

    def update(self, batch):
        s = batch['observations'].to(self.device)
        a = batch['actions'].to(self.device)
        r = batch['rewards'].to(self.device)
        s_next = batch['next_observations'].to(self.device)
        d = batch['terminals'].to(self.device)

        autocast_ctx = torch.cuda.amp.autocast(enabled=self.use_amp)
        
        with torch.no_grad():
            with autocast_ctx:
                q_target = torch.min(
                    self.q1_target(torch.cat([s, a], dim=-1)),
                    self.q2_target(torch.cat([s, a], dim=-1))
                )
            
        # 1. Update Value Function (V)
        with autocast_ctx:
            v_pred = self.vf(s)
            v_loss = self._expectile_loss(q_target - v_pred, self.expectile)

        self.v_optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(v_loss).backward()
            self.scaler.step(self.v_optimizer)
            self.scaler.update()
        else:
            v_loss.backward()
            self.v_optimizer.step()
        
        # 2. Update Q-Functions
        with torch.no_grad():
            with autocast_ctx:
                next_v = self.vf(s_next)
                target_q = r + (1 - d) * self.gamma * next_v
            
        with autocast_ctx:
            current_q1 = self.q1(torch.cat([s, a], dim=-1))
            current_q2 = self.q2(torch.cat([s, a], dim=-1))
            q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.q_optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(q_loss).backward()
            self.scaler.step(self.q_optimizer)
            self.scaler.update()
        else:
            q_loss.backward()
            self.q_optimizer.step()
        
        # 3. Update Actor (Advantage Weighted Regression)
        with torch.no_grad():
            adv = q_target - v_pred
            exp_adv = torch.exp(adv * self.beta).clamp(max=100.0)
            
        with autocast_ctx:
            a_pred = self.actor(s)
            actor_loss = torch.mean(exp_adv * F.mse_loss(a_pred, a, reduction='none'))

        self.actor_optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(actor_loss).backward()
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            actor_loss.backward()
            self.actor_optimizer.step()
        
        # 4. Target Soft Update
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
            
        return {
            "q_loss": q_loss.item(),
            "v_loss": v_loss.item(),
            "actor_loss": actor_loss.item(),
            "avg_advantage": adv.mean().item()
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
            "q2": _sd(self.q2),
            "vf": _sd(self.vf)
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        (self.actor.module if hasattr(self.actor, "module") else self.actor).load_state_dict(checkpoint["actor"])
        (self.q1.module if hasattr(self.q1, "module") else self.q1).load_state_dict(checkpoint["q1"])
        (self.q2.module if hasattr(self.q2, "module") else self.q2).load_state_dict(checkpoint["q2"])
        (self.vf.module if hasattr(self.vf, "module") else self.vf).load_state_dict(checkpoint["vf"])
