import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.common.networks import Actor, MLP

class TD3_BC:
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=2.5,     # The BC weight (Standard in TD3+BC paper)
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        
        # Networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        self.q1 = MLP(state_dim + action_dim, 1).to(device)
        self.q2 = MLP(state_dim + action_dim, 1).to(device)
        self.q1_target = MLP(state_dim + action_dim, 1).to(device)
        self.q2_target = MLP(state_dim + action_dim, 1).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.actor(state).cpu().numpy()[0]

    def update(self, batch):
        self.total_it += 1
        s = batch['observations'].to(self.device)
        a = batch['actions'].to(self.device)
        r = batch['rewards'].to(self.device)
        s_next = batch['next_observations'].to(self.device)
        d = batch['terminals'].to(self.device)
        
        # 1. Update Q-Functions
        with torch.no_grad():
            # Select action with target actor and add noise
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_a = (self.actor_target(s_next) + noise).clamp(0, 1)
            
            target_q1 = self.q1_target(torch.cat([s_next, next_a], dim=-1))
            target_q2 = self.q2_target(torch.cat([s_next, next_a], dim=-1))
            target_q = r + (1 - d) * self.gamma * torch.min(target_q1, target_q2)
            
        current_q1 = self.q1(torch.cat([s, a], dim=-1))
        current_q2 = self.q2(torch.cat([s, a], dim=-1))
        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # 2. Delayed Policy Update
        actor_loss = None
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(s)
            q = self.q1(torch.cat([s, pi], dim=-1))
            
            # TD3+BC secret: Lambda = Alpha / Mean(|Q|)
            lmbda = self.alpha / q.abs().mean().detach()
            
            actor_loss = -lmbda * q.mean() + F.mse_loss(pi, a)
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft Update Targets
            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)
            self._soft_update(self.actor, self.actor_target)
            
        return {
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item() if actor_loss is not None else 0.0,
            "q_val": current_q1.mean().item()
        }

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
