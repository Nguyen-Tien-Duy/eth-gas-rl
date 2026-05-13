import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.common.networks import VAE, MLP

class BCQ:
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        phi=0.05,     # Perturbation range
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.phi = phi
        self.action_dim = action_dim
        
        # Networks
        self.vae = VAE(state_dim, action_dim, action_dim * 2, device=device).to(device)
        self.actor = MLP(state_dim + action_dim, action_dim).to(device) # Perturbation network
        self.q1 = MLP(state_dim + action_dim, 1).to(device)
        self.q2 = MLP(state_dim + action_dim, 1).to(device)
        
        self.actor_target = MLP(state_dim + action_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.q1_target = MLP(state_dim + action_dim, 1).to(device)
        self.q2_target = MLP(state_dim + action_dim, 1).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).repeat(10, 1)
            # Sample 10 actions from VAE
            action = self.vae.decode(state)
            # Perturb them
            action = (action + self.phi * self.actor(torch.cat([state, action], 1))).clamp(0, 1)
            # Pick the best one according to Q-network
            q = self.q1(torch.cat([state, action], 1))
            ind = q.argmax(0)
            return action[ind].cpu().numpy().flatten()

    def update(self, batch):
        s = batch['observations'].to(self.device)
        a = batch['actions'].to(self.device)
        r = batch['rewards'].to(self.device)
        s_next = batch['next_observations'].to(self.device)
        d = batch['terminals'].to(self.device)
        
        # 1. Update VAE
        recon, mean, std = self.vae(s, a)
        recon_loss = F.mse_loss(recon, a)
        KL_loss = -0.5 * (1 + torch.log(std**2) - mean**2 - std**2).mean()
        vae_loss = recon_loss + 0.5 * KL_loss
        
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        
        # 2. Update Q-Functions
        with torch.no_grad():
            # Sample 10 actions for target calculation
            s_next_rep = s_next.repeat_interleave(10, 0)
            a_next = self.vae.decode(s_next_rep)
            a_next = (a_next + self.phi * self.actor_target(torch.cat([s_next_rep, a_next], 1))).clamp(0, 1)
            
            target_q1 = self.q1_target(torch.cat([s_next_rep, a_next], 1))
            target_q2 = self.q2_target(torch.cat([s_next_rep, a_next], 1))
            # Max over sampled actions, then Min over twin Qs
            target_q = torch.min(target_q1, target_q2).reshape(s.shape[0], 10).max(1)[0].reshape(-1, 1)
            target_q = r + (1 - d) * self.gamma * target_q
            
        current_q1 = self.q1(torch.cat([s, a], dim=-1))
        current_q2 = self.q2(torch.cat([s, a], dim=-1))
        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # 3. Update Actor (Perturbation Network)
        sampled_a = self.vae.decode(s)
        perturbed_a = (sampled_a + self.phi * self.actor(torch.cat([s, sampled_a], 1))).clamp(0, 1)
        actor_loss = -self.q1(torch.cat([s, perturbed_a], 1)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 4. Target Soft Update
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        self._soft_update(self.actor, self.actor_target)
            
        return {
            "vae_loss": vae_loss.item(),
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item()
        }

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path):
        torch.save({
            "vae": self.vae.state_dict(),
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.vae.load_state_dict(checkpoint["vae"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
