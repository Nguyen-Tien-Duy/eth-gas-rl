import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Callable

class MLP(nn.Module):
    """Simple Multi-Layer Perceptron."""
    features: Sequence[int]
    activation: Callable = nn.relu
    activate_final: bool = False

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            if i < len(self.features) - 1 or self.activate_final:
                x = self.activation(x)
        return x

class Critic(nn.Module):
    """Twin Q-Network for Offline RL."""
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations, actions):
        inputs = jnp.concatenate([observations, actions], axis=-1)
        
        # Twin Q-networks to reduce overestimation bias
        q1 = MLP(self.hidden_dims + (1,))(inputs)
        q2 = MLP(self.hidden_dims + (1,))(inputs)
        
        return q1, q2

class ValueNet(nn.Module):
    """Value function network for IQL."""
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations):
        v = MLP(self.hidden_dims + (1,))(observations)
        return v

class Actor(nn.Module):
    """Policy network (Deterministic for Gas RL)."""
    hidden_dims: Sequence[int]
    action_dim: int = 1

    @nn.compact
    def __call__(self, observations):
        x = MLP(self.hidden_dims)(observations)
        x = nn.relu(x)
        # Action is [0, 1] for % of queue
        mu = nn.Dense(self.action_dim)(x)
        return nn.sigmoid(mu)

def create_iql_model(observation_dim: int, action_dim: int, hidden_dims=(256, 256)):
    """Helper to initialize model architectures."""
    actor = Actor(hidden_dims, action_dim)
    critic = Critic(hidden_dims)
    value = ValueNet(hidden_dims)
    return actor, critic, value
