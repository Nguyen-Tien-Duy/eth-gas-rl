import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from typing import Any, Tuple, Dict
from functools import partial
from src.models_jax import create_iql_model

@partial(jax.jit, static_argnums=(0,))
def _update_awac_jit(agent, actor_state, critic_state, target_critic_params, batch):
    """
    JIT update function for AWAC.
    """
    obs, actions, rewards, next_obs, masks = (
        batch['observations'], batch['actions'], batch['rewards'], 
        batch['next_observations'], batch['masks']
    )

    # 1. Critic Update (Standard SARSA-style for Offline RL)
    def critic_loss_fn(c_params):
        q1, q2 = critic_state.apply_fn({'params': c_params}, obs, actions)
        
        # In AWAC, we use the current policy to get next actions for target
        next_actions = actor_state.apply_fn({'params': actor_state.params}, next_obs)
        t_q1, t_q2 = critic_state.apply_fn({'params': target_critic_params}, next_obs, next_actions)
        target_q = rewards + agent.discount * masks * jnp.minimum(t_q1, t_q2)
        
        return ((q1 - target_q)**2 + (q2 - target_q)**2).mean()

    q_loss, q_grads = jax.value_and_grad(critic_loss_fn)(critic_state.params)
    critic_state = critic_state.apply_gradients(grads=q_grads)

    # 2. Actor Update (Advantage Weighted Regression with Beta Distribution)
    def actor_loss_fn(a_params):
        # Q(s, a) - Q(s, pi(s))
        q1, q2 = critic_state.apply_fn({'params': critic_state.params}, obs, actions)
        q = jnp.minimum(q1, q2)
        
        alpha, beta = actor_state.apply_fn({'params': a_params}, obs)
        mu = alpha / (alpha + beta)
        
        v_q1, v_q2 = critic_state.apply_fn({'params': critic_state.params}, obs, mu)
        v = jnp.minimum(v_q1, v_q2)
        
        adv = q - v
        weight = jnp.exp(jnp.minimum(adv / agent.lam, 100.0))
        
        # Beta Log-Prob for BC
        eps = 1e-6
        actions_clipped = jnp.clip(actions, eps, 1.0 - eps)
        log_prob = (alpha - 1.0) * jnp.log(actions_clipped) + \
                   (beta - 1.0) * jnp.log(1.0 - actions_clipped) - \
                   (jax.scipy.special.gammaln(alpha) + jax.scipy.special.gammaln(beta) - jax.scipy.special.gammaln(alpha + beta))
        
        return -(weight * log_prob).mean()

    a_loss, a_grads = jax.value_and_grad(actor_loss_fn)(actor_state.params)
    actor_state = actor_state.apply_gradients(grads=a_grads)

    # 3. Target Update
    new_target_params = optax.incremental_update(
        critic_state.params, target_critic_params, agent.target_update_rate
    )

    return actor_state, critic_state, new_target_params, {
        "q_loss": q_loss, "a_loss": a_loss
    }

class AWACAgent:
    def __init__(self, 
                 observation_dim: int, 
                 action_dim: int, 
                 seed: int = 42,
                 lr: float = 3e-4,
                 lam: float = 1.0,      # Temperature for AWAC
                 discount: float = 0.99,
                 target_update_rate: float = 0.005):
        
        self.lam = lam
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.n_devices = jax.local_device_count()
        
        key = jax.random.PRNGKey(seed)
        actor_def, critic_def, _ = create_iql_model(observation_dim, action_dim)
        
        key, actor_key, critic_key = jax.random.split(key, 3)
        obs_dummy = jnp.zeros((1, observation_dim))
        act_dummy = jnp.zeros((1, action_dim))

        self.actor_state = train_state.TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_def.init(actor_key, obs_dummy)['params'],
            tx=optax.adam(lr)
        )
        
        self.critic_state = train_state.TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_def.init(critic_key, obs_dummy, act_dummy)['params'],
            tx=optax.adam(lr)
        )
        self.target_critic_params = self.critic_state.params
        
        if self.n_devices > 1:
            from flax.training import common_utils
            self.actor_state = common_utils.replicate(self.actor_state)
            self.critic_state = common_utils.replicate(self.critic_state)
            self.target_critic_params = common_utils.replicate(self.target_critic_params)

    def update(self, batch):
        if self.n_devices > 1:
            self.actor_state, self.critic_state, self.target_critic_params, metrics = \
                jax.pmap(_update_awac_jit, axis_name='num_devices', static_broadcasted_argnums=(0,))(
                    self, self.actor_state, self.critic_state, self.target_critic_params, batch
                )
            return jax.tree_map(lambda x: x.mean(), metrics)
        else:
            self.actor_state, self.critic_state, self.target_critic_params, metrics = \
                _update_awac_jit(self, self.actor_state, self.critic_state, self.target_critic_params, batch)
            return metrics

    def select_action(self, observations):
        state = self.actor_state
        if self.n_devices > 1:
            from flax.training import common_utils
            state = common_utils.unreplicate(state)
            
        if observations.ndim == 1:
            observations = observations[None, ...]
            
        alpha, beta = state.apply_fn({'params': state.params}, observations)
        return (alpha / (alpha + beta))[0]
