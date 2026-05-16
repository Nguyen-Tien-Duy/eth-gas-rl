import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from typing import Any, Tuple, Dict
from functools import partial
from src.models_jax import create_iql_model

@partial(jax.jit, static_argnums=(0,))
def _update_jit(agent, actor_state, critic_state, value_state, target_critic_params, batch):
    """
    Standalone JIT update function for IQL.
    agent: contains hyperparameters (tau, beta, discount, etc.)
    """
    obs, actions, rewards, next_obs, masks = (
        batch['observations'], batch['actions'], batch['rewards'], 
        batch['next_observations'], batch['masks']
    )

    # 1. Value Update (Expectile Regression)
    def value_loss_fn(v_params):
        v = value_state.apply_fn({'params': v_params}, obs)
        # Use target critic to get Q values
        q1, q2 = critic_state.apply_fn({'params': target_critic_params}, obs, actions)
        q = jnp.minimum(q1, q2)
        
        diff = q - v
        weight = jnp.where(diff > 0, agent.tau, 1 - agent.tau)
        v_loss = (weight * (diff ** 2)).mean()
        return v_loss, v

    (v_loss, v), v_grads = jax.value_and_grad(value_loss_fn, has_aux=True)(value_state.params)
    value_state = value_state.apply_gradients(grads=v_grads)

    # 2. Critic Update (TD Error)
    def critic_loss_fn(c_params):
        q1, q2 = critic_state.apply_fn({'params': c_params}, obs, actions)
        # Target Q: r + gamma * V(s')
        next_v = value_state.apply_fn({'params': value_state.params}, next_obs)
        target_q = rewards + agent.discount * masks * next_v
        q_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return q_loss

    q_loss, q_grads = jax.value_and_grad(critic_loss_fn)(critic_state.params)
    critic_state = critic_state.apply_gradients(grads=q_grads)

    # 3. Actor Update (Advantage Weighted Regression with Beta Distribution)
    def actor_loss_fn(a_params):
        q1, q2 = critic_state.apply_fn({'params': target_critic_params}, obs, actions)
        q = jnp.minimum(q1, q2)
        adv = q - v
        exp_adv = jnp.exp(jnp.minimum(adv * agent.beta, 100.0))
        
        alpha, beta = actor_state.apply_fn({'params': a_params}, obs)
        
        # Beta Log-Prob: (alpha-1)log(x) + (beta-1)log(1-x) - logB(alpha, beta)
        # Clip actions to avoid log(0)
        eps = 1e-6
        actions_clipped = jnp.clip(actions, eps, 1.0 - eps)
        
        log_prob = (alpha - 1.0) * jnp.log(actions_clipped) + \
                   (beta - 1.0) * jnp.log(1.0 - actions_clipped) - \
                   (jax.scipy.special.gammaln(alpha) + jax.scipy.special.gammaln(beta) - jax.scipy.special.gammaln(alpha + beta))
        
        # Weighted NLL loss
        a_loss = -(exp_adv * log_prob).mean()
        return a_loss

    a_loss, a_grads = jax.value_and_grad(actor_loss_fn)(actor_state.params)
    actor_state = actor_state.apply_gradients(grads=a_grads)

    # 4. Polyak Update for Target Critic
    new_target_params = optax.incremental_update(
        critic_state.params, target_critic_params, agent.target_update_rate
    )

    return actor_state, critic_state, value_state, new_target_params, {
        "v_loss": v_loss, "q_loss": q_loss, "a_loss": a_loss
    }

class IQLAgent:
    def __init__(self, 
                 observation_dim: int, 
                 action_dim: int, 
                 seed: int = 42,
                 lr: float = 3e-4,
                 tau: float = 0.7,
                 beta: float = 3.0,
                 discount: float = 0.99,
                 target_update_rate: float = 0.005):
        
        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.target_update_rate = target_update_rate
        
        self.n_devices = jax.local_device_count()
        print(f"JAX Agent initialized on {self.n_devices} device(s).")
        
        key = jax.random.PRNGKey(seed)
        actor_def, critic_def, value_def = create_iql_model(observation_dim, action_dim)
        
        key, actor_key, critic_key, value_key = jax.random.split(key, 4)
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
        
        self.value_state = train_state.TrainState.create(
            apply_fn=value_def.apply,
            params=value_def.init(value_key, obs_dummy)['params'],
            tx=optax.adam(lr)
        )
        
        # Replicate states if multi-device
        if self.n_devices > 1:
            from flax.training import common_utils
            self.actor_state = common_utils.replicate(self.actor_state)
            self.critic_state = common_utils.replicate(self.critic_state)
            self.value_state = common_utils.replicate(self.value_state)
            self.target_critic_params = common_utils.replicate(self.target_critic_params)

    def update(self, batch):
        if self.n_devices > 1:
            # Multi-device pmap update
            self.actor_state, self.critic_state, self.value_state, self.target_critic_params, metrics = \
                jax.pmap(_update_jit, axis_name='num_devices', static_broadcasted_argnums=(0,))(
                    self, self.actor_state, self.critic_state, self.value_state, self.target_critic_params, batch
                )
            # Average metrics across devices
            return jax.tree_map(lambda x: x.mean(), metrics)
        else:
            # Single-device jit update
            self.actor_state, self.critic_state, self.value_state, self.target_critic_params, metrics = \
                _update_jit(self, self.actor_state, self.critic_state, self.value_state, self.target_critic_params, batch)
            return metrics

    def select_action(self, observations):
        state = self.actor_state
        # Use unreplicated state for inference if multi-device
        if self.n_devices > 1:
            from flax.training import common_utils
            state = common_utils.unreplicate(state)
            
        if observations.ndim == 1:
            observations = observations[None, ...]
        
        alpha, beta = state.apply_fn({'params': state.params}, observations)
        # Return mean of Beta distribution: alpha / (alpha + beta)
        return (alpha / (alpha + beta))[0]
