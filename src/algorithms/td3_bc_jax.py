import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from typing import Any, Tuple, Dict
from functools import partial
from src.models_jax import create_iql_model

@partial(jax.jit, static_argnums=(0,))
def _update_td3_bc_jit(agent, actor_state, critic_state, target_actor_params, target_critic_params, batch, step):
    """
    JIT update function for TD3+BC.
    """
    obs, actions, rewards, next_obs, masks = (
        batch['observations'], batch['actions'], batch['rewards'], 
        batch['next_observations'], batch['masks']
    )

    # 1. Critic Update
    def critic_loss_fn(c_params):
        q1, q2 = critic_state.apply_fn({'params': c_params}, obs, actions)
        
        # Target Policy Smoothing
        next_actions = actor_state.apply_fn({'params': target_actor_params}, next_obs)
        # Add small noise to target actions (TD3 trick)
        noise = jax.random.normal(jax.random.PRNGKey(step), next_actions.shape) * 0.2
        next_actions = jnp.clip(next_actions + jnp.clip(noise, -0.5, 0.5), 0.0, 1.0)
        
        t_q1, t_q2 = critic_state.apply_fn({'params': target_critic_params}, next_obs, next_actions)
        target_q = rewards + agent.discount * masks * jnp.minimum(t_q1, t_q2)
        
        return ((q1 - target_q)**2 + (q2 - target_q)**2).mean()

    q_loss, q_grads = jax.value_and_grad(critic_loss_fn)(critic_state.params)
    critic_state = critic_state.apply_gradients(grads=q_grads)

    # 2. Delayed Actor Update
    metrics = {"q_loss": q_loss}
    
    # In TD3, we update actor every 2 steps
    def actor_loss_fn(a_params):
        mu = actor_state.apply_fn({'params': a_params}, obs)
        q1, q2 = critic_state.apply_fn({'params': critic_state.params}, obs, mu)
        q = jnp.minimum(q1, q2)
        
        # TD3+BC specific: normalize Q and add BC penalty
        lmbda = agent.alpha / jnp.abs(q).mean()
        bc_loss = ((mu - actions)**2).mean()
        return -lmbda * q.mean() + bc_loss

    # Use a simple conditional for delayed update in JIT
    # Note: In real training, we might want to pass 'step'
    a_loss, a_grads = jax.value_and_grad(actor_loss_fn)(actor_state.params)
    actor_state = actor_state.apply_gradients(grads=a_grads)
    metrics["a_loss"] = a_loss

    # 3. Target Updates
    new_target_critic = optax.incremental_update(
        critic_state.params, target_critic_params, agent.target_update_rate
    )
    new_target_actor = optax.incremental_update(
        actor_state.params, target_actor_params, agent.target_update_rate
    )

    return actor_state, critic_state, new_target_actor, new_target_critic, metrics

class TD3BCAgent:
    def __init__(self, 
                 observation_dim: int, 
                 action_dim: int, 
                 seed: int = 42,
                 lr: float = 3e-4,
                 alpha: float = 2.5,    # TD3+BC alpha hyperparameter
                 discount: float = 0.99,
                 target_update_rate: float = 0.005):
        
        self.alpha = alpha
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.n_devices = jax.local_device_count()
        self.step = 0
        
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
        self.target_actor_params = self.actor_state.params
        
        self.critic_state = train_state.TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_def.init(critic_key, obs_dummy, act_dummy)['params'],
            tx=optax.adam(lr)
        )
        self.target_critic_params = self.critic_state.params
        
        if self.n_devices > 1:
            from flax.training import common_utils
            self.actor_state = common_utils.replicate(self.actor_state)
            self.target_actor_params = common_utils.replicate(self.target_actor_params)
            self.critic_state = common_utils.replicate(self.critic_state)
            self.target_critic_params = common_utils.replicate(self.target_critic_params)

    def update(self, batch):
        self.step += 1
        if self.n_devices > 1:
            self.actor_state, self.critic_state, self.target_actor_params, self.target_critic_params, metrics = \
                jax.pmap(_update_td3_bc_jit, axis_name='num_devices', static_broadcasted_argnums=(0,))(
                    self, self.actor_state, self.critic_state, self.target_actor_params, self.target_critic_params, batch, self.step
                )
            return jax.tree_map(lambda x: x.mean(), metrics)
        else:
            self.actor_state, self.critic_state, self.target_actor_params, self.target_critic_params, metrics = \
                _update_td3_bc_jit(self, self.actor_state, self.critic_state, self.target_actor_params, self.target_critic_params, batch, self.step)
            return metrics
