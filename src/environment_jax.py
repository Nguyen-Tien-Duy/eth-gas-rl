import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import NamedTuple, Dict
from src.physics_jax import step_physics_jax, compute_reward_jax

class EnvState(NamedTuple):
    queue: jnp.ndarray
    current_step: jnp.ndarray
    gas_history: jnp.ndarray # Array of shape (num_lags,)
    done: jnp.ndarray

class EnvParams(NamedTuple):
    c_cap: float
    c_base: float
    beta: float
    alpha: float
    lambda_d: float
    # Normalization bounds
    min_log_fee: float
    max_log_fee: float
    max_queue: float
    max_volatility: float

class TraceData(NamedTuple):
    gas_prices: jnp.ndarray # (H,)
    gas_ref: jnp.ndarray    # (H,)
    arrivals: jnp.ndarray   # (H,)

@partial(jit, static_argnums=(2,))
def get_obs(state: EnvState, params: EnvParams, horizon: int):
    """
    Vectorized Observation function for JAX.
    """
    time_ratio = state.current_step / jnp.float32(horizon)
    current_gas = state.gas_history[-1]
    lags = state.gas_history
    
    # 1. Normalize Queue
    norm_q = state.queue / (params.max_queue + 1e-9)
    
    # 2. Normalize Gas (Log-scale)
    def norm_log_gas(g):
        log_g = jnp.log(g + 1e-9)
        return (log_g - params.min_log_fee) / (params.max_log_fee - params.min_log_fee + 1e-9)
    
    norm_gas = norm_log_gas(current_gas)
    norm_lags = jax.vmap(norm_log_gas)(lags)
    
    # 3. Volatility (Standard deviation of log-lags)
    log_lags = jnp.log(lags + 1e-9)
    vol = jnp.std(log_lags)
    norm_vol = vol / (params.max_volatility + 1e-9)
    
    # Combine into 9-dim vector
    return jnp.concatenate([
        jnp.array([norm_q]),
        jnp.array([norm_gas]),
        jnp.array([norm_vol]),
        norm_lags,
        jnp.array([time_ratio])
    ])

@partial(jit, static_argnums=(4,))
def step_env_jax(state: EnvState, action: jnp.ndarray, params: EnvParams, trace: TraceData, horizon: int):
    """
    A single step in the JAX environment.
    """
    # 1. Fetch trace data for current step
    current_gas = trace.gas_prices[state.current_step]
    ref_gas = trace.gas_ref[state.current_step]
    arrival = trace.arrivals[state.current_step]
    time_ratio = state.current_step / jnp.float32(horizon)
    
    # 2. Physics Step
    n_intended = action[0] * state.queue
    n_intended = jnp.minimum(n_intended, params.c_cap)
    
    next_queue, action_clamped = step_physics_jax(state.queue, n_intended, arrival)
    
    # 3. Reward
    reward = compute_reward_jax(
        action_clamped, next_queue, current_gas, ref_gas,
        params.c_base, params.beta, params.alpha, time_ratio
    )
    
    # 4. Update State
    next_step = state.current_step + 1
    done = next_step >= (horizon - 1)
    
    # Update gas history (rolling window)
    next_history = jnp.roll(state.gas_history, -1)
    next_history = next_history.at[-1].set(current_gas)
    
    # 5. Final Penalty
    final_penalty = jnp.where(done, (next_queue * params.lambda_d) / 100.0, 0.0)
    reward = reward - final_penalty
    
    next_state = EnvState(
        queue=next_queue,
        current_step=next_step,
        gas_history=next_history,
        done=done
    )
    
    return next_state, reward, done, {"savings": action_clamped * (ref_gas - current_gas)}

@partial(jit, static_argnums=(2,))
def reset_env_jax(params: EnvParams, trace: TraceData, num_lags: int):
    """
    Reset function for JAX environment.
    """
    initial_gas = trace.gas_prices[0]
    state = EnvState(
        queue=0.0,
        current_step=0,
        gas_history=jnp.full((num_lags,), initial_gas),
        done=False
    )
    return state
