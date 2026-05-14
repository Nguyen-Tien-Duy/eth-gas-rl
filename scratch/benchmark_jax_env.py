import os
import sys
import jax
import jax.numpy as jnp
import time
from typing import NamedTuple

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment_jax import EnvState, EnvParams, TraceData, reset_env_jax, step_env_jax, get_obs

def run_vectorized_benchmark():
    num_episodes = 1000
    horizon = 128
    num_lags = 5
    
    # 1. Setup Parameters (Mock values)
    params = EnvParams(
        c_cap=100.0, c_base=21000.0,
        beta=0.1, alpha=2.0, lambda_d=100.0,
        min_log_fee=1.0, max_log_fee=5.0, max_queue=1000.0, max_volatility=0.5
    )
    
    # 2. Mock Trace Data for 1000 episodes
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    batch_gas = jax.random.uniform(key1, (num_episodes, horizon), minval=20.0, maxval=40.0)
    batch_ref = jax.random.uniform(key2, (num_episodes, horizon), minval=25.0, maxval=35.0)
    batch_arrivals = jax.random.randint(key3, (num_episodes, horizon), minval=0, maxval=20).astype(jnp.float32)
    
    traces = TraceData(gas_prices=batch_gas, gas_ref=batch_ref, arrivals=batch_arrivals)
    
    # 3. Vectorized Policy (Mock)
    def mock_policy(obs):
        return jnp.array([0.5])
    
    v_policy = jax.vmap(mock_policy)
    
    # 4. Vectorized Rollout Function
    @jax.jit
    def rollout_batch(params, traces):
        # Initial states - num_lags passed as static 5
        v_reset = jax.vmap(reset_env_jax, in_axes=(None, 0, None))
        states = v_reset(params, traces, 5)
        
        # Initial obs
        v_obs = jax.vmap(get_obs, in_axes=(0, None, None))
        obss = v_obs(states, params, 128)
        
        def scan_fn(carry, _):
            states, obss, total_rewards = carry
            actions = v_policy(obss)
            
            # Step all environments - horizon passed as static 128
            v_step = jax.vmap(step_env_jax, in_axes=(0, 0, None, 0, None))
            next_states, rewards, dones, infos = v_step(states, actions, params, traces, 128)
            
            # Get next observations
            next_obss = v_obs(next_states, params, 128)
            
            return (next_states, next_obss, total_rewards + rewards), rewards
        
        # Use jax.lax.scan for efficient looping
        (final_states, final_obss, total_rewards), _ = jax.lax.scan(
            scan_fn, (states, obss, jnp.zeros(num_episodes)), None, length=horizon-1
        )
        return total_rewards

    print(f"=== Benchmarking Vectorized Rollout ({num_episodes} episodes) ===")
    
    print("Compiling JIT kernels...")
    _ = rollout_batch(params, traces)
    
    start = time.time()
    rewards = rollout_batch(params, traces)
    rewards.block_until_ready()
    end = time.time()
    
    print(f"Executed {num_episodes} episodes in {end - start:.4f} seconds!")
    print(f"Average time per episode: {(end - start) / num_episodes * 1000:.4f} ms")
    print(f"Mean Reward: {jnp.mean(rewards):.4f}")

if __name__ == "__main__":
    run_vectorized_benchmark()
