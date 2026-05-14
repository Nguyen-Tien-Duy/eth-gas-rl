import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import time
import pandas as pd
import yaml
from functools import partial

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment import EthGasEnv
from src.environment_jax import EnvParams, TraceData, reset_env_jax, step_env_jax, get_obs

def run_grand_benchmark():
    num_episodes = 10000
    horizon = 128
    
    # 1. Load Real Config for authentic params
    config_path = "configs/exp_benchmark.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Mock Trace Data (shared for both)
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    gas_prices = np.random.uniform(20.0, 40.0, (num_episodes, horizon)).astype(np.float32)
    gas_ref = np.random.uniform(25.0, 35.0, (num_episodes, horizon)).astype(np.float32)
    arrivals = np.random.randint(0, 20, (num_episodes, horizon)).astype(np.float32)
    
    # ---------------------------------------------------------
    # SCENARIO 1: ORIGINAL (Numba + Python Loop)
    # ---------------------------------------------------------
    print(f"\n--- Scenario 1: Original (Numba + Python Loop) ---")
    # We mock a dataframe to fit EthGasEnv expectation
    mock_df = pd.DataFrame({
        'episode_id': np.repeat(np.arange(num_episodes), horizon),
        'base_fee_per_gas': gas_prices.flatten() * 1e9,
        'transaction_count': (arrivals.flatten() / 0.1).astype(int), # arrival_scale=0.1
        'gas_reference': gas_ref.flatten() * 1e9
    })
    
    env_original = EthGasEnv(config, trace_df=mock_df)
    
    start = time.time()
    original_rewards = []
    for ep in range(num_episodes):
        obs, _ = env_original.reset(options={"episode_id": ep})
        done = False
        total_r = 0
        while not done:
            action = np.array([0.5]) # Constant policy
            obs, reward, done, _, _ = env_original.step(action)
            total_r += reward
        original_rewards.append(total_r)
    end_original = time.time() - start
    print(f"Time: {end_original:.4f}s | Avg: {end_original/num_episodes*1000:.2f}ms/ep")

    # ---------------------------------------------------------
    # SCENARIO 2: JAX Vectorized (vmap + scan)
    # ---------------------------------------------------------
    print(f"\n--- Scenario 2: JAX Vectorized (vmap + scan) ---")
    params = EnvParams(
        c_cap=float(config['env']['execution_capacity']),
        c_base=float(config['env'].get('C_base', 21000.0)),
        beta=float(config['rl'].get('urgency_beta', 0.1)),
        alpha=float(config['rl'].get('urgency_alpha', 2.0)),
        lambda_d=float(config['rl'].get('deadline_penalty', 100.0)),
        min_log_fee=1.0, max_log_fee=10.0, max_queue=1000.0, max_volatility=0.5
    )
    
    traces_jax = TraceData(
        gas_prices=jnp.array(gas_prices),
        gas_ref=jnp.array(gas_ref),
        arrivals=jnp.array(arrivals)
    )
    
    @jax.jit
    def rollout_batch_jax(params, traces):
        v_reset = jax.vmap(reset_env_jax, in_axes=(None, 0, None))
        states = v_reset(params, traces, 5)
        
        v_obs = jax.vmap(get_obs, in_axes=(0, None, None))
        obss = v_obs(states, params, 128)
        
        def scan_fn(carry, _):
            states, obss, total_rewards = carry
            actions = jnp.full((num_episodes, 1), 0.5) # Constant policy
            
            v_step = jax.vmap(step_env_jax, in_axes=(0, 0, None, 0, None))
            next_states, rewards, dones, _ = v_step(states, actions, params, traces, 128)
            next_obss = v_obs(next_states, params, 128)
            return (next_states, next_obss, total_rewards + rewards), None
            
        (final_states, final_obss, total_rewards), _ = jax.lax.scan(
            scan_fn, (states, obss, jnp.zeros(num_episodes)), None, length=horizon-1
        )
        return total_rewards

    # Warmup
    _ = rollout_batch_jax(params, traces_jax)
    
    start = time.time()
    jax_rewards = rollout_batch_jax(params, traces_jax)
    jax_rewards.block_until_ready()
    end_jax = time.time() - start
    print(f"Time: {end_jax:.4f}s | Avg: {end_jax/num_episodes*1000:.4f}ms/ep")

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------
    print(f"\n" + "="*50)
    print(f"FINAL COMPARISON (Scientific Proof)")
    print(f"="*50)
    print(f"Original Time: {end_original:.4f}s")
    print(f"JAX Time:      {end_jax:.4f}s")
    print(f"SPEEDUP:       {end_original / end_jax:.1f}x")
    print(f"="*50)
    
    # Check numerical drift (should be very small)
    mean_diff = np.mean(np.abs(np.array(original_rewards) - np.array(jax_rewards)))
    print(f"Mean Reward Difference: {mean_diff:.6f}")

if __name__ == "__main__":
    run_grand_benchmark()
