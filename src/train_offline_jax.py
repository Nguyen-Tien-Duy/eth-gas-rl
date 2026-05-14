import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
import time
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import get_offline_loader
from src.algorithms.iql_jax import IQLAgent
from src.algorithms.cql_jax import CQLAgent
from src.algorithms.awac_jax import AWACAgent
from src.algorithms.td3_bc_jax import TD3BCAgent
from src.algorithms.bcq_jax import BCQAgent
from src.environment_jax import EnvParams, TraceData, reset_env_jax, step_env_jax, get_obs

def evaluate_vectorized(agent, params, traces, num_episodes, horizon):
    """Fast evaluation using JAX vmap across all validation episodes."""
    v_reset = jax.vmap(reset_env_jax, in_axes=(None, 0, None))
    states = v_reset(params, traces, 5) # num_lags=5
    
    v_obs = jax.vmap(get_obs, in_axes=(0, None, None))
    obss = v_obs(states, params, horizon)
    
    # Handle multi-device state
    actor_state = agent.actor_state
    if agent.n_devices > 1:
        from flax.training import common_utils
        actor_state = common_utils.unreplicate(actor_state)
    
    def scan_fn(carry, _):
        states, obss, total_rewards, total_savings, max_backlogs = carry
        
        # Get actions from agent
        actions = actor_state.apply_fn({'params': actor_state.params}, obss)
        
        v_step = jax.vmap(step_env_jax, in_axes=(0, 0, None, 0, None))
        next_states, rewards, dones, infos = v_step(states, actions, params, traces, horizon)
        
        next_obss = v_obs(next_states, params, horizon)
        
        return (
            next_states, next_obss, 
            total_rewards + rewards, 
            total_savings + infos['savings'],
            jnp.maximum(max_backlogs, next_states.queue)
        ), None
        
    (final_states, final_obss, total_rewards, total_savings, max_backlogs), _ = jax.lax.scan(
        scan_fn, (states, obss, jnp.zeros(num_episodes), jnp.zeros(num_episodes), jnp.zeros(num_episodes)), 
        None, length=horizon-1
    )
    
    return {
        "mean_reward": jnp.mean(total_rewards),
        "total_savings": jnp.sum(total_savings),
        "avg_backlog": jnp.mean(final_states.queue),
        "max_backlog": jnp.max(max_backlogs)
    }

def train_offline():
    # Force JAX to use CPU platform if needed, but let it auto-detect first
    # 1. Load Config
    config_path = "configs/exp_benchmark.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    exp_name = config['experiment_name']
    
    # 2. Setup Data Loader (PyTorch) - Tăng lên 6 workers để bóc lột 6/8 threads
    print("Loading data with 6 workers...")
    data_dir = f"data/processed/{exp_name}"
    train_loader = get_offline_loader(
        parquet_path=f"{data_dir}/train.parquet",
        metadata_path=f"{data_dir}/metadata.json",
        batch_size=8192,
        num_workers=6
    )
    
    # Load validation trace for fast eval
    val_path = f"{data_dir}/val.parquet"
    val_df = pd.read_parquet(val_path)
    
    # Prepare JAX Trace Data (FULL Validation Set)
    print(f"Preparing full validation set ({len(val_df['episode_id'].unique())} episodes)...")
    ep_ids = val_df['episode_id'].unique()
    num_val_episodes = len(ep_ids)
    
    horizon = config['env']['horizon']
    val_gas = val_df['base_fee_per_gas'].values.reshape(-1, horizon) / 1e9
    val_ref = val_df['gas_reference'].values.reshape(-1, horizon) / 1e9
    
    # Fix arrival scale calculation
    a_scale = config['env'].get('arrival_scale', 0.1)
    val_arr = (val_df['transaction_count'].values.reshape(-1, horizon) * a_scale)
    
    traces_jax = TraceData(
        gas_prices=jnp.array(val_gas),
        gas_ref=jnp.array(val_ref),
        arrivals=jnp.array(val_arr)
    )
    
    env_params = EnvParams(
        c_cap=float(config['env']['execution_capacity']),
        c_base=21000.0, beta=0.1, alpha=2.0, lambda_d=100.0,
        min_log_fee=1.0, max_log_fee=10.0, max_queue=1000.0, max_volatility=0.5
    )

    # 3. Initialize Agent
    algo_name = config.get('rl', {}).get('algorithm', 'iql').lower()
    print(f"Initializing JAX Agent: {algo_name.upper()}")
    
    if algo_name == 'iql':
        agent = IQLAgent(observation_dim=9, action_dim=1)
    elif algo_name == 'cql':
        agent = CQLAgent(observation_dim=9, action_dim=1)
    elif algo_name == 'awac':
        agent = AWACAgent(observation_dim=9, action_dim=1)
    elif algo_name == 'td3_bc':
        agent = TD3BCAgent(observation_dim=9, action_dim=1)
    elif algo_name == 'bcq':
        agent = BCQAgent(observation_dim=9, action_dim=1)
    else:
        raise ValueError(f"Unsupported JAX algorithm: {algo_name}")
    
    # 4. Training Loop
    epochs = 10
    print(f"Starting Training ({algo_name.upper()} on JAX/Flax)...")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Convert torch tensors to jax arrays
            observations = jnp.array(batch['observations'].numpy())
            actions = jnp.array(batch['actions'].numpy())
            rewards = jnp.array(batch['rewards'].numpy().flatten())
            next_obs = jnp.array(batch['next_observations'].numpy())
            masks = 1.0 - jnp.array(batch['terminals'].numpy().flatten())
            
            # Shard data if multi-device
            if agent.n_devices > 1:
                def shard(x):
                    return x.reshape((agent.n_devices, -1) + x.shape[1:])
                jax_batch = {
                    'observations': shard(observations),
                    'actions': shard(actions),
                    'rewards': shard(rewards),
                    'next_observations': shard(next_obs),
                    'masks': shard(masks)
                }
            else:
                jax_batch = {
                    'observations': observations,
                    'actions': actions,
                    'rewards': rewards,
                    'next_observations': next_obs,
                    'masks': masks
                }
                
            agent.update(jax_batch)
            
        # Evaluation ON FULL SET
        metrics = evaluate_vectorized(agent, env_params, traces_jax, num_val_episodes, horizon)
        
        duration = time.time() - epoch_start
        print(f"\n--- Epoch {epoch+1} Report ({duration:.2f}s) ---")
        print(f"  Eval Reward:   {metrics['mean_reward']:.4f}")
        print(f"  Total Savings: {metrics['total_savings']:.0f} Gwei")
        print(f"  Avg Backlog:   {metrics['avg_backlog']:.2f}")
        print(f"  Max Backlog:   {metrics['max_backlog']:.2f}")
        print("-" * 30)

if __name__ == "__main__":
    train_offline()
