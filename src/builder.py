import argparse
import yaml
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from numba import njit, prange
import math
from concurrent.futures import ProcessPoolExecutor
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.milp_oracle import solve_episode_milp

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@njit
def _compute_features_numba(log_fee, gas_used, target_gas, tx_count, window_size, num_lags):
    n = len(log_fee)
    
    # Pre-allocate arrays
    returns = np.zeros(n)
    volatility = np.zeros(n)
    utilization = np.zeros(n)
    lags = np.zeros((n, num_lags))
    
    # NEW 14-D Features
    acceleration = np.zeros(n)
    surprise = np.zeros(n)
    backlog_pressure = np.zeros(n)
    
    # Calculate returns and utilization
    for i in range(n):
        utilization[i] = gas_used[i] / target_gas[i]
        if i > 0:
            returns[i] = log_fee[i] - log_fee[i-1]
            acceleration[i] = returns[i] - returns[i-1]
            
    # Calculate rolling std and lags
    for i in range(n):
        # Lags
        for j in range(num_lags):
            if i > j:
                lags[i, j] = log_fee[i - j - 1]
            else:
                lags[i, j] = log_fee[0] # Pad with first value
                
        # Rolling Volatility and Surprise
        if i >= window_size - 1:
            window_sum_ret = 0.0
            window_sq_sum_ret = 0.0
            window_sum_tx = 0.0
            window_sq_sum_tx = 0.0
            for k in range(i - window_size + 1, i + 1):
                val_ret = returns[k]
                window_sum_ret += val_ret
                window_sq_sum_ret += val_ret * val_ret
                
                val_tx = tx_count[k]
                window_sum_tx += val_tx
                window_sq_sum_tx += val_tx * val_tx
            
            mean_ret = window_sum_ret / window_size
            variance_ret = (window_sq_sum_ret / window_size) - (mean_ret * mean_ret)
            if variance_ret > 0:
                volatility[i] = math.sqrt(variance_ret)
            else:
                volatility[i] = 0.0
                
            mean_tx = window_sum_tx / window_size
            variance_tx = (window_sq_sum_tx / window_size) - (mean_tx * mean_tx)
            std_tx = math.sqrt(variance_tx) if variance_tx > 0 else 1.0
            surprise[i] = (tx_count[i] - mean_tx) / std_tx
        else:
            volatility[i] = 0.0
            surprise[i] = 0.0
            
        # Backlog Pressure (AR model)
        if i > 0:
            backlog_pressure[i] = max(0.0, 0.95 * backlog_pressure[i-1] + 0.3 * utilization[i] + 0.2 * surprise[i])
        else:
            backlog_pressure[i] = 0.0
                
    return returns, volatility, utilization, lags, acceleration, surprise, backlog_pressure

@njit
def _simulate_queue_numba(arrivals, capacity):
    max_queue = 0
    current_queue = 0
    for i in range(len(arrivals)):
        current_queue += arrivals[i]
        if current_queue < capacity:
            current_queue = 0
        else:
            current_queue -= capacity
            
        if current_queue > max_queue:
            max_queue = current_queue
            
    return max_queue

def simulate_queue_to_find_max(df, capacity, arrival_scale):
    print(f"Simulating queue with C_cap={capacity}, Arrival_Scale={arrival_scale}...")
    arrivals = (df['transaction_count'].values * arrival_scale).astype(np.int64)
    max_queue = _simulate_queue_numba(arrivals, int(capacity))
    return max_queue

@njit
def _simulate_policy_numba(policy_type_idx, gas_prices, gas_ref, arrivals, c_cap, H):
    """Extreme Vectorization for non-MILP policies using Numba JIT"""
    n = np.zeros(H, dtype=np.float32)
    q = np.zeros(H, dtype=np.float32)
    q_curr = 0.0
    
    # policy_type_idx: 2=heuristic, 3=random
    for t in range(H):
        q[t] = q_curr
        if policy_type_idx == 2: # heuristic
            panic_mode = (t > 0.9 * H)
            cheap_gas = (gas_prices[t] < gas_ref[t])
            n_t = min(q_curr, c_cap) if (panic_mode or cheap_gas) else 0.0
        else: # random (Note: np.random.randint is not JIT friendly in some versions, using uniform)
            if q_curr > 0.5:
                # Simple deterministic-ish random for JIT
                n_t = (t % (int(min(q_curr, c_cap)) + 1)) 
            else:
                n_t = 0.0
        
        n[t] = n_t
        q_curr = q_curr - n_t + arrivals[t]
    return n, q

# --- ORACLE WORKER (MULTI-POLICY & PHYSICALLY CONSISTENT) ---
def _oracle_worker(args):
    # PASS NUMPY ARRAYS DIRECTLY TO AVOID HUGE IPC OVERHEAD
    gas_prices, gas_ref, arrivals, c_cap, c_base, beta, alpha, lambda_d, policy_type = args
    H = len(gas_prices)
    
    # 1. INITIALIZE BASE ACTIONS
    gap = 0.0 # Default gap is 0 for heuristic/random
    if policy_type in ['expert', 'noisy']:
        n_raw, gap = solve_episode_milp(
            gas_prices=gas_prices,
            arrivals=arrivals,
            Q_initial=0.0, 
            C_cap=c_cap,
            C_base=c_base,
            beta=beta,
            alpha=alpha,
            lambda_d=lambda_d,
            gas_ref=gas_ref
        )
        if n_raw is None:
            n_raw = np.zeros(H, dtype=np.float32)
            gap = 1.0 # 100% gap if solver fails
            
        if policy_type == 'noisy':
            n_raw = n_raw + np.random.normal(0, 5.0, size=H).astype(np.float32)
            
        # Step-by-step to enforce constraints for MILP-based policies
        n_final = np.zeros(H, dtype=np.float32)
        q_traj = np.zeros(H, dtype=np.float32)
        q_current = 0.0
        for t in range(H):
            q_traj[t] = q_current
            n_t = np.clip(n_raw[t], 0, min(q_current, c_cap))
            n_final[t] = n_t
            q_current = q_current - n_t + arrivals[t]
            
    else:
        # Use JIT Optimized Simulation for Heuristic and Random
        p_idx = 2 if policy_type == 'heuristic' else 3
        n_final, q_traj = _simulate_policy_numba(p_idx, gas_prices, gas_ref, arrivals, c_cap, H)
        # q_current for reward calculation at the end
        q_current = q_traj[-1] - n_final[-1] + arrivals[-1]
        
    # ==========================================
    # COMPUTE 3-TIER REWARD (Report-Consistent Version)
    # ==========================================
    c_mar = 15000.0
    sigma = 1e9
    
    # 1. Efficiency Tier: Rewards for executing at low gas prices
    # Formula: (n * gas_ref - (c_base + c_mar * n) * gas_prices) / sigma
    execution_cost = (c_base * (n_final > 0.5) + c_mar * n_final) * gas_prices
    r_eff = (n_final * gas_ref - execution_cost) / sigma
    
    # 2. Urgency Tier: Penalty INCREASES over time (Corrected)
    remaining_q_instant = q_traj - n_final
    time_ratio = np.arange(H) / float(H)
    r_urg = (beta / sigma) * remaining_q_instant * np.exp(alpha * time_ratio)
    
    # 3. Catastrophe Tier: Final leftover penalty
    final_leftover = q_current 
    r_cat = np.zeros(H, dtype=np.float32)
    r_cat[-1] = (lambda_d / sigma) * np.maximum(0.0, final_leftover)
    
    # Total Reward
    total_reward = r_eff - r_urg - r_cat
    
    return n_final, q_traj, total_reward, gap

def process_single_file(file_path, config, is_train=False):
    print(f"\n--- Processing {Path(file_path).name} ---")
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
        
    print(f"Data loaded: {len(df):,} rows.")
    
    # 1. Compute basis feature using NUMBA
    print("Computing fundamental features with Numba Extreme Optimization...")
    log_fee = np.log(df['base_fee_per_gas'].values)
    gas_used = df['gas_used'].values
    target_gas = df['gas_limit'].values / 2.0 # 50% of gas limit
    num_lags = int(config['state']['num_lags'])
    window_size = 20
    tx_count = df['transaction_count'].values
    
    returns, volatility, utilization, lags, acceleration, surprise, backlog_pressure = _compute_features_numba(
        log_fee, gas_used, target_gas, tx_count, window_size, num_lags
    )
    
    df['log_fee'] = log_fee
    df['momentum'] = returns
    df['acceleration'] = acceleration
    df['surprise'] = surprise
    df['backlog_pressure'] = backlog_pressure
    df['volatility'] = volatility
    df['utilization'] = utilization
    for j in range(num_lags):
        df[f'lag_{j+1}'] = lags[:, j]
    
    df = df.iloc[window_size:].reset_index(drop=True)
    
    # Compute GLOBAL Gas Reference (Window=128 blocks = 1 Episode)
    print("Computing Global Gas Reference (Window=128)...")
    df['gas_reference'] = df['base_fee_per_gas'].rolling(window=128, min_periods=1).mean()
    
    bounds = None
    c_cap = config['env']['execution_capacity']
    a_scale = config['env']['arrival_scale']
    
    # 2. Find NORMALIZATION BOUNDS (TRAIN SET ONLY)
    if is_train:
        print("Extracting empirical bounds for normalization from TRAIN set...")
        max_gas_log = float(df['log_fee'].max())
        min_gas_log = float(df['log_fee'].min())
        max_volatility = float(df['volatility'].max())
        max_queue = simulate_queue_to_find_max(df, c_cap, a_scale)
        
        # New Feature Bounds
        max_momentum = float(df['momentum'].max())
        min_momentum = float(df['momentum'].min())
        max_acceleration = float(df['acceleration'].max())
        min_acceleration = float(df['acceleration'].min())
        max_surprise = float(df['surprise'].max())
        min_surprise = float(df['surprise'].min())
        max_backlog_pressure = float(df['backlog_pressure'].max())
        
        bounds = {
            "max_log_fee": max_gas_log,
            "min_log_fee": min_gas_log,
            "max_volatility": max_volatility,
            "max_queue": int(max_queue),
            "max_momentum": max_momentum,
            "min_momentum": min_momentum,
            "max_acceleration": max_acceleration,
            "min_acceleration": min_acceleration,
            "max_surprise": max_surprise,
            "min_surprise": min_surprise,
            "max_backlog_pressure": max_backlog_pressure
        }
    
    # 3. HINDSIGHT LABELING VIA MILP ORACLE
    H = int(config['env'].get('episode_length', 128))
    
    # ONLY apply overlapping to Train set. Val/Test should remain disjoint for fair evaluation.
    stride = H // 2 if is_train else H
    
    if is_train:
        print(f"Generating Overlapping Episodes (Stride={stride}) for Training...")
    else:
        print(f"Generating Disjoint Episodes (Stride={stride}) for Evaluation...")
        
    c_base = config['env'].get('C_base', 21000)
    beta = config['env'].get('urgency_beta', 0.1)
    alpha = config['env'].get('urgency_alpha', 2.0)
    lambda_d = config['env'].get('deadline_penalty', 100.0)
    
    # Pre-extract numpy arrays for blazing fast IPC
    all_gas = df['base_fee_per_gas'].values / 1e9
    all_ref = df['gas_reference'].values / 1e9
    all_arrivals = (df['transaction_count'].values * a_scale).astype(np.int64)
    
    tasks = []
    episode_starts = list(range(0, len(df) - H, stride))
    
    # Behavior Policy Mixing (Train only: 40% Expert, 30% Noisy, 20% Heuristic, 10% Random)
    policy_choices = ['expert', 'noisy', 'heuristic', 'random']
    policy_probs = [0.4, 0.3, 0.2, 0.1]
    
    if is_train:
        assigned_policies = np.random.choice(policy_choices, size=len(episode_starts), p=policy_probs)
    else:
        # Keep Val/Test 100% Expert as the Golden Baseline
        assigned_policies = ['expert'] * len(episode_starts)
    
    for i, start in enumerate(episode_starts):
        tasks.append((
            all_gas[start : start + H],
            all_ref[start : start + H],
            all_arrivals[start : start + H],
            c_cap, c_base, beta, alpha, lambda_d,
            assigned_policies[i]
        ))
        
    print(f"Total Episodes to solve: {len(tasks)} (Mix: {dict(zip(*np.unique(assigned_policies, return_counts=True)))})")
    
    # Result holders (Pre-allocated for maximum speed)
    total_episodic_rows = len(episode_starts) * H
    actions_flat = np.zeros(total_episodic_rows, dtype=np.float32)
    queues_flat = np.zeros(total_episodic_rows, dtype=np.float32)
    rewards_flat = np.zeros(total_episodic_rows, dtype=np.float32)
    gaps_flat = np.zeros(total_episodic_rows, dtype=np.float32)
    episode_ids_flat = np.zeros(total_episodic_rows, dtype=np.int32)
    terminals_flat = np.zeros(total_episodic_rows, dtype=bool)
    
    max_workers = os.cpu_count() or 4
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, (opt_n, opt_q, total_reward, gap) in enumerate(tqdm(executor.map(_oracle_worker, tasks), total=len(tasks))):
            offset = i * H
            actions_flat[offset : offset + H] = opt_n
            queues_flat[offset : offset + H] = opt_q
            rewards_flat[offset : offset + H] = total_reward
            gaps_flat[offset : offset + H] = gap
            episode_ids_flat[offset : offset + H] = i
            terminals_flat[offset + H - 1] = True # Mark the end of each episode
            
    # Vectorized Index Replication for original features
    print("Replicating state features via vectorized indexing (No copying loop)...")
    # This creates the map of original row indices for all episodes
    row_indices = np.concatenate([np.arange(start, start + H) for start in episode_starts])
    
    # Single-pass slice of the original dataframe
    final_df = df.iloc[row_indices].copy()
    
    # Plug in the newly generated episodic data
    final_df['episode_id'] = episode_ids_flat
    final_df['policy_type'] = np.repeat(assigned_policies, H)
    final_df['action'] = actions_flat
    final_df['queue_size'] = queues_flat
    final_df['reward'] = rewards_flat
    final_df['oracle_gap'] = gaps_flat
    final_df['terminal'] = terminals_flat
    final_df['time_ratio'] = (final_df.groupby('episode_id').cumcount() / (H - 1)).astype(np.float32)
    
    # Print Research Stats
    expert_gaps = gaps_flat[episode_ids_flat % 1 == 0] # Filter only relevant rows if needed
    print(f"\n--- Oracle Quality Assessment ---")
    print(f"Mean MIP Gap:   {np.mean(gaps_flat):.4%}")
    print(f"Median MIP Gap: {np.median(gaps_flat):.4%}")
    print(f"P95 MIP Gap:    {np.percentile(gaps_flat, 95):.4%}")
    
    return final_df, bounds


def build_dataset(config_path, train_path, val_path, test_path):
    config = load_config(config_path)
    exp_name = config['experiment_name']
    
    out_dir = Path(f"data/processed/{exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Building Dataset for Experiment: {exp_name} ===")
    
    # Process Train Set (Extract Bounds)
    train_df, bounds = process_single_file(train_path, config, is_train=True)
    train_df.to_parquet(out_dir / "train.parquet", engine='pyarrow', index=False)
    
    metadata = {
        "experiment_name": exp_name,
        "config": config,
        "normalization_bounds": bounds,
        "dataset_stats": {
            "train_rows": len(train_df)
        }
    }
    
    # Process Val Set (No Bounds Extraction)
    if val_path:
        val_df, _ = process_single_file(val_path, config, is_train=False)
        val_df.to_parquet(out_dir / "val.parquet", engine='pyarrow', index=False)
        metadata["dataset_stats"]["val_rows"] = len(val_df)
        
    # Process Test Set (No Bounds Extraction)
    if test_path:
        test_df, _ = process_single_file(test_path, config, is_train=False)
        test_df.to_parquet(out_dir / "test.parquet", engine='pyarrow', index=False)
        metadata["dataset_stats"]["test_rows"] = len(test_df)
    
    # Save Normalization Metadata
    print("\nSaving Metadata...")
    meta_path = out_dir / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\n Build complete! All datasets and metadata saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--train", type=str, required=True, help="Path to raw train data")
    parser.add_argument("--val", type=str, required=False, help="Path to raw val data (optional)")
    parser.add_argument("--test", type=str, required=False, help="Path to raw test data (optional)")
    args = parser.parse_args()
    
    build_dataset(args.config, args.train, args.val, args.test)
