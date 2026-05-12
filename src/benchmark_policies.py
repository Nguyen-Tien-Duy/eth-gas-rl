import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from tqdm import tqdm

# Add project root to path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.milp_oracle import solve_episode_milp

def run_benchmark(exp_name):
    print(f"=== Policy Benchmarking on Validation Set: {exp_name} ===")
    
    data_dir = Path(f"data/processed/{exp_name}")
    val_path = data_dir / "val.parquet"
    config_path = Path("configs/exp_benchmark.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    df = pd.read_parquet(val_path)
    episodes = df['episode_id'].unique()
    
    # Environment Params (Standardize with Config keys)
    H = config['env'].get('horizon', 128)
    c_cap = config['env'].get('execution_capacity', 100)
    c_base = config['env'].get('C_base', 21000.0)
    beta = config['rl'].get('urgency_beta', 0.1)
    alpha = config['rl'].get('urgency_alpha', 2.0)
    lambda_d = config['rl'].get('deadline_penalty', 500.0)
    
    results = []

    print(f"Running simulations over {len(episodes)} episodes...")
    for ep_id in tqdm(episodes):
        ep_data = df[df['episode_id'] == ep_id]
        gas_prices = ep_data['base_fee_per_gas'].values / 1e9
        gas_ref = ep_data['gas_reference'].values / 1e9
        arrivals = ep_data['transaction_count'].values * config['env']['arrival_scale']
        arrivals = arrivals.astype(np.int64)
        
        # 1. Expert Policy (MILP)
        n_exp, _ = solve_episode_milp(gas_prices, arrivals, 0.0, c_cap, c_base, beta, alpha, lambda_d, gas_ref)
        if n_exp is None: n_exp = np.zeros(H)
        
        # 2. Noisy Expert (Expert + Sigma=5.0 noise)
        n_noisy = np.clip(n_exp + np.random.normal(0, 5.0, size=H), 0, c_cap)
        
        # 3. Greedy Policy (Flush everything immediately)
        n_greedy = np.zeros(H)
        q_g = 0.0
        for t in range(H):
            n_t = min(q_g, c_cap)
            n_greedy[t] = n_t
            q_g = q_g - n_t + arrivals[t]
            
        # 4. Heuristic Policy (Panic + Cheap)
        n_heur = np.zeros(H)
        q_h = 0.0
        for t in range(H):
            panic = (t > 0.9 * H)
            cheap = (gas_prices[t] < gas_ref[t])
            n_t = min(q_h, c_cap) if (panic or cheap) else 0.0
            n_heur[t] = n_t
            q_h = q_h - n_t + arrivals[t]
            
        # 5. Random Policy
        n_rand = np.random.uniform(0, c_cap, size=H)

        # Helper to compute Business Metrics (Money & SLA)
        def get_business_metrics(n_array):
            q_sim = 0.0
            total_savings = 0.0
            for t in range(H):
                # Enforce n_t <= q_t
                n_t = min(n_array[t], q_sim)
                # Savings = n * (gas_ref - gas_price) - FixedCost
                savings = n_t * (gas_ref[t] - gas_prices[t])
                fixed = (c_base / 1e9) * gas_prices[t] * (n_t > 0.5)
                
                total_savings += (savings - fixed)
                q_sim = q_sim - n_t + arrivals[t]
            
            return total_savings, q_sim

        # Run for all
        res_exp = get_business_metrics(n_exp)
        res_noisy = get_business_metrics(n_noisy)
        res_greedy = get_business_metrics(n_greedy)
        res_heur = get_business_metrics(n_heur)
        res_rand = get_business_metrics(n_rand)

        results.append({
            'Expert_Gwei': res_exp[0], 'Expert_SLA': res_exp[1],
            'Noisy_Gwei': res_noisy[0], 'Noisy_SLA': res_noisy[1],
            'Greedy_Gwei': res_greedy[0], 'Greedy_SLA': res_greedy[1],
            'Heuristic_Gwei': res_heur[0], 'Heuristic_SLA': res_heur[1],
            'Random_Gwei': res_rand[0], 'Random_SLA': res_rand[1]
        })

    res_df = pd.DataFrame(results)
    
    # 1. Money Summary
    gwei_cols = [c for c in res_df.columns if 'Gwei' in c]
    mean_savings = res_df[gwei_cols].mean().sort_values(ascending=False)
    
    # 2. SLA Summary
    sla_cols = [c for c in res_df.columns if 'SLA' in c]
    mean_backlog = res_df[sla_cols].mean().sort_values()

    print("\n--- [MONEY] Net Savings (Gwei per Episode) ---")
    print(mean_savings)
    
    print("\n--- [SLA] Mean Backlog at Deadline (Lower is better) ---")
    print(mean_backlog)

    # Plotting 2-panel
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    mean_savings.plot(kind='bar', ax=ax1, color='green', alpha=0.7)
    ax1.set_title("Net Gas Savings (Gwei)")
    ax1.set_ylabel("Gwei")
    ax1.grid(axis='y', linestyle='--')

    mean_backlog.plot(kind='bar', ax=ax2, color='red', alpha=0.7)
    ax2.set_title("SLA Violations (Backlog at H)")
    ax2.set_ylabel("Tx Count")
    ax2.grid(axis='y', linestyle='--')

    plt.tight_layout()
    out_img = data_dir / "business_benchmark.png"
    plt.savefig(out_img)
    print(f"\n Business benchmark plot saved to: {out_img}")

if __name__ == "__main__":
    run_benchmark("exp_benchmark")
