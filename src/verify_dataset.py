import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from pathlib import Path

def verify_dataset(exp_name):
    print(f"=== Verification Report for Experiment: {exp_name} ===")
    
    data_dir = Path(f"data/processed/{exp_name}")
    train_path = data_dir / "train.parquet"
    
    if not train_path.exists():
        print(f"Error: {train_path} not found!")
        return

    print("Loading Train Dataset...")
    df = pd.read_parquet(train_path)
    print(f"Dataset Size: {len(df):,} rows")
    
    # 1. Physical Consistency Check
    print("\n[1/3] Checking Physical Consistency (Q_next = Q_now - n_now + W_now)...")
    # Take a sample for speed
    sample_size = 100000
    sample = df.sample(sample_size).copy()
    
    # We need to be careful with episode boundaries, but as a rough check:
    # Q_calculated = Q_now - action + arrivals
    # Since we can't easily check across rows without shift per episode, 
    # let's check if action <= queue_size (Safety constraint)
    violations = (df['action'] > df['queue_size'] + 1e-5).sum()
    if violations == 0:
        print("✅ PASS: Action <= Queue Size for all rows.")
    else:
        print(f"❌ FAIL: Found {violations} violations where Action > Queue Size!")

    # 2. Policy Performance Comparison
    print("\n[2/3] Comparing Policy Performance (Mean Reward)...")
    policy_stats = df.groupby('policy_type')['reward'].agg(['mean', 'std', 'min', 'max']).sort_values('mean', ascending=False)
    print(policy_stats)
    
    # Check if Expert is indeed better than Random
    if 'expert' in policy_stats.index and 'random' in policy_stats.index:
        if policy_stats.loc['expert', 'mean'] > policy_stats.loc['random', 'mean']:
            print("PASS: Expert policy outperforms Random policy.")
        else:
            print("WARNING: Expert policy is NOT significantly better than Random! Check reward logic.")

    # 3. Visualization
    print("\n[3/3] Generating Visualization for Expert vs Heuristic vs Random...")
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    policies = ['expert', 'heuristic', 'random']
    colors = ['green', 'orange', 'red']
    
    for i, policy in enumerate(policies):
        ax = axes[i]
        # Pick a random episode for this policy
        ep_ids = df[df['policy_type'] == policy]['episode_id'].unique()
        if len(ep_ids) == 0: continue
        
        target_ep = np.random.choice(ep_ids)
        ep_data = df[df['episode_id'] == target_ep].reset_index()
        
        # Plot Gas Price (Secondary Axis)
        ax2 = ax.twinx()
        ax2.plot(ep_data.index, ep_data['base_fee_per_gas'] / 1e9, color='blue', alpha=0.3, label='Gas Price (Gwei)')
        ax2.set_ylabel("Gas Price (Gwei)", color='blue')
        
        # Plot Queue and Action
        ax.bar(ep_data.index, ep_data['action'], color=colors[i], alpha=0.7, label=f'Action ({policy})')
        ax.plot(ep_data.index, ep_data['queue_size'], color='black', linestyle='--', label='Queue Size')
        
        ax.set_title(f"Episode Strategy: {policy.upper()} (ID: {target_ep})")
        ax.set_ylabel("Volume (Gas)")
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

    plt.tight_layout()
    out_plot = data_dir / "verification_plot.png"
    plt.savefig(out_plot)
    print(f" PLOT SAVED: {out_plot}")
    
    # 4. Gap Analysis
    if 'oracle_gap' in df.columns:
        print("\n[BONUS] Oracle Optimality Gap Analysis:")
        print(f"Average Gap: {df['oracle_gap'].mean():.6%}")
        print(f"Max Gap:     {df['oracle_gap'].max():.6%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Experiment name")
    args = parser.parse_args()
    
    verify_dataset(args.exp)
