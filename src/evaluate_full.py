import os
import yaml
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.environment import EthGasEnv
from src.algorithms.cql import CQL
from src.algorithms.iql import IQL
from src.algorithms.td3_bc import TD3_BC
from src.algorithms.awac import AWAC
from src.algorithms.bcq import BCQ

def evaluate_full():
    # 1. Load Config
    config_path = "configs/exp_benchmark.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    exp_name = config['experiment_name']
    env_config = config['env']
    env_slug = f"H{env_config['horizon']}_C{env_config['execution_capacity']}_A{env_config['arrival_scale']}"
    device = torch.device("cpu") # Eval on CPU is enough

    # 2. Load Validation Data
    val_path = f"data/processed/{exp_name}/val.parquet"
    if not os.path.exists(val_path):
        print(f"❌ Validation data not found at {val_path}")
        return
    
    val_df = pd.read_parquet(val_path)
    eval_env = EthGasEnv(config, trace_df=val_df)
    
    # 3. Discover Trained Algorithms
    checkpoint_root = f"checkpoints/{exp_name}/{env_slug}"
    if not os.path.exists(checkpoint_root):
        print(f"❌ No checkpoints found at {checkpoint_root}")
        return
    
    algos_to_test = [d for d in os.listdir(checkpoint_root) if os.path.isdir(os.path.join(checkpoint_root, d))]
    print(f"🔍 Found {len(algos_to_test)} algorithms to evaluate: {algos_to_test}")

    results = []

    for algo_name in algos_to_test:
        print(f"\n🚀 Evaluating {algo_name.upper()} on FULL Validation Set...")
        
        # Initialize Architecture
        if algo_name == 'iql':
            algo = IQL(state_dim=9, action_dim=1, device=device)
        elif algo_name == 'td3_bc':
            algo = TD3_BC(state_dim=9, action_dim=1, device=device)
        elif algo_name == 'awac':
            algo = AWAC(state_dim=9, action_dim=1, device=device)
        elif algo_name == 'bcq':
            algo = BCQ(state_dim=9, action_dim=1, device=device)
        else:
            algo = CQL(state_dim=9, action_dim=1, device=device)

        # Load Weights
        model_path = f"{checkpoint_root}/{algo_name}/best_model.pt"
        if not os.path.exists(model_path):
            print(f"⚠️ Skip {algo_name}: best_model.pt not found.")
            continue
        
        algo.load(model_path)

        # Full Sequential Evaluation (Non-overlapping episodes)
        horizon = env_config['horizon']
        total_steps = len(val_df)
        num_episodes = total_steps // horizon
        
        ep_rewards = []
        ep_savings = []
        ep_backlogs = []
        
        # We walk through the entire dataset block by block (sequential)
        for i in tqdm(range(num_episodes), desc=f"Scanning Val Set"):
            start_idx = i * horizon
            # Manually reset env to specific start_idx for full coverage
            obs, _ = eval_env.reset()
            eval_env.current_step = start_idx
            eval_env.queue = 0.0 # Reset queue for each independent chunk
            
            done = False
            steps_in_ep = 0
            ep_reward = 0
            ep_savings_val = 0
            
            while not done and steps_in_ep < horizon:
                action = algo.select_action(obs)
                obs, reward, done, _, info = eval_env.step(action)
                ep_reward += reward
                ep_savings_val += info['savings']
                steps_in_ep += 1
                if eval_env.current_step >= total_steps - 1:
                    break
            
            ep_rewards.append(ep_reward)
            ep_savings.append(ep_savings_val)
            ep_backlogs.append(eval_env.queue)

        # Aggregate Results
        results.append({
            "Algorithm": algo_name.upper(),
            "Total_Savings_Gwei": np.sum(ep_savings),
            "Mean_Reward": np.mean(ep_rewards),
            "Mean_Backlog": np.mean(ep_backlogs),
            "Max_Backlog": np.max(ep_backlogs),
            "Efficiency_Score": np.sum(ep_savings) / (np.mean(ep_backlogs) + 1e-9)
        })

    # 4. Final Report
    report_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("🏆 FINAL BENCHMARK REPORT (FULL VALIDATION SET)")
    print("="*60)
    print(report_df.to_markdown(index=False))
    print("="*60)
    
    # Save to CSV
    report_df.to_csv(f"{checkpoint_root}/final_benchmark_report.csv", index=False)
    print(f"✅ Report saved to {checkpoint_root}/final_benchmark_report.csv")

if __name__ == "__main__":
    evaluate_full()
