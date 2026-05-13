import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
import wandb
from tqdm import tqdm

# Add project root to path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import get_offline_loader
from src.algorithms.cql import CQL
from src.algorithms.iql import IQL
from src.algorithms.td3_bc import TD3_BC
from src.algorithms.awac import AWAC
from src.algorithms.bcq import BCQ
from src.environment import EthGasEnv

def train():
    # 1. Hardware Exploitation (Use 4 real cores for i5-11300H)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    
    # 2. Load Configuration
    config_path = "configs/exp_benchmark.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    exp_name = config['experiment_name']
    algorithms = config['rl'].get('algorithms', ['cql'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a unique slug for the environment configuration
    env_config = config['env']
    env_slug = f"H{env_config['horizon']}_C{env_config['execution_capacity']}_A{env_config['arrival_scale']}"
    
    print(f"Starting Benchmark on {len(algorithms)} algorithms: {algorithms}")

    for algo_name in algorithms:
        algo_name = algo_name.lower()
        print(f"\n{'='*50}\nTRAINING ALGORITHM: {algo_name.upper()}\n{'='*50}")

        # 3. Initialize WandB for this specific run
        run = wandb.init(
            project="eth-gas-rl",
            name=f"{algo_name}-{env_slug}",
            group=exp_name,
            config=config,
            reinit=True # Important for sequential runs
        )

        # 4. Paths & Setup
        train_path = f"data/processed/{exp_name}/train.parquet"
        val_path = f"data/processed/{exp_name}/val.parquet"
        metadata_path = f"data/processed/{exp_name}/metadata.json"
        checkpoint_dir = f"checkpoints/{exp_name}/{env_slug}/{algo_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 5. Data Loader
        train_loader = get_offline_loader(train_path, metadata_path, batch_size=1024)
        
        # 6. Initialize Algorithm (CORL Style)
        if algo_name == 'iql':
            algo = IQL(
                state_dim=9, action_dim=1, device=device,
                expectile=config['rl'].get('iql_expectile', 0.7),
                beta=config['rl'].get('iql_beta', 3.0)
            )
        elif algo_name == 'td3_bc':
            algo = TD3_BC(
                state_dim=9, action_dim=1, device=device,
                alpha=config['rl'].get('td3_bc_alpha', 2.5)
            )
        elif algo_name == 'awac':
            algo = AWAC(
                state_dim=9, action_dim=1, device=device,
                beta=config['rl'].get('awac_beta', 1.0)
            )
        elif algo_name == 'bcq':
            algo = BCQ(
                state_dim=9, action_dim=1, device=device,
                phi=config['rl'].get('bcq_phi', 0.05)
            )
        else:
            algo = CQL(
                state_dim=9, action_dim=1, device=device,
                alpha=config['rl'].get('cql_alpha', 1.0)
            )

        # 7. Evaluation Environment
        val_df = pd.read_parquet(val_path)
        eval_env = EthGasEnv(config, trace_df=val_df)

        # 8. Main Training Loop
        epochs = 100
        patience = 15
        best_reward = -np.inf
        no_improvement_count = 0
        history = []
        
        for epoch in range(epochs):
            algo_metrics = []
            
            # Training Phase
            for batch in tqdm(train_loader, desc=f"{algo_name.upper()} | Epoch {epoch+1}"):
                metrics = algo.update(batch)
                algo_metrics.append(metrics)
            
            # Aggregate Metrics
            avg_metrics = {k: np.mean([m[k] for m in algo_metrics]) for k in algo_metrics[0].keys()}
            
            # Evaluation Phase
            eval_rewards = []
            eval_backlog = []
            eval_savings = []
            
            # Use FIXED SEEDS for evaluation to ensure FAIRNESS across algorithms
            for i in range(100):
                obs, _ = eval_env.reset(seed=i)
                done = False
                ep_reward = 0
                ep_savings = 0
                while not done:
                    action = algo.select_action(obs)
                    obs, reward, done, _, info = eval_env.step(action)
                    ep_reward += reward
                    ep_savings += info['savings']
                
                eval_rewards.append(ep_reward)
                eval_backlog.append(eval_env.queue)
                eval_savings.append(ep_savings)

            mean_eval_reward = np.mean(eval_rewards)
            mean_eval_backlog = np.mean(eval_backlog)
            mean_eval_savings = np.mean(eval_savings)
            
            # 9. LOG TO WANDB
            wandb_log = {
                "epoch": epoch + 1,
                "eval/mean_reward": mean_eval_reward,
                "eval/mean_backlog": mean_eval_backlog,
                "eval/mean_savings": mean_eval_savings,
                **{f"train/{k}": v for k, v in avg_metrics.items()}
            }
            wandb.log(wandb_log)
            
            # 9.5 LOCAL CSV LOGGING (For custom plotting)
            history.append(wandb_log)
            pd.DataFrame(history).to_csv(f"{checkpoint_dir}/metrics.csv", index=False)
            
            print(f"[{algo_name.upper()}] Epoch {epoch+1}: Reward={mean_eval_reward:.2f}, Savings={mean_eval_savings:.2f}")
            
            # 10. Early Stopping & Best Model Saving
            if mean_eval_reward > best_reward:
                best_reward = mean_eval_reward
                no_improvement_count = 0
                algo.save(f"{checkpoint_dir}/best_model.pt")
                print(f" New Best Model Saved! (Reward: {best_reward:.2f})")
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= patience:
                print(f" Early Stopping triggered for {algo_name.upper()}.")
                break
                
            # Periodic Save
            if (epoch + 1) % 10 == 0:
                algo.save(f"{checkpoint_dir}/{algo_name}_epoch_{epoch+1}.pt")

        # Cleanup for next algorithm
        run.finish()
        import gc
        del algo, train_loader, eval_env, val_df
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f" Finished {algo_name.upper()}. Memory cleared.\n")

if __name__ == "__main__":
    train()
