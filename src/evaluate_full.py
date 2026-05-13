import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment import EthGasEnv
from src.algorithms.cql import CQL
from src.algorithms.iql import IQL
from src.algorithms.td3_bc import TD3_BC
from src.algorithms.awac import AWAC
from src.algorithms.bcq import BCQ


def _load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _make_env_slug(env_config: dict) -> str:
    return f"H{env_config['horizon']}_C{env_config['execution_capacity']}_A{env_config['arrival_scale']}"


def _discover_algorithms(checkpoint_root: str) -> list:
    if not os.path.exists(checkpoint_root):
        return []
    return [
        d
        for d in os.listdir(checkpoint_root)
        if os.path.isdir(os.path.join(checkpoint_root, d))
    ]


def _make_algo(algo_name: str, device: torch.device):
    algo_name = algo_name.lower()
    if algo_name == 'iql':
        return IQL(state_dim=9, action_dim=1, device=device)
    if algo_name == 'td3_bc':
        return TD3_BC(state_dim=9, action_dim=1, device=device)
    if algo_name == 'awac':
        return AWAC(state_dim=9, action_dim=1, device=device)
    if algo_name == 'bcq':
        return BCQ(state_dim=9, action_dim=1, device=device)
    return CQL(state_dim=9, action_dim=1, device=device)


def _get_episode_ids(val_df: pd.DataFrame) -> np.ndarray:
    if 'episode_id' not in val_df.columns:
        raise ValueError("Validation parquet missing 'episode_id'.")
    episode_ids = np.sort(val_df['episode_id'].unique())
    if len(episode_ids) == 0:
        raise ValueError("No episodes found in validation set.")
    return episode_ids


def _eval_algo_on_episodes(algo, eval_env: EthGasEnv, episode_ids: np.ndarray, algo_label: str) -> dict:
    ep_rewards = []
    ep_savings = []
    ep_backlogs = []

    for ep_id in tqdm(episode_ids, desc="Scanning Val Episodes"):
        obs, _ = eval_env.reset(options={"episode_id": int(ep_id)})
        done = False
        ep_reward = 0.0
        ep_savings_val = 0.0

        while not done:
            action = algo.select_action(obs)
            if np.isscalar(action):
                action = np.array([action], dtype=np.float32)
            else:
                action = np.asarray(action, dtype=np.float32).reshape(1,)

            obs, reward, done, _, info = eval_env.step(action)
            ep_reward += float(reward)
            ep_savings_val += float(info.get('savings', 0.0))

        ep_rewards.append(ep_reward)
        ep_savings.append(ep_savings_val)
        ep_backlogs.append(float(eval_env.queue))

    return {
        "Algorithm": algo_label,
        "Total_Savings_Gwei": float(np.sum(ep_savings)),
        "Mean_Reward": float(np.mean(ep_rewards)),
        "Mean_Backlog": float(np.mean(ep_backlogs)),
        "Max_Backlog": float(np.max(ep_backlogs)),
        "Efficiency_Score": float(np.sum(ep_savings) / (np.mean(ep_backlogs) + 1e-9)),
    }

def evaluate_full():
    config_path = "configs/exp_benchmark.yaml"
    config = _load_config(config_path)
    
    exp_name = config['experiment_name']
    env_config = config['env']
    env_slug = _make_env_slug(env_config)
    device = torch.device("cpu") # Eval on CPU is enough

    # 2. Load Validation Data
    val_path = f"data/processed/{exp_name}/val.parquet"
    if not os.path.exists(val_path):
        print(f" Validation data not found at {val_path}")
        return
    
    val_df = pd.read_parquet(val_path)
    eval_env = EthGasEnv(config, trace_df=val_df)
    
    checkpoint_root = f"checkpoints/{exp_name}/{env_slug}"
    algos_to_test = _discover_algorithms(checkpoint_root)
    if not algos_to_test:
        print(f"❌ No checkpoints found at {checkpoint_root}")
        return
    print(f"🔍 Found {len(algos_to_test)} algorithms to evaluate: {algos_to_test}")

    try:
        episode_ids = _get_episode_ids(val_df)
    except ValueError as e:
        print(f"❌ {e}")
        return

    results = []

    for algo_name in algos_to_test:
        print(f"\n🚀 Evaluating {algo_name.upper()} on FULL Validation Set...")
        
        algo = _make_algo(algo_name, device=device)

        # Load Weights
        model_path = f"{checkpoint_root}/{algo_name}/best_model.pt"
        if not os.path.exists(model_path):
            print(f"⚠️ Skip {algo_name}: best_model.pt not found.")
            continue
        
        algo.load(model_path)

        results.append(_eval_algo_on_episodes(algo, eval_env, episode_ids, algo_name.upper()))

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
