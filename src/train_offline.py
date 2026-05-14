import os
import sys
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
import wandb
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time

# Add project root to path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import get_offline_loader
from src.algorithms.cql import CQL
from src.algorithms.iql import IQL
from src.algorithms.td3_bc import TD3_BC
from src.algorithms.awac import AWAC
from src.algorithms.bcq import BCQ
from src.environment import EthGasEnv


def _setup_distributed():
    """Initialize DDP if launched with torchrun.

    Returns: (is_ddp, local_rank, rank, world_size, is_main)
    """
    if "LOCAL_RANK" not in os.environ or not torch.cuda.is_available():
        return False, 0, 0, 1, True

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return True, local_rank, rank, world_size, rank == 0


def _cleanup_distributed(is_ddp: bool):
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


def _wrap_algo_ddp(algo, algo_name: str, local_rank: int):
    """Wrap trainable modules in DDP for gradient sync."""
    algo_name = algo_name.lower()
    if algo_name == "bcq":
        attrs = ["vae", "actor", "q1", "q2"]
    elif algo_name == "iql":
        attrs = ["actor", "q1", "q2", "vf"]
    else:
        attrs = ["actor", "q1", "q2"]

    for attr in attrs:
        m = getattr(algo, attr)
        setattr(
            algo,
            attr,
            DDP(m, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False),
        )


def _get_eval_episode_ids(val_df: pd.DataFrame, max_episodes: int) -> np.ndarray:
    if "episode_id" not in val_df.columns:
        raise ValueError("val.parquet missing 'episode_id' column")
    episode_ids = np.sort(val_df["episode_id"].unique())
    if len(episode_ids) == 0:
        raise ValueError("No episodes found in val.parquet")
    if max_episodes and len(episode_ids) > max_episodes:
        episode_ids = episode_ids[:max_episodes]
    return episode_ids


def _make_algo(algo_name: str, config: dict, device: torch.device, use_amp: bool):
    algo_name = algo_name.lower()
    if algo_name == 'iql':
        return IQL(
            state_dim=9,
            action_dim=1,
            device=device,
            expectile=config['rl'].get('iql_expectile', 0.7),
            beta=config['rl'].get('iql_beta', 3.0),
            use_amp=use_amp,
        )
    if algo_name == 'td3_bc':
        return TD3_BC(
            state_dim=9,
            action_dim=1,
            device=device,
            alpha=config['rl'].get('td3_bc_alpha', 2.5),
            use_amp=use_amp,
        )
    if algo_name == 'awac':
        return AWAC(
            state_dim=9,
            action_dim=1,
            device=device,
            beta=config['rl'].get('awac_beta', 1.0),
            use_amp=use_amp,
        )
    if algo_name == 'bcq':
        return BCQ(
            state_dim=9,
            action_dim=1,
            device=device,
            phi=config['rl'].get('bcq_phi', 0.05),
            use_amp=use_amp,
        )
    return CQL(
        state_dim=9,
        action_dim=1,
        device=device,
        alpha=config['rl'].get('cql_alpha', 1.0),
        use_amp=use_amp,
    )


def _aggregate_metrics(metrics_list):
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}


def _train_one_epoch(algo, train_loader, algo_name: str, epoch: int, is_ddp: bool, is_main: bool):
    if is_ddp and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)

    train_iter = train_loader
    if is_main:
        train_iter = tqdm(train_loader, desc=f"{algo_name.upper()} | Epoch {epoch+1}")

    metrics_list = []
    processed = 0
    start_ts = time.time()
    for batch in train_iter:
        metrics = algo.update(batch)
        # count samples in this batch (observations first dim)
        try:
            batch_n = int(batch['observations'].shape[0])
        except Exception:
            batch_n = 0
        processed += batch_n
        if is_main:
            metrics_list.append(metrics)

    elapsed = max(1e-6, time.time() - start_ts)
    local_sps = processed / elapsed
    # If distributed, estimate global samples/sec by multiplying by world size
    if is_ddp and dist.is_initialized():
        try:
            world_size = dist.get_world_size()
        except Exception:
            world_size = 1
    else:
        world_size = 1

    global_sps_est = local_sps * world_size
    if is_main:
        print(
            f"[PERF] Epoch {epoch+1}: processed={processed} samples, "
            f"local_sps={local_sps:.2f} samples/s, approx_global_sps={global_sps_est:.2f} samples/s"
        )

    return _aggregate_metrics(metrics_list)


def _evaluate(algo, eval_env: EthGasEnv, episode_ids: np.ndarray):
    eval_rewards = []
    eval_backlog = []
    eval_savings = []

    for ep_id in episode_ids:
        obs, _ = eval_env.reset(options={"episode_id": int(ep_id)})
        done = False
        ep_reward = 0.0
        ep_savings = 0.0
        while not done:
            action = algo.select_action(obs)
            action = np.asarray(action, dtype=np.float32).reshape(1,)
            obs, reward, done, _, info = eval_env.step(action)
            ep_reward += float(reward)
            ep_savings += float(info.get('savings', 0.0))

        eval_rewards.append(ep_reward)
        eval_backlog.append(float(eval_env.queue))
        eval_savings.append(ep_savings)

    return {
        "eval/mean_reward": float(np.mean(eval_rewards)),
        "eval/mean_backlog": float(np.mean(eval_backlog)),
        "eval/mean_savings": float(np.mean(eval_savings)),
    }


def _broadcast_stop(stop_now: bool, is_ddp: bool, device: torch.device) -> bool:
    if not is_ddp:
        return stop_now
    stop_tensor = torch.tensor(1 if stop_now else 0, device=device)
    dist.broadcast(stop_tensor, src=0)
    return stop_tensor.item() == 1


def _main_epoch_eval_log_and_ckpt(
    *,
    algo,
    algo_name: str,
    epoch: int,
    eval_env: EthGasEnv,
    eval_episode_ids: np.ndarray,
    avg_metrics: dict,
    checkpoint_dir: str,
    run,
    history: list,
    best_reward: float,
    no_improvement_count: int,
    patience: int,
):
    eval_metrics = _evaluate(algo, eval_env, eval_episode_ids)
    mean_eval_reward = float(eval_metrics["eval/mean_reward"])

    wandb_log = {
        "epoch": epoch + 1,
        **eval_metrics,
        **{f"train/{k}": v for k, v in avg_metrics.items()},
    }
    if run is not None:
        wandb.log(wandb_log)

    history.append(wandb_log)
    pd.DataFrame(history).to_csv(f"{checkpoint_dir}/metrics.csv", index=False)
    print(
        f"[{algo_name.upper()}] Epoch {epoch+1}: Reward={mean_eval_reward:.2f}, "
        f"Savings={eval_metrics['eval/mean_savings']:.2f}"
    )

    stop_now = False
    if mean_eval_reward > best_reward:
        best_reward = mean_eval_reward
        no_improvement_count = 0
        algo.save(f"{checkpoint_dir}/best_model.pt")
        print(f" New Best Model Saved! (Reward: {best_reward:.2f})")
    else:
        no_improvement_count += 1
        if no_improvement_count >= patience:
            print(f" Early Stopping triggered for {algo_name.upper()}.")
            stop_now = True

    if (epoch + 1) % 10 == 0:
        algo.save(f"{checkpoint_dir}/{algo_name}_epoch_{epoch+1}.pt")

    return stop_now, best_reward, no_improvement_count


def _run_epochs(
    *,
    algo,
    algo_name: str,
    train_loader,
    args: argparse.Namespace,
    is_ddp: bool,
    is_main: bool,
    device: torch.device,
    eval_env: EthGasEnv | None,
    eval_episode_ids: np.ndarray | None,
    checkpoint_dir: str,
    run,
):
    best_reward = -np.inf
    no_improvement_count = 0
    history = []

    for epoch in range(args.epochs):
        avg_metrics = _train_one_epoch(algo, train_loader, algo_name, epoch, is_ddp, is_main)

        stop_now = False
        if is_main:
            stop_now, best_reward, no_improvement_count = _main_epoch_eval_log_and_ckpt(
                algo=algo,
                algo_name=algo_name,
                epoch=epoch,
                eval_env=eval_env,
                eval_episode_ids=eval_episode_ids,
                avg_metrics=avg_metrics,
                checkpoint_dir=checkpoint_dir,
                run=run,
                history=history,
                best_reward=best_reward,
                no_improvement_count=no_improvement_count,
                patience=args.patience,
            )

        if _broadcast_stop(stop_now, is_ddp, device):
            break


def _train_one_algorithm(
    *,
    algo_name: str,
    config: dict,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
    is_ddp: bool,
    local_rank: int,
    rank: int,
    world_size: int,
    is_main: bool,
    exp_name: str,
    env_slug: str,
):
    algo_name = algo_name.lower()
    if is_main:
        print(f"\n{'='*50}\nTRAINING ALGORITHM: {algo_name.upper()}\n{'='*50}")

    run = None
    if is_main:
        run = wandb.init(
            project="eth-gas-rl",
            name=f"{algo_name}-{env_slug}",
            group=exp_name,
            config=config,
            reinit=True,
        )

    train_path = f"data/processed/{exp_name}/train.parquet"
    val_path = f"data/processed/{exp_name}/val.parquet"
    metadata_path = f"data/processed/{exp_name}/metadata.json"
    checkpoint_dir = f"checkpoints/{exp_name}/{env_slug}/{algo_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_loader = get_offline_loader(
        train_path,
        metadata_path,
        batch_size=args.batch_size,
        shuffle=True,
        distributed=is_ddp,
        rank=rank,
        world_size=world_size,
    )

    algo = _make_algo(algo_name, config, device, use_amp)
    if is_ddp:
        _wrap_algo_ddp(algo, algo_name, local_rank)

    val_df = None
    eval_env = None
    eval_episode_ids = None
    if is_main:
        val_df = pd.read_parquet(val_path)
        eval_env = EthGasEnv(config, trace_df=val_df)
        eval_episode_ids = _get_eval_episode_ids(val_df, args.eval_episodes)

    _run_epochs(
        algo=algo,
        algo_name=algo_name,
        train_loader=train_loader,
        args=args,
        is_ddp=is_ddp,
        is_main=is_main,
        device=device,
        eval_env=eval_env,
        eval_episode_ids=eval_episode_ids,
        checkpoint_dir=checkpoint_dir,
        run=run,
    )

    if run is not None:
        run.finish()

    import gc
    del algo, train_loader
    if is_main:
        del eval_env, val_df, eval_episode_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if is_ddp:
        dist.barrier()

    if is_main:
        print(f" Finished {algo_name.upper()}. Memory cleared.\n")

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/exp_benchmark.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP (autocast + GradScaler)")
    args = parser.parse_args()

    is_ddp, local_rank, rank, world_size, is_main = _setup_distributed()

    # Avoid CPU thread oversubscription under DDP
    base_threads = os.cpu_count() // 2
    threads = max(1, base_threads // max(1, world_size))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Load Configuration
    config_path = args.config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    exp_name = config['experiment_name']
    algorithms = config['rl'].get('algorithms', ['cql'])
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if is_ddp else "cuda")
    else:
        device = torch.device("cpu")

    use_amp = (not args.no_amp) and device.type == "cuda"

    # Create a unique slug for the environment configuration
    env_config = config['env']
    env_slug = f"H{env_config['horizon']}_C{env_config['execution_capacity']}_A{env_config['arrival_scale']}"
    
    if is_main:
        print(f"Starting Benchmark on {len(algorithms)} algorithms: {algorithms}")

    for algo_name in algorithms:
        _train_one_algorithm(
            algo_name=algo_name,
            config=config,
            args=args,
            device=device,
            use_amp=use_amp,
            is_ddp=is_ddp,
            local_rank=local_rank,
            rank=rank,
            world_size=world_size,
            is_main=is_main,
            exp_name=exp_name,
            env_slug=env_slug,
        )

    _cleanup_distributed(is_ddp)

if __name__ == "__main__":
    train()
