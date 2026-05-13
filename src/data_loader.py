import torch
import pandas as pd
import numpy as np
import json
import gc
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class GasOfflineDataset(Dataset):
    """
    Memory-optimized PyTorch Dataset for Offline RL.
    """
    def __init__(self, parquet_path, metadata_path, context_length=1):
        self.parquet_path = parquet_path
        self.context_length = context_length
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.bounds = self.metadata['normalization_bounds']
        
        print(f"Loading dataset from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        self.num_rows = len(self.df)
        
        self._prepare_features()
        
        # CRITICAL: Delete raw dataframe to free up GBs of RAM
        del self.df
        gc.collect()

    def _prepare_features(self):
        # 1. Normalize Base Fee (Min-Max)
        log_fee = self.df['log_fee'].values.astype(np.float32)
        norm_fee = (log_fee - self.bounds['min_log_fee']) / (self.bounds['max_log_fee'] - self.bounds['min_log_fee'] + 1e-9)
        
        # 2. Normalize Queue
        norm_queue = (self.df['queue_size'].values / (self.bounds['max_queue'] + 1e-9)).astype(np.float32)
        
        # 3. Normalize Volatility
        norm_vol = (self.df['volatility'].values / (self.bounds['max_volatility'] + 1e-9)).astype(np.float32)
        
        # 4. Lags (lags_1 to lags_5 in builder.py)
        num_lags = self.metadata['config']['state']['num_lags']
        lag_cols = [f'lag_{i+1}' for i in range(num_lags)]
        lags = self.df[lag_cols].values.astype(np.float32)
        norm_lags = (lags - self.bounds['min_log_fee']) / (self.bounds['max_log_fee'] - self.bounds['min_log_fee'] + 1e-9)
        
        # 5. Time Ratio (Fix: Calculate on-the-fly if missing in parquet)
        if 'time_ratio' not in self.df.columns:
            H = self.metadata['config']['env'].get('horizon', 128)
            time_ratio = (self.df.groupby('episode_id').cumcount() / (H - 1)).astype(np.float32).values
        else:
            time_ratio = self.df['time_ratio'].values.astype(np.float32)
            
        # 6. Normalize Actions (% of Queue)
        raw_actions = self.df['action'].values
        queue_vals = self.df['queue_size'].values
        norm_actions = np.where(queue_vals > 1e-6, raw_actions / (queue_vals + 1e-9), 0.0)
        self.actions = np.clip(norm_actions, 0.0, 1.0).astype(np.float32)
        
        # Combine into Observation Matrix
        self.obs_matrix = np.column_stack([
            norm_queue,
            norm_fee,
            norm_vol,
            norm_lags,
            time_ratio
        ]).astype(np.float32)
        
        # Targets
        self.rewards = self.df['reward'].values.astype(np.float32)
        self.terminals = self.df['terminal'].values.astype(np.float32)
        
    def __len__(self):
        return self.num_rows - 1

    def __getitem__(self, idx):
        s = self.obs_matrix[idx]
        a = self.actions[idx]
        r = self.rewards[idx]
        s_next = self.obs_matrix[idx + 1]
        d = self.terminals[idx]
        
        if d:
            s_next = np.zeros_like(s)
            
        return {
            'observations': torch.tensor(s),
            'actions': torch.tensor([a]),
            'rewards': torch.tensor([r]),
            'next_observations': torch.tensor(s_next),
            'terminals': torch.tensor([d])
        }

def get_offline_loader(
    parquet_path,
    metadata_path,
    batch_size=256,
    shuffle=True,
    num_workers=None,
    distributed=False,
    rank=0,
    world_size=1,
    drop_last=True,
):
    dataset = GasOfflineDataset(parquet_path, metadata_path)

    if num_workers is None:
        # A safe default for Kaggle/Colab-like environments
        cpu = os.cpu_count() or 2
        num_workers = min(4, max(1, cpu // max(1, world_size)))

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    return DataLoader(dataset, **loader_kwargs)