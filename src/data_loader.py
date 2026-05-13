import torch
import pandas as pd
import numpy as np
import json
import gc
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class GasOfflineDataset(Dataset):
    """
    Memory-optimized PyTorch Dataset for Offline RL.
    """
    def __init__(self, parquet_path, metadata_path, context_length=1):
        self.parquet_path = parquet_path
        self.context_length = context_length
        
        with open(metadata_path, 'r') as f:
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

def get_offline_loader(parquet_path, metadata_path, batch_size=256, shuffle=True):
    dataset = GasOfflineDataset(parquet_path, metadata_path)
    # Optimized for 4-core CPU, using 2 workers for data prep
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)