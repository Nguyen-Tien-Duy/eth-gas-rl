import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from src.physics import calculate_gas_used, calculate_next_base_fee, compute_reward_numba, step_physics_numba

class EthGasEnv(gym.Env):
    """
    Ethereum Gas Management Environment (Gymnasium Standard)
    Supports both Trace-based evaluation and EIP-1559 Simulation.
    """
    def __init__(self, config, trace_df=None):
        super(EthGasEnv, self).__init__()
        
        self.config = config
        self.env_config = config.get('env', {})
        self.rl_config = config.get('rl', {})
        
        # Space Definitions
        self.H = self.env_config.get('horizon', 128)
        self.C_cap = self.env_config.get('execution_capacity', 100)
        self.C_base = self.env_config.get('C_base', 21000)
        
        # Action: Percentage of current queue to execute [0, 1]
        self.action_space = spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation: [Queue, Current_Gas, Lags..., Time_Ratio]
        self.num_lags = config.get('state', {}).get('num_lags', 5)
        obs_dim = 1 + 1 + self.num_lags + 1 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Cache params for Numba
        self.beta = self.rl_config.get('urgency_beta', 0.1)
        self.alpha = self.rl_config.get('urgency_alpha', 2.0)
        self.lambda_d = self.rl_config.get('deadline_penalty', 500.0)
        
        # Load Normalization Stats
        metadata_path = f"data/processed/{config['experiment_name']}/metadata.json"
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        self.bounds = meta['normalization_bounds']
        
        # State variables
        self.trace_df = trace_df
        self.current_step = 0
        self.queue = 0.0
        self.gas_history = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.trace_df is not None:
            ep_ids = self.trace_df['episode_id'].unique()
            self.target_ep = np.random.choice(ep_ids)
            self.ep_data = self.trace_df[self.trace_df['episode_id'] == self.target_ep].reset_index()
            
            self.gas_prices = self.ep_data['base_fee_per_gas'].values / 1e9
            self.arrivals = (self.ep_data['transaction_count'].values * self.env_config.get('arrival_scale', 0.1)).astype(np.int64)
            self.gas_ref = self.ep_data['gas_reference'].values / 1e9
        else:
            self.gas_prices = np.full(self.H, 20.0) 
            self.arrivals = np.random.poisson(10, self.H)
            self.gas_ref = np.full(self.H, 20.0)

        self.current_step = 0
        self.queue = 0.0
        self.gas_history = [self.gas_prices[0]] * self.num_lags
        
        return self._get_obs(), {}

    def _get_obs(self):
        time_ratio = self.current_step / float(self.H)
        current_gas = self.gas_prices[self.current_step]
        lags = np.array(self.gas_history[-self.num_lags:])
        
        # 1. Normalize Queue
        norm_q = self.queue / (self.bounds['max_queue'] + 1e-9)
        
        # 2. Normalize Current Gas & Lags
        def norm_log_gas(g):
            log_g = np.log(g + 1e-9)
            return (log_g - self.bounds['min_log_fee']) / (self.bounds['max_log_fee'] - self.bounds['min_log_fee'] + 1e-9)
        
        norm_gas = norm_log_gas(current_gas)
        norm_lags = np.array([norm_log_gas(l) for l in lags])
        
        # 3. Volatility
        vol = np.std(np.log(lags + 1e-9))
        norm_vol = vol / (self.bounds['max_volatility'] + 1e-9)
        
        # Combine into 9-dim Observation Vector
        obs = np.concatenate([
            [norm_q],
            [norm_gas],
            [norm_vol],
            norm_lags,
            [time_ratio]
        ]).astype(np.float32)
        return obs

    def step(self, action):
        # 1. Get current state data
        current_gas = self.gas_prices[self.current_step]
        ref_gas = self.gas_ref[self.current_step]
        arrival = self.arrivals[self.current_step]
        time_ratio = self.current_step / float(self.H)
        
        # 2. Physics & Reward (Action is % of Queue)
        # Actual n = min(prob * queue, C_cap)
        n_intended = action[0] * self.queue
        self.queue, action_clamped = step_physics_numba(self.queue, n_intended, arrival)
        
        # Limit by C_cap is handled by the solver/physics logic or explicitly here
        action_clamped = min(action_clamped, self.C_cap)
        
        reward = compute_reward_numba(
            action_clamped, self.queue, current_gas, ref_gas, 
            self.C_base, self.beta, self.alpha, time_ratio
        )
        
        # 3. Update History & Step
        self.gas_history.append(current_gas)
        self.current_step += 1
        
        done = (self.current_step >= self.H - 1)
        
        # Final Catastrophe Penalty (Numba-compatible logic)
        final_penalty = 0.0
        if done:
            final_penalty = (self.queue * self.lambda_d) / 100.0
            reward -= final_penalty
            
        info = {
            "savings": action_clamped * (ref_gas - current_gas),
            "n_executed": action_clamped,
            "queue": self.queue,
            "final_penalty": final_penalty
        }
            
        return self._get_obs() if not done else np.zeros(self.observation_space.shape), reward, done, False, info

    def render(self):
        pass
