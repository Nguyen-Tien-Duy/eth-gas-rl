import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.physics_jax import compute_reward_jax

def profile_reward():
    # 1. Setup Ranges
    # Gas diff from -20 to 20 Gwei (Reference - Current)
    gas_diffs = jnp.linspace(-20, 20, 100)
    # Queue size from 0 to 1000
    queues = jnp.linspace(0, 1000, 100)
    
    # Grid
    G, Q = jnp.meshgrid(gas_diffs, queues)
    
    # 2. Parameters (Based on your config)
    c_base = 21000.0
    beta = 0.1
    alpha = 2.0
    time_ratio = 0.5 # At the middle of episode
    action_pct = 0.5 # Agent executes 50% of queue
    
    # 3. Vectorized Calculation
    # We need to simulate n_executed and next_queue
    # n_executed = action_pct * Q (simplified for profiling)
    n_exec = action_pct * Q
    # next_queue = Q - n_exec
    nq = Q - n_exec
    
    # Compute Reward across the grid
    # gas_diff = gas_ref - gas_curr, so gas_ref = gas_curr + G
    # Let's assume gas_curr is 30.0 for simplicity
    gas_curr = 30.0
    gas_ref = gas_curr + G
    
    v_reward = jax.vmap(jax.vmap(compute_reward_jax, in_axes=(0, 0, None, 0, None, None, None, None)), 
                        in_axes=(0, 0, None, 0, None, None, None, None))
    
    rewards = v_reward(n_exec, nq, gas_curr, gas_ref, c_base, beta, alpha, time_ratio)
    
    # 4. Plotting
    plt.figure(figsize=(12, 8))
    cp = plt.contourf(G, Q, rewards, levels=50, cmap='RdYlGn')
    plt.colorbar(cp, label='Reward Value')
    
    # Add a line where Reward = 0
    plt.contour(G, Q, rewards, levels=[0], colors='white', linestyles='--')
    
    plt.title(f'Reward Surface Analysis (Time Ratio: {time_ratio}, Beta: {beta}, Alpha: {alpha})')
    plt.xlabel('Gas Savings per unit (Gas_ref - Gas_curr) [Gwei]')
    plt.ylabel('Current Queue Size (Backlog)')
    
    plt.annotate('Negative Reward (Loss)', xy=(-10, 800), color='white', fontweight='bold')
    plt.annotate('Positive Reward (Profit)', xy=(10, 200), color='black', fontweight='bold')
    
    os.makedirs("artifacts", exist_ok=True)
    output_path = "artifacts/reward_surface.png"
    plt.savefig(output_path)
    print(f"Reward surface plot saved to {output_path}")
    
    # Analysis: Find the "Backlog Limit" 
    # For a given gas_diff, at what Queue does the reward become negative?
    # R = n*G - Q*beta*exp(alpha*t)
    # 0.5*Q*G = Q*beta*exp(alpha*t) => G_threshold = 2*beta*exp(alpha*t)
    g_threshold = 2 * beta * np.exp(alpha * time_ratio)
    print(f"\n--- Scientific Analysis ---")
    print(f"Current Settings: Beta={beta}, Alpha={alpha}, Time={time_ratio}")
    print(f"To be profitable, the Gas Savings (Gwei) MUST be greater than {g_threshold:.2f}")
    print(f"If Gas Savings is 10 Gwei, the Agent will backlog UNTIL the penalty exceeds 10 Gwei.")

if __name__ == "__main__":
    profile_reward()
