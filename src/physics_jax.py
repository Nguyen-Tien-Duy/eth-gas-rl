import jax
import jax.numpy as jnp
from jax import jit

@jit
def calculate_gas_used_jax(n_transaction):
    """
    JAX version of gas calculation.
    Formula: Base(21k) + n * 15k
    """
    base_gas = 21000
    per_tx_gas = 15000
    return base_gas + (n_transaction * per_tx_gas)

@jit
def calculate_next_base_fee_jax(current_base_fee, gas_used, target_gas):
    """
    JAX version of EIP-1559 base fee update.
    Formula: current_base_fee * (1 + (gas_used - target_gas) / (8 * target_gas))
    """
    delta = gas_used - target_gas
    adjustment = delta / (8.0 * target_gas) # 8 is CELL_FACTOR from EIP1559
    
    next_fee = current_base_fee * (1.0 + adjustment)
    return jnp.maximum(1e-6, next_fee)

@jit
def compute_reward_jax(n_t, q_t, gas_price, gas_ref, c_base, beta, alpha, time_ratio):
    """
    Pure JAX Reward Calculation.
    n_t: transactions executed
    q_t: queue AFTER execution
    """
    # 1. Efficiency Tier
    savings = n_t * (gas_ref - gas_price)
    # n_t > 0.5 is represented as jnp.where for JIT compatibility
    fixed_cost = (c_base / 1e9) * gas_price * jnp.where(n_t > 0.5, 1.0, 0.0)
    r_eff = savings - fixed_cost
    
    # 2. Urgency Tier
    r_urg = q_t * beta * jnp.exp(alpha * (1.0 - time_ratio))
    
    # Reward is scaled for stability
    reward = (r_eff - r_urg) / 100.0
    return reward

@jit
def step_physics_jax(q_current, n_action, arrival):
    """
    Mass conservation step in JAX.
    Returns: (q_next, n_clamped)
    """
    n_clamped = jnp.minimum(n_action, q_current)
    q_next = q_current - n_clamped + arrival
    return q_next, n_clamped
