import jax
import jax.numpy as jnp
from jax import jit

@jit
def calculate_gas_used_jax(n_transaction):
    """Formula: Base(21k) + n * 15k"""
    base_gas = 21000
    per_tx_gas = 15000
    return base_gas + (n_transaction * per_tx_gas)

@jit
def calculate_next_base_fee_jax(current_base_fee, gas_used, target_gas):
    """EIP-1559 base fee update logic"""
    delta = gas_used - target_gas
    adjustment = delta / (8.0 * target_gas)
    next_fee = current_base_fee * (1.0 + adjustment)
    return jnp.maximum(1e-6, next_fee)

@jit
def compute_reward_jax(n_executed, next_queue, gas_curr, gas_ref, c_base, beta, alpha, time_ratio, c_mar=15000.0, sigma=1e9):
    """
    Certified Reward Function from Final Project Report.
    - Efficiency: Savings based on Gas Reference vs Current Cost
    - Urgency: Exponential penalty INCREASING over time
    - Scale: sigma = 1e9 (Gwei to ETH)
    """
    # 1. Efficiency Tier (Profit/Savings)
    # Gas cost = (base + mar * n) * price
    # We use Gwei for price, so results are in Gwei. Divide by 1e9 for ETH.
    execution_cost = (c_base * (n_executed > 0.5) + c_mar * n_executed) * gas_curr
    savings = (n_executed * gas_ref) - execution_cost
    r_eff = savings / sigma
    
    # 2. Urgency Tier (Time Pressure)
    # Increases as time_ratio goes from 0 to 1
    r_urg = (beta / sigma) * next_queue * jnp.exp(alpha * time_ratio)
    
    return r_eff - r_urg

@jit
def step_physics_jax(q_current, n_action, arrival):
    """Mass conservation: Q_next = Q_curr - n + w"""
    n_clamped = jnp.minimum(n_action, q_current)
    q_next = q_current - n_clamped + arrival
    return q_next, n_clamped
