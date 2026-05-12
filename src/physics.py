import numpy as np 
from numba import njit 

@njit 
def calculate_gas_used(n_transaction: int) -> int:
    """
    This function calculate the gas used for a batch of transactions.
    The formula is : Base(21k) + n * 15k
    """
    base_gas = 21000
    per_tx_gas = 15000
    return base_gas + (n_transaction * per_tx_gas)

@njit
def calculate_next_base_fee(current_base_fee: float, gas_used: int, target_gas: int) -> float:
    """
    This function calculate the next base fee for the next block.
    The formula is : current_base_fee * (1 + (gas_used - target_gas) / (8 * target_gas))
    """
    delta = gas_used - target_gas
    adjustment = delta / (8 * target_gas) # 8 is CELL_FACTOR from EIP1559
    
    next_fee = current_base_fee * (1 + adjustment)
    return max(0.000001, next_fee)
    

    