import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.physics import step_physics_numba, compute_reward_numba
from src.physics_jax import step_physics_jax, compute_reward_jax

def test_equivalence():
    print("=== Testing Numba vs JAX Equivalence ===")
    
    # Test Inputs
    q_current = 50.0
    n_action = 30.0
    arrival = 10
    gas_price = 20.0
    gas_ref = 25.0
    c_base = 21000
    beta = 0.1
    alpha = 2.0
    time_ratio = 0.5
    
    # 1. Test Physics Step
    q_next_n, n_c_n = step_physics_numba(q_current, n_action, arrival)
    q_next_j, n_c_j = step_physics_jax(q_current, n_action, arrival)
    
    print(f"Physics Step:")
    print(f"  Numba: q_next={q_next_n}, n_clamped={n_c_n}")
    print(f"  JAX:   q_next={q_next_j}, n_clamped={n_c_j}")
    
    assert np.allclose(q_next_n, q_next_j), "Physics q_next mismatch!"
    
    # 2. Test Reward Calculation
    r_n = compute_reward_numba(n_c_n, q_next_n, gas_price, gas_ref, c_base, beta, alpha, time_ratio)
    r_j = compute_reward_jax(n_c_j, q_next_j, gas_price, gas_ref, c_base, beta, alpha, time_ratio)
    
    print(f"Reward Calculation:")
    print(f"  Numba: reward={r_n}")
    print(f"  JAX:   reward={r_j}")
    
    assert np.allclose(r_n, r_j), "Reward mismatch!"
    print("✅ Equivalence Test Passed!")

def benchmark_speed():
    print("\n=== Benchmarking Speed (Single Call) ===")
    q = 50.0
    a = 30.0
    arr = 10
    
    # Warmup
    _ = step_physics_numba(q, a, arr)
    _ = step_physics_jax(q, a, arr)
    
    # Numba Benchmark
    start = time.time()
    for _ in range(10000):
        _ = step_physics_numba(q, a, arr)
    print(f"Numba (10k calls): {time.time() - start:.4f}s")
    
    # JAX Benchmark
    start = time.time()
    for _ in range(10000):
        _ = step_physics_jax(q, a, arr)
    print(f"JAX (10k calls):   {time.time() - start:.4f}s")
    
    # THE REAL POWER: JAX Vectorization
    print("\n=== JAX Vectorization Power (vmap) ===")
    v_step = jax.vmap(step_physics_jax)
    key = jax.random.PRNGKey(0)
    key1, key2, key3 = jax.random.split(key, 3)
    qs = jax.random.uniform(key1, (10000,), minval=0.0, maxval=100.0)
    as_ = jax.random.uniform(key2, (10000,), minval=0.0, maxval=50.0)
    arrs = jax.random.randint(key3, (10000,), minval=0, maxval=20)
    
    # Warmup
    _ = v_step(qs, as_, arrs)
    
    start = time.time()
    _ = v_step(qs, as_, arrs)
    print(f"JAX vmap (10k parallel): {time.time() - start:.6f}s")

if __name__ == "__main__":
    test_equivalence()
    benchmark_speed()
