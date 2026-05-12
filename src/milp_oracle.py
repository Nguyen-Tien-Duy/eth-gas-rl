import numpy as np
from ortools.linear_solver import pywraplp

def solve_episode_milp(
    gas_prices: np.ndarray, 
    arrivals: np.ndarray, 
    Q_initial: float, 
    C_cap: int, 
    C_base: float, 
    beta: float, 
    alpha: float,
    lambda_d: float,
    gas_ref: np.ndarray
):
    """
    Solve the Global Optimum problem using Mixed-Integer Linear Programming (MILP).
    The objective is designed to PERFECTLY MIRROR the 3-tier reward function.
    """
    H = len(gas_prices)
    
    # Initialize the Solver with SCIP engine
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise Exception("SCIP solver not found.")

    # ================= 1. VARIABLES =================
    n = {}  # Action: Number of tx to execute at block t
    Q = {}  # State: Queue backlog at the START of block t
    I = {}  # Binary: 1 if n > 0 (execute), 0 if idle
    
    for t in range(H):
        n[t] = solver.IntVar(0, C_cap, f'n_{t}')
        I[t] = solver.IntVar(0, 1, f'I_{t}')
        Q[t] = solver.NumVar(0, solver.infinity(), f'Q_{t}')
        
    Q[H] = solver.NumVar(0, solver.infinity(), f'Q_{H}')

    # ================= 2. CONSTRAINTS =================
    solver.Add(Q[0] == Q_initial)
    
    for t in range(H):
        # A. Queue Dynamics: Q_next = Q_now - n + arrivals
        solver.Add(Q[t+1] == Q[t] - n[t] + arrivals[t])
        
        # B. Cannot execute more than current queue
        solver.Add(n[t] <= Q[t])
        
        # C. Fixed cost trigger: n[t] <= C_cap * I[t]
        solver.Add(n[t] <= C_cap * I[t])

    # ================= 3. OBJECTIVE (MIRRORING REWARD) =================
    # Reward = (n*(gas_ref - gas_price) - I*FixedCost) - (Q_next*Urgency) - (Q_H*Catastrophe)
    # We MINIMIZE the negative of the Reward
    objective = solver.Objective()
    
    for t in range(H):
        # Savings component: -n * (gas_ref - gas_price)
        objective.SetCoefficient(n[t], float(-(gas_ref[t] - gas_prices[t])))
        
        # Fixed cost component: I * (C_base * gas_price / 1e9)
        fixed_cost = (C_base / 1e9) * gas_prices[t]
        objective.SetCoefficient(I[t], float(fixed_cost))
        
        # Urgency penalty component: Q_next * penalty_at_t
        time_ratio = t / H
        penalty_coeff = beta * np.exp(alpha * (1.0 - time_ratio))
        objective.SetCoefficient(Q[t+1], float(penalty_coeff))

    # Final Catastrophe Penalty: Q[H] * lambda_d
    objective.SetCoefficient(Q[H], float(lambda_d))
    
    objective.SetMinimization()

    # 1. SET LIMITS
    solver.SetTimeLimit(500) 
    
    solver_params = pywraplp.MPSolverParameters()
    solver_params.SetDoubleParam(pywraplp.MPSolverParameters.RELATIVE_MIP_GAP, 1e-4)
    
    status = solver.Solve(solver_params)
    
    # 2. COMPUTE GAP
    incumbent = solver.Objective().Value()
    best_bound = solver.Objective().BestBound()
    
    if abs(incumbent) > 1e-7:
        gap = abs(incumbent - best_bound) / abs(incumbent)
    else:
        gap = 0.0
        
    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        opt_n = np.array([n[t].solution_value() for t in range(H)])
        return opt_n, gap
    else:
        return None, 1.0

# Simple Simulation Test
if __name__ == "__main__":
    np.random.seed(42)
    # Simulate a 10-block episode
    H = 10
    test_gas = np.random.uniform(10, 50, H)
    test_arrivals = np.random.randint(0, 15, H)
    
    print("Gas Prices:", np.round(test_gas, 1))
    print("Arrivals:  ", test_arrivals)
    
    opt_action, cost = solve_episode_milp(
        gas_prices=test_gas,
        arrivals=test_arrivals,
        Q_initial=50,
        C_cap=100,
        C_base=21000,
        beta=0.1,
        alpha=2.0
    )
    
    print("\n[GOD MODE] Optimal Action per block:")
    print(opt_action)
    print("Minimum Total Cost:", cost)
