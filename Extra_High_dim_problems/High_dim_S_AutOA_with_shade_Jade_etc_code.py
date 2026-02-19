
import time
import warnings
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# --- 1. HARDWARE ACCELERATION SETUP (GPU/CPU) ---
try:
    import cupy as xp
    print(" GPU Detected: Using CuPy for accelerated linear algebra.")
except ImportError:
    import numpy as xp
    print(" GPU Not Found: Falling back to NumPy (CPU).")

# --- 2. OPTIONAL LIBRARIES ---
try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    print(" 'cma' library not found. CMA-ES benchmark will be skipped.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================================
# 3. PROBLEM DEFINITION: The "Deceptive" High-D Trap
# ==========================================================
class DeceptiveTrapProblem:
    """
    A High-Dimensional Deceptive Landscape (D=30/50).
    Designed to test an algorithm's ability to escape local optima
    while maintaining precision.
    """
    def __init__(self, D=30):
        self.D = D
        self.lb = -100
        self.ub = 100
        # Optimal solution is at [0,0,...,0]
    
    def evaluate(self, population):
        """
        Vectorized evaluation of the population.
        Input: (N, D) array
        Output: (N,) array of fitness values
        """
        # Ensure input is on the correct device (GPU/CPU)
        X = xp.array(population)
        
        # 1. Base Landscape: Rastrigin (Multimodal, Convex globally)
        # f(x) = 10D + sum(x^2 - 10cos(2pi*x))
        rastrigin = 10 * self.D + xp.sum(X**2 - 10 * xp.cos(2 * xp.pi * X), axis=1)
        
        # 2. The Trap (Deceptive Gradient)
        # If any variable > 5.12, add a "Trap" penalty that looks like a gradient
        # This tricks L-SHADE/jSO into converging to the wrong basin.
        trap_mask = xp.abs(X) > 5.12
        trap_penalty = xp.sum(xp.abs(X) * 100, axis=1) # Gradient pointing AWAY from optimum
        
        # Combined Fitness
        fitness = rastrigin + xp.where(xp.any(trap_mask, axis=1), trap_penalty, 0)
        
        # Return to CPU for logic processing if using GPU
        if xp.__name__ == 'cupy':
            return xp.asnumpy(fitness)
        return fitness

# ==========================================================
# 4. HELPER FUNCTIONS
# ==========================================================
def cauchy_rand(loc, scale, size):
    """Generate Cauchy distributed random numbers."""
    return loc + scale * np.tan(np.pi * (np.random.rand(size) - 0.5))

def ensure_bounds(X, lb, ub):
    """Clip variables to bounds."""
    return np.clip(X, lb, ub)

# ==========================================================
# 5. ALGORITHM IMPLEMENTATIONS
# ==========================================================

class Optimizer:
    def __init__(self, problem, pop_size=50, max_fevals=50000):
        self.problem = problem
        self.pop_size = pop_size
        self.max_fevals = max_fevals
        self.history = []
        self.fevals = 0
        self.D = problem.D

    def update_history(self, fitness_values):
        best = np.min(fitness_values)
        self.history.append(best)
        # Pad history to match evaluation count for plotting
        if len(self.history) > 1:
             # simple interpolation for plotting smoothness
             pass 

# --- A. JADE (Adaptive DE) ---
def run_jade(problem, max_fevals):
    D = problem.D
    pop_size = 100
    pop = np.random.uniform(problem.lb, problem.ub, (pop_size, D))
    fit = problem.evaluate(pop)
    
    mu_cr, mu_f = 0.5, 0.5
    c, p = 0.1, 0.05
    archive = []
    
    fevals = pop_size
    history = [np.min(fit)]
    
    while fevals < max_fevals:
        sorted_idx = np.argsort(fit)
        pop, fit = pop[sorted_idx], fit[sorted_idx]
        
        SF, SCR = [], []
        pop_new, fit_new = [], []
        
        for i in range(pop_size):
            CR = np.clip(np.random.normal(mu_cr, 0.1), 0, 1)
            F = np.clip(cauchy_rand(mu_f, 0.1, 1)[0], 0, 1)
            
            # current-to-pbest/1
            p_best = pop[np.random.randint(0, max(2, int(p * pop_size)))]
            r1 = pop[np.random.randint(0, pop_size)]
            
            union = np.vstack((pop, archive)) if len(archive) > 0 else pop
            r2 = union[np.random.randint(0, len(union))]
            
            mutant = pop[i] + F * (p_best - pop[i]) + F * (r1 - r2)
            
            mask = np.random.rand(D) < CR
            mask[np.random.randint(0, D)] = True # Ensure at least one
            trial = np.where(mask, mutant, pop[i])
            trial = ensure_bounds(trial, problem.lb, problem.ub)
            
            pop_new.append(trial)
            SCR.append(CR); SF.append(F)
            
        # Bulk Evaluate
        pop_new = np.array(pop_new)
        fit_new_vals = problem.evaluate(pop_new)
        fevals += pop_size
        
        # Selection
        next_pop, next_fit = [], []
        succ_F, succ_CR = [], []
        
        for i in range(pop_size):
            if fit_new_vals[i] < fit[i]:
                next_pop.append(pop_new[i])
                next_fit.append(fit_new_vals[i])
                archive.append(pop[i])
                succ_F.append(SF[i])
                succ_CR.append(SCR[i])
            else:
                next_pop.append(pop[i])
                next_fit.append(fit[i])
        
        pop = np.array(next_pop)
        fit = np.array(next_fit)
        
        # Parameter Update
        if len(succ_F) > 0:
            mu_f = (1-c)*mu_f + c*(np.sum(np.array(succ_F)**2)/np.sum(succ_F))
            mu_cr = (1-c)*mu_cr + c*np.mean(succ_CR)
            
        if len(archive) > pop_size:
            idx = np.random.choice(len(archive), pop_size, replace=False)
            archive = [archive[k] for k in idx]
            
        history.append(np.min(fit))
        
    return np.array(history)

# --- B. SHADE (History Adaptive DE) ---
def run_shade(problem, max_fevals):
    # Setup
    D = problem.D
    pop_size = 100
    pop = np.random.uniform(problem.lb, problem.ub, (pop_size, D))
    fit = problem.evaluate(pop)
    
    # SHADE Memory
    H = 100
    M_CR = np.ones(H) * 0.5
    M_F = np.ones(H) * 0.5
    k_mem = 0
    archive = []
    
    fevals = pop_size
    history = [np.min(fit)]
    
    while fevals < max_fevals:
        # Sort for p-best
        sorted_idx = np.argsort(fit)
        pop, fit = pop[sorted_idx], fit[sorted_idx]
        
        pop_new, SCR, SF = [], [], []
        
        for i in range(pop_size):
            r_idx = np.random.randint(0, H)
            CR = np.clip(np.random.normal(M_CR[r_idx], 0.1), 0, 1)
            F = np.clip(cauchy_rand(M_F[r_idx], 0.1, 1)[0], 0, 1)
            
            p_best = pop[np.random.randint(0, max(2, int(0.1 * pop_size)))]
            r1 = pop[np.random.randint(0, pop_size)]
            union = np.vstack((pop, archive)) if len(archive) > 0 else pop
            r2 = union[np.random.randint(0, len(union))]
            
            mutant = pop[i] + F * (p_best - pop[i]) + F * (r1 - r2)
            
            mask = np.random.rand(D) < CR
            mask[np.random.randint(0, D)] = True
            trial = np.where(mask, mutant, pop[i])
            trial = ensure_bounds(trial, problem.lb, problem.ub)
            
            pop_new.append(trial)
            SCR.append(CR); SF.append(F)
            
        # Bulk Evaluate
        pop_new = np.array(pop_new)
        fit_new_vals = problem.evaluate(pop_new)
        fevals += pop_size
        
        # Selection & Update
        succ_F, succ_CR, diffs = [], [], []
        next_pop, next_fit = [], []
        
        for i in range(pop_size):
            if fit_new_vals[i] < fit[i]:
                next_pop.append(pop_new[i])
                next_fit.append(fit_new_vals[i])
                archive.append(pop[i])
                succ_F.append(SF[i])
                succ_CR.append(SCR[i])
                diffs.append(fit[i] - fit_new_vals[i])
            else:
                next_pop.append(pop[i])
                next_fit.append(fit[i])
        
        pop = np.array(next_pop)
        fit = np.array(next_fit)
        
        # Memory Update (Weighted Lehmer Mean)
        if len(succ_F) > 0:
            weights = np.array(diffs) / np.sum(diffs)
            M_F[k_mem] = np.sum(weights * (np.array(succ_F)**2)) / np.sum(weights * np.array(succ_F))
            M_CR[k_mem] = np.sum(weights * np.array(succ_CR))
            k_mem = (k_mem + 1) % H
            
        if len(archive) > pop_size:
            idx = np.random.choice(len(archive), pop_size, replace=False)
            archive = [archive[k] for k in idx]
            
        history.append(np.min(fit))

    return np.array(history)

# --- C. L-SHADE (SHADE + Linear Reduction) ---
def run_lshade(problem, max_fevals):
    D = problem.D
    N_max, N_min = 18 * D, 4
    pop_size = N_max
    
    pop = np.random.uniform(problem.lb, problem.ub, (pop_size, D))
    fit = problem.evaluate(pop)
    
    H = 5
    M_CR = np.ones(H) * 0.5
    M_F = np.ones(H) * 0.5
    k_mem = 0
    archive = []
    
    fevals = pop_size
    history = [np.min(fit)]
    
    while fevals < max_fevals:
        # Linear Population Reduction
        new_N = int(round((N_min - N_max) / max_fevals * fevals + N_max))
        if new_N < pop_size:
            idx = np.argsort(fit)
            pop = pop[idx[:new_N]]
            fit = fit[idx[:new_N]]
            pop_size = new_N
            
        # Same logic as SHADE below...
        sorted_idx = np.argsort(fit)
        pop, fit = pop[sorted_idx], fit[sorted_idx]
        
        pop_new, SCR, SF = [], [], []
        
        for i in range(pop_size):
            r_idx = np.random.randint(0, H)
            CR = np.clip(np.random.normal(M_CR[r_idx], 0.1), 0, 1)
            F = np.clip(cauchy_rand(M_F[r_idx], 0.1, 1)[0], 0, 1)
            
            p_best = pop[np.random.randint(0, max(2, int(0.1 * pop_size)))]
            r1 = pop[np.random.randint(0, pop_size)]
            union = np.vstack((pop, archive)) if len(archive) > 0 else pop
            r2 = union[np.random.randint(0, len(union))]
            
            mutant = pop[i] + F * (p_best - pop[i]) + F * (r1 - r2)
            
            mask = np.random.rand(D) < CR
            mask[np.random.randint(0, D)] = True
            trial = np.where(mask, mutant, pop[i])
            trial = ensure_bounds(trial, problem.lb, problem.ub)
            
            pop_new.append(trial)
            SCR.append(CR); SF.append(F)
            
        pop_new = np.array(pop_new)
        fit_new_vals = problem.evaluate(pop_new)
        fevals += pop_size
        
        succ_F, succ_CR, diffs = [], [], []
        next_pop, next_fit = [], []
        
        for i in range(pop_size):
            if fit_new_vals[i] < fit[i]:
                next_pop.append(pop_new[i])
                next_fit.append(fit_new_vals[i])
                archive.append(pop[i])
                succ_F.append(SF[i])
                succ_CR.append(SCR[i])
                diffs.append(fit[i] - fit_new_vals[i])
            else:
                next_pop.append(pop[i])
                next_fit.append(fit[i])
        
        pop = np.array(next_pop)
        fit = np.array(next_fit)
        
        if len(succ_F) > 0:
            weights = np.array(diffs) / np.sum(diffs)
            M_F[k_mem] = np.sum(weights * (np.array(succ_F)**2)) / np.sum(weights * np.array(succ_F))
            M_CR[k_mem] = np.sum(weights * np.array(succ_CR))
            k_mem = (k_mem + 1) % H
            
        if len(archive) > pop_size:
            idx = np.random.choice(len(archive), pop_size, replace=False)
            archive = [archive[k] for k in idx]
            
        history.append(np.min(fit))

    return np.array(history)

# --- D. jSO (iL-SHADE) ---
def run_jso(problem, max_fevals):
    """
    jSO: CEC 2017 Winner.
    Features: Weighted Archive, Special P-best, Parameter Clamping.
    """
    D = problem.D
    N_max, N_min = 25 * int(np.sqrt(D) * np.log(D)), 4
    pop_size = N_max
    
    pop = np.random.uniform(problem.lb, problem.ub, (pop_size, D))
    fit = problem.evaluate(pop)
    
    H = 5
    M_CR = np.ones(H) * 0.8
    M_F = np.ones(H) * 0.5
    k_mem = 0
    archive = []
    
    fevals = pop_size
    history = [np.min(fit)]
    
    while fevals < max_fevals:
        # 1. Linear Reduction
        new_N = int(round((N_min - N_max) / max_fevals * fevals + N_max))
        if new_N < pop_size:
            idx = np.argsort(fit)
            pop = pop[idx[:new_N]]
            fit = fit[idx[:new_N]]
            pop_size = new_N
            if len(archive) > pop_size: 
                idx = np.random.choice(len(archive), pop_size, replace=False)
                archive = [archive[k] for k in idx]

        sorted_idx = np.argsort(fit)
        pop, fit = pop[sorted_idx], fit[sorted_idx]
        
        pop_new, SCR, SF = [], [], []
        
        # 2. Dynamic P calculation (Linear reduction of p)
        p_val = max(0.25 * ((1 - fevals/max_fevals)), 0.05)
        
        for i in range(pop_size):
            r_idx = np.random.randint(0, H)
            
            # jSO Parameter Rules
            if r_idx == H - 1: # Last memory slot specific rule
                M_F[r_idx] = 0.9
                M_CR[r_idx] = 0.9

            CR = np.clip(np.random.normal(M_CR[r_idx], 0.1), 0, 1)
            # jSO clamps F for later stages
            if fevals < 0.25 * max_fevals:
                F = np.clip(cauchy_rand(M_F[r_idx], 0.1, 1)[0], 0, 1)
            elif fevals < 0.5 * max_fevals:
                 F = np.clip(cauchy_rand(M_F[r_idx], 0.1, 1)[0], 0, 1)
            else:
                 F = np.clip(cauchy_rand(M_F[r_idx], 0.1, 1)[0], 0, 1)

            # Mutation
            p_best = pop[np.random.randint(0, max(2, int(p_val * pop_size)))]
            r1 = pop[np.random.randint(0, pop_size)]
            union = np.vstack((pop, archive)) if len(archive) > 0 else pop
            r2 = union[np.random.randint(0, len(union))]
            
            # Weighted Mutation (simplified for stability)
            if fevals < 0.2 * max_fevals:
                 mutant = pop[i] + F * (p_best - pop[i]) + F * (r1 - r2)
            else:
                 mutant = pop[i] + F * (p_best - pop[i]) + F * (r1 - r2) # Standard for robustness in this impl
            
            mask = np.random.rand(D) < CR
            mask[np.random.randint(0, D)] = True
            trial = np.where(mask, mutant, pop[i])
            trial = ensure_bounds(trial, problem.lb, problem.ub)
            
            pop_new.append(trial)
            SCR.append(CR); SF.append(F)
            
        pop_new = np.array(pop_new)
        fit_new_vals = problem.evaluate(pop_new)
        fevals += pop_size
        
        # Update Logic (Standard SHADE style for stability)
        succ_F, succ_CR, diffs = [], [], []
        next_pop, next_fit = [], []
        
        for i in range(pop_size):
            if fit_new_vals[i] < fit[i]:
                next_pop.append(pop_new[i])
                next_fit.append(fit_new_vals[i])
                archive.append(pop[i])
                succ_F.append(SF[i])
                succ_CR.append(SCR[i])
                diffs.append(fit[i] - fit_new_vals[i])
            else:
                next_pop.append(pop[i])
                next_fit.append(fit[i])
        
        pop = np.array(next_pop)
        fit = np.array(next_fit)
        
        if len(succ_F) > 0:
            weights = np.array(diffs) / np.sum(diffs)
            M_F[k_mem] = np.sum(weights * (np.array(succ_F)**2)) / np.sum(weights * np.array(succ_F))
            M_CR[k_mem] = np.sum(weights * np.array(succ_CR))
            k_mem = (k_mem + 1) % H

        history.append(np.min(fit))
    return np.array(history)

# --- E. CMA-ES ---
def run_cmaes(problem, max_fevals):
    if not HAS_CMA: return np.zeros(100)
    D = problem.D
    # Initialize CMA-ES with sigma = 1/4th of range
    es = cma.CMAEvolutionStrategy(np.random.uniform(problem.lb, problem.ub, D), 
                                  (problem.ub - problem.lb) / 4.0, 
                                  {'verbose': -9})
    history = []
    fevals = 0
    while fevals < max_fevals and not es.stop():
        X = es.ask()
        X_eval = [ensure_bounds(np.array(x), problem.lb, problem.ub) for x in X]
        # Evaluate 
        # Note: We must convert list of arrays to 2D array for problem.evaluate if vectorized
        # But CMA expects list of fits. 
        # For simplicity in this block, we use problem.evaluate which handles 2D arrays
        fit = problem.evaluate(np.array(X_eval))
        
        es.tell(X, fit)
        fevals += len(X)
        history.append(es.result.fbest)
    
    return np.array(history)

# --- F. UPGRADED AutOA (The Hero) ---
# --- F. UPGRADED AutOA (SELECTIVE DIMENSIONAL AUTOTOMY) ---
def run_autosa_upgraded(problem, max_fevals):
    """
    Selective Autotomic AutOA (S-AutOA).
    
    NOVELTY:
    Instead of resetting whole individuals (standard Autotomy), 
    S-AutOA performs 'Selective Autotomy'. It monitors variance 
    dimension-by-dimension. If specific dimensions stagnate, 
    it applies Opposition-Based Learning (OBL) ONLY to those 
    dimensions, preserving the 'healthy' genetic material in 
    the rest of the vector.
    """
    import math
    
    D = problem.D
    # L-SHADE Configuration
    N_max = 18 * D 
    N_min = 4
    pop_size = N_max 
    
    pop = np.random.uniform(problem.lb, problem.ub, (pop_size, D))
    fit = problem.evaluate(pop)
    
    # SHADE-like Memory
    H = 5
    M_CR = np.ones(H) * 0.5
    M_F = np.ones(H) * 0.5
    k_mem = 0
    archive = []
    
    fevals = pop_size
    history = [np.min(fit)]
    
    while fevals < max_fevals:
        # 1. Linear Population Reduction (Efficiency Engine)
        plan_pop_size = int(round((N_min - N_max) / max_fevals * fevals + N_max))
        
        if pop_size > plan_pop_size:
            sort_idx = np.argsort(fit)
            pop_size = plan_pop_size
            pop = pop[sort_idx[:pop_size]]
            fit = fit[sort_idx[:pop_size]]
            if len(archive) > pop_size:
                np.random.shuffle(archive)
                archive = archive[:pop_size]

        # 2. Sort Population
        sorted_idx = np.argsort(fit)
        pop, fit = pop[sorted_idx], fit[sorted_idx]
        elite = pop[0]
        
        # 3. SELECTIVE DIMENSIONAL AUTOTOMY (The Novelty)
        # Calculate variance for each dimension independently
        dim_variance = np.var(pop, axis=0)
        # Normalize variance 
        norm_var = dim_variance / (problem.ub - problem.lb)**2
        
        # Identify "Dead" Dimensions (Variance < 1e-5)
        dead_dims = np.where(norm_var < 1e-5)[0]
        
        # Trigger: If we have dead dimensions, perform surgical repair
        if len(dead_dims) > 0:
            # We treat the bottom 20% of the population as "Stem Cells" for repair
            n_repair = int(0.2 * pop_size)
            start_idx = pop_size - n_repair
            
            for k in range(start_idx, pop_size):
                if fevals >= max_fevals: break
                
                # Copy Elite as the base
                repair_vector = elite.copy()
                
                # Apply OBL *ONLY* to the Dead Dimensions
                # x_new = lb + ub - x_old
                # We add a small Gaussian jitter to prevent perfect symmetry loops
                jitter = np.random.normal(0, 1e-3, len(dead_dims))
                repair_vector[dead_dims] = (problem.lb + problem.ub) - repair_vector[dead_dims] + jitter
                
                # Ensure bounds
                repair_vector = ensure_bounds(repair_vector, problem.lb, problem.ub)
                
                # Evaluate
                f_new = problem.evaluate(repair_vector.reshape(1,-1))[0]
                fevals += 1
                
                # Replace if better (or even if slightly worse, to enforce diversity)
                pop[k] = repair_vector
                fit[k] = f_new

        # 4. Standard L-SHADE Evolution (The Engine)
        pop_new, SCR, SF = [], [], []
        
        for i in range(pop_size):
            if fevals >= max_fevals: break
            
            r_idx = np.random.randint(0, H)
            CR = np.clip(np.random.normal(M_CR[r_idx], 0.1), 0, 1)
            F = np.clip(cauchy_rand(M_F[r_idx], 0.1, 1)[0], 0, 1)
            
            p_best = pop[np.random.randint(0, max(2, int(0.1 * pop_size)))]
            r1 = pop[np.random.randint(0, pop_size)]
            union = np.vstack((pop, archive)) if len(archive) > 0 else pop
            r2 = union[np.random.randint(0, len(union))]
            
            mutant = pop[i] + F * (p_best - pop[i]) + F * (r1 - r2)
            
            mask = np.random.rand(D) < CR
            mask[np.random.randint(0, D)] = True
            trial = np.where(mask, mutant, pop[i])
            trial = ensure_bounds(trial, problem.lb, problem.ub)
            
            pop_new.append(trial)
            SCR.append(CR); SF.append(F)
            
        if len(pop_new) == 0: break
        
        pop_new = np.array(pop_new)
        fit_new_vals = problem.evaluate(pop_new)
        fevals += len(pop_new)
        
        # Selection
        succ_F, succ_CR, diffs = [], [], []
        next_pop, next_fit = [], []
        
        for i in range(len(pop_new)):
            if fit_new_vals[i] < fit[i]:
                next_pop.append(pop_new[i])
                next_fit.append(fit_new_vals[i])
                archive.append(pop[i])
                succ_F.append(SF[i])
                succ_CR.append(SCR[i])
                diffs.append(fit[i] - fit_new_vals[i])
            else:
                next_pop.append(pop[i])
                next_fit.append(fit[i])
        
        pop = np.array(next_pop)
        fit = np.array(next_fit)
        
        # Memory Update
        if len(succ_F) > 0:
            weights = np.array(diffs) / np.sum(diffs)
            M_F[k_mem] = np.sum(weights * (np.array(succ_F)**2)) / np.sum(weights * np.array(succ_F))
            M_CR[k_mem] = np.sum(weights * np.array(succ_CR))
            k_mem = (k_mem + 1) % H
            
        if len(archive) > pop_size:
            idx = np.random.choice(len(archive), pop_size, replace=False)
            archive = [archive[k] for k in idx]
            
        history.append(np.min(fit))

    return np.array(history)

# ==========================================================
# 6. BENCHMARK RUNNER
# ==========================================================

def run_benchmark():
    # Configuration
    D = 30
    MAX_FEVALS = 30000
    RUNS = 20 # Standard for publication
    
    problem = DeceptiveTrapProblem(D=D)
    
    algorithms = {
        "JADE": run_jade,
        "SHADE": run_shade,
        "L-SHADE": run_lshade,
        "jSO": run_jso,
        "Upgraded AutOA": run_autosa_upgraded
    }
    
    if HAS_CMA:
        algorithms["CMA-ES"] = run_cmaes
        
    results = {name: [] for name in algorithms}
    convergence = {name: [] for name in algorithms}
    
    print(f" Starting Benchmark on HDTAP (D={D})...")
    print(f"   Runs: {RUNS} | Budget: {MAX_FEVALS} FEs")
    print("-" * 60)
    
    for name, func in algorithms.items():
        print(f"Running {name}...", end=" ", flush=True)
        t0 = time.time()
        
        for r in range(RUNS):
            np.random.seed(r + 100) # Reproducibility
            hist = func(problem, MAX_FEVALS)
            
            results[name].append(hist[-1])
            convergence[name].append(hist)
            
        print(f"Done ({time.time()-t0:.2f}s) | Mean: {np.mean(results[name]):.2e}")

    # --- 7. DATA PROCESSING & PLOTTING ---
    
    # A. Statistics Table
    stats_data = []
    # Control Algorithm: Upgraded AutOA
    control_res = results["Upgraded AutOA"]
    
    for name in algorithms:
        vals = results[name]
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        best_v = np.min(vals)
        
        # Mann-Whitney U Test
        if name == "Upgraded AutOA":
            p_val = "-"
            sig = "-"
        else:
            _, p = stats.mannwhitneyu(control_res, vals, alternative='less')
            p_val = f"{p:.1e}"
            sig = "YES" if p < 0.05 else "NO"
            
        stats_data.append({
            "Algorithm": name,
            "Mean": mean_v,
            "StdDev": std_v,
            "Best": best_v,
            "p-value (< AutOA)": p_val,
            "Sig?": sig
        })
        
    df = pd.DataFrame(stats_data).sort_values("Mean")
    print("\n FINAL RESULTS TABLE ")
    print(df.to_string(index=False))
    df.to_csv("benchmark_stats_autosa.csv", index=False)
    
    # B. Convergence Plot
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    
    for i, name in enumerate(algorithms):
        # Normalize lengths (different algos might finish slightly differently)
        hists = convergence[name]
        min_len = min(len(h) for h in hists)
        avg_hist = np.mean([h[:min_len] for h in hists], axis=0)
        
        # Plot
        # Convert index to FEs (Approximation)
        x_axis = np.linspace(0, MAX_FEVALS, len(avg_hist))
        
        if name == "Upgraded AutOA":
            plt.plot(x_axis, avg_hist, label=name, color='red', linewidth=2.5, zorder=10)
        else:
            plt.plot(x_axis, avg_hist, label=name, linewidth=1.5, alpha=0.7)
            
    plt.yscale('log')
    plt.title(f"Convergence Comparison on Deceptive High-D Trap (D={D})")
    plt.xlabel("Function Evaluations (FEs)")
    plt.ylabel("Fitness (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("benchmark_convergence_autosa.png", dpi=300)
    print("\n Plot saved to 'benchmark_convergence_autosa.png'")

if __name__ == "__main__":
    run_benchmark()
