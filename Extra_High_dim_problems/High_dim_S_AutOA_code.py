"""
High-Dimensional Publishable Benchmark Suite
AutOA vs SOTA
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings
import time
import torch


# Attempt to import CMA for IPOP-CMA-ES
try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    print(" 'cma' library not found. IPOP-CMA-ES will be skipped. (pip install cma)")

warnings.filterwarnings('ignore')


# ==========================================================
# GPU DETECTION (Informational Only)
# ==========================================================
if torch.cuda.is_available():
    print("GPU detected:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU (NumPy-based implementation).")

# ==========================================================
# PROBLEM
# ==========================================================
class DeceptiveTrapProblem:
    def __init__(self, D):
        self.D = D
        self.lb = -100
        self.ub = 100

    def evaluate(self, population):
        X = np.array(population)
        rastrigin = 10 * self.D + np.sum(X**2 - 10*np.cos(2*np.pi*X), axis=1)
        trap_mask = np.abs(X) > 5.12
        trap_penalty = np.sum(np.abs(X) * 100, axis=1)
        fitness = rastrigin + np.where(np.any(trap_mask, axis=1), trap_penalty, 0)
        return fitness


# ==========================================================
# ==========================================================
# 2. HELPER FUNCTIONS
# ==========================================================
def ensure_bounds(X, lb, ub):
    return np.clip(X, lb, ub)

def cauchy_rand(loc, scale, size):
    return loc + scale * np.tan(np.pi * (np.random.rand(size) - 0.5))

# ==========================================================
# 3. ALGORITHM: L-SHADE-EpSin (Adaptive Benchmark)
# ==========================================================
def run_lshade_epsin(problem, max_fevals):
    """
    Approximation of L-SHADE-cnEpSin.
    Key Feature: Sinusoidal Parameter Adaptation.
    """
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
    freq = 0.5 # Frequency for sinusoidal adaptation
    
    while fevals < max_fevals:
        # Linear Population Reduction
        plan_pop_size = int(round((N_min - N_max) / max_fevals * fevals + N_max))
        if pop_size > plan_pop_size:
            sort_idx = np.argsort(fit)
            pop_size = plan_pop_size
            pop = pop[sort_idx[:pop_size]]
            fit = fit[sort_idx[:pop_size]]
            if len(archive) > pop_size:
                np.random.shuffle(archive)
                archive = archive[:pop_size]

        # Sort
        sorted_idx = np.argsort(fit)
        pop, fit = pop[sorted_idx], fit[sorted_idx]
        
        pop_new, SCR, SF = [], [], []
        
        for i in range(pop_size):
            if fevals >= max_fevals: break
            
            r_idx = np.random.randint(0, H)
            
            # --- SINUSOIDAL ADAPTATION (The EpSin feature) ---
            # Modulate CR and F with a Sinusoidal wave to enforce diversity cycles
            sin_val = np.sin(2 * np.pi * freq * (fevals / max_fevals))
            
            mu_cr = M_CR[r_idx] + 0.1 * sin_val
            mu_f  = M_F[r_idx]  + 0.1 * sin_val
            
            CR = np.clip(np.random.normal(mu_cr, 0.1), 0, 1)
            F  = np.clip(cauchy_rand(mu_f, 0.1, 1)[0], 0, 1)
            
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
        
        # Standard SHADE Update
        succ_F, succ_CR, diffs = [], [], []
        next_pop, next_fit = [], []
        for i in range(len(pop_new)):
            if fit_new_vals[i] < fit[i]:
                next_pop.append(pop_new[i])
                next_fit.append(fit_new_vals[i])
                archive.append(pop[i])
                succ_F.append(SF[i]); succ_CR.append(SCR[i])
                diffs.append(fit[i] - fit_new_vals[i])
            else:
                next_pop.append(pop[i])
                next_fit.append(fit[i])
        
        pop, fit = np.array(next_pop), np.array(next_fit)
        
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
# 4. ALGORITHM: MadDE (High-D Benchmark)
# ==========================================================
def run_madde(problem, max_fevals):
    """
    Approximation of MadDE (Multi-Adaptive DE).
    Key Feature: Multi-Strategy Composition (Pool of Strategies).
    """
    D = problem.D
    pop_size = 100 # MadDE typically keeps larger populations
    pop = np.random.uniform(problem.lb, problem.ub, (pop_size, D))
    fit = problem.evaluate(pop)
    archive = []
    
    fevals = pop_size
    history = [np.min(fit)]
    
    # Strategy Pool Probabilities
    # Strat 1: DE/current-to-pbest/1 (Exploitation)
    # Strat 2: DE/rand-to-pbest/1 (Exploration)
    prob_strat = 0.5 
    
    while fevals < max_fevals:
        sorted_idx = np.argsort(fit)
        pop, fit = pop[sorted_idx], fit[sorted_idx]
        
        pop_new, strategies_used = [], []
        
        for i in range(pop_size):
            if fevals >= max_fevals: break
            
            # Select Strategy
            strat = 1 if np.random.rand() < prob_strat else 2
            strategies_used.append(strat)
            
            F, CR = 0.5, 0.9 # Simplified fixed params for clarity, MadDE adapts these too
            
            p_best = pop[np.random.randint(0, max(2, int(0.1 * pop_size)))]
            r1 = pop[np.random.randint(0, pop_size)]
            r2 = pop[np.random.randint(0, pop_size)]
            
            if strat == 1: # current-to-pbest
                mutant = pop[i] + F * (p_best - pop[i]) + F * (r1 - r2)
            else: # rand-to-pbest
                mutant = r1 + F * (p_best - r1) + F * (r2 - pop[i])
                
            mask = np.random.rand(D) < CR
            mask[np.random.randint(0, D)] = True
            trial = np.where(mask, mutant, pop[i])
            trial = ensure_bounds(trial, problem.lb, problem.ub)
            pop_new.append(trial)
            
        if len(pop_new) == 0: break
        pop_new = np.array(pop_new)
        fit_new_vals = problem.evaluate(pop_new)
        fevals += len(pop_new)
        
        # Selection & Adaptation of Probabilities
        succ_1, succ_2 = 0, 0
        next_pop, next_fit = [], []
        
        for i in range(len(pop_new)):
            if fit_new_vals[i] < fit[i]:
                next_pop.append(pop_new[i])
                next_fit.append(fit_new_vals[i])
                if strategies_used[i] == 1: succ_1 += 1
                else: succ_2 += 1
            else:
                next_pop.append(pop[i])
                next_fit.append(fit[i])
        
        pop, fit = np.array(next_pop), np.array(next_fit)
        
        # Adapt Strategy Probability
        total_succ = succ_1 + succ_2
        if total_succ > 0:
            prob_strat = 0.1 * prob_strat + 0.9 * (succ_1 / total_succ)
        prob_strat = np.clip(prob_strat, 0.1, 0.9)
            
        history.append(np.min(fit))
    return np.array(history)

# ==========================================================
# 5. ALGORITHM: IPOP-CMA-ES (Restart Benchmark)
# ==========================================================
def run_ipop_cmaes(problem, max_fevals):
    """
    Restart-CMA-ES.
    Key Feature: Restarts with doubled population upon stagnation.
    """
    if not HAS_CMA: return np.zeros(100)
    
    D = problem.D
    fevals = 0
    history = []
    
    restart_count = 0
    pop_size_multiplier = 1
    
    while fevals < max_fevals:
        # Calculate new population size (IPOP: increase size)
        base_pop = int(10 + 2 * np.sqrt(D))
        pop_size = min(base_pop * pop_size_multiplier, 4 * base_pop)

        
        # Initialize CMA
        x0 = np.random.uniform(problem.lb, problem.ub, D)
        sigma0 = (problem.ub - problem.lb) / 4.0
        
        try:
            es = cma.CMAEvolutionStrategy(x0, sigma0, {
                'popsize': pop_size, 
                'verbose': -9,
                'maxfevals': max_fevals - fevals
            })
        except:
            break
            
        # Run local restart
        while not es.stop() and fevals < max_fevals:
            X = es.ask()
            X_eval = [ensure_bounds(np.array(x), problem.lb, problem.ub) for x in X]
            
            # Need to convert to array for vector eval
            fit = problem.evaluate(np.array(X_eval))
            es.tell(X, fit)
            
            fevals += len(X)
            history.append(es.result.fbest)
            
            # Pad history for plotting if batch size > 1
            for _ in range(len(X)-1):
                if len(history) < max_fevals: history.append(es.result.fbest)
        
        # Prepare for next restart
        pop_size_multiplier *= 2
        restart_count += 1
        
    return np.array(history)

# ==========================================================
# 6. ALGORITHM: S-AutOA (The HERO - Selective Autotomy)
# ==========================================================
def run_s_autoa(problem, max_fevals):
    """
    Selective Autotomic AutOA.
    Strategy: Selective Dimensional OBL + L-SHADE backbone.
    """
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
        # 1. Linear Reduction
        plan_pop_size = int(round((N_min - N_max) / max_fevals * fevals + N_max))
        if pop_size > plan_pop_size:
            sort_idx = np.argsort(fit)
            pop_size = plan_pop_size
            pop = pop[sort_idx[:pop_size]]
            fit = fit[sort_idx[:pop_size]]
            if len(archive) > pop_size:
                np.random.shuffle(archive)
                archive = archive[:pop_size]

        sorted_idx = np.argsort(fit)
        pop, fit = pop[sorted_idx], fit[sorted_idx]
        elite = pop[0]
        
        # 2. SELECTIVE DIMENSIONAL AUTOTOMY
        # Calculate variance dimension-wise
        dim_variance = np.var(pop, axis=0)
        norm_var = dim_variance / ((problem.ub - problem.lb)**2 + 1e-9)
        
        # Identify "Dead" Dimensions
        dead_dims = np.where(norm_var < 1e-6)[0]
        
        if len(dead_dims) > 0:
            # Repair the tail (Bottom 20%)
            n_repair = max(1, int(0.2 * pop_size))
            start_idx = pop_size - n_repair
            
            for k in range(start_idx, pop_size):
                if fevals >= max_fevals: break
                
                # Copy Elite
                repair_vec = elite.copy()
                
                # Apply OBL only to dead dimensions
                # Add jitter to break symmetry
                jitter = np.random.normal(0, 1e-3, len(dead_dims))
                repair_vec[dead_dims] = (problem.lb + problem.ub) - repair_vec[dead_dims] + jitter
                
                repair_vec = ensure_bounds(repair_vec, problem.lb, problem.ub)
                f_new = problem.evaluate(repair_vec.reshape(1,-1))[0]
                fevals += 1
                
                # Replace
                pop[k] = repair_vec
                fit[k] = f_new

        # 3. L-SHADE Evolution
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
                succ_F.append(SF[i]); succ_CR.append(SCR[i])
                diffs.append(fit[i] - fit_new_vals[i])
            else:
                next_pop.append(pop[i])
                next_fit.append(fit[i])
        
        pop, fit = np.array(next_pop), np.array(next_fit)
        
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


# ==========================================================
# EXPERIMENT DRIVER
# ==========================================================
def run_dimension(D, max_fevals=30000, runs=20):

    print(f"\n===== Running Dimension D = {D} =====")

    problem = DeceptiveTrapProblem(D)

    algorithms = {
        "MadDE": run_madde,
        "L-SHADE-EpSin": run_lshade_epsin,
        "IPOP-CMA-ES": run_ipop_cmaes,
        "S-AutOA": run_s_autoa
    }

    results = {}
    histories = {}

    for name, func in algorithms.items():
        finals = []
        hist_runs = []

        t0 = time.time()

        for r in range(runs):
            np.random.seed(100 + r)
            hist = func(problem, max_fevals)
            finals.append(hist[-1])
            hist_runs.append(hist)

        t1 = time.time()

        results[name] = np.array(finals)
        histories[name] = hist_runs

        print(f"{name:<18} Mean={np.mean(finals):.3e} "
              f"Std={np.std(finals):.3e} "
              f"Time/Run={(t1-t0)/runs:.2f}s")

    return results, histories


# ==========================================================
# PLOTS
# ==========================================================

def plot_convergence(histories, D):
    plt.figure(figsize=(8,6))

    for name, hist in histories.items():
         min_len = min(len(h) for h in hist)
         hist = np.array([h[:min_len] for h in hist])

         mean_curve = np.mean(hist, axis=0)
         std_curve = np.std(hist, axis=0)
         x = np.arange(len(mean_curve))

         plt.plot(x, mean_curve, label=name)
         plt.fill_between(
              x,
              mean_curve - std_curve,
              mean_curve + std_curve,
              alpha=0.2
               )


    plt.yscale("log")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Best Fitness (log scale)")
    plt.title(f"Convergence Curves (D={D})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_boxplot(results, D):
    plt.figure(figsize=(8,6))
    plt.boxplot(results.values(), labels=results.keys())
    plt.yscale("log")
    plt.ylabel("Final Fitness (log scale)")
    plt.title(f"Final Distribution (D={D})")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()


def plot_scaling(all_results):
    plt.figure(figsize=(8,6))

    dimensions = sorted(all_results.keys())
    algorithms = list(next(iter(all_results.values())).keys())

    for algo in algorithms:
        means = [np.mean(all_results[D][algo]) for D in dimensions]
        plt.plot(dimensions, means, marker='o', label=algo)

    plt.yscale("log")
    plt.xlabel("Dimension (D)")
    plt.ylabel("Mean Final Fitness (log scale)")
    plt.title("Scalability Analysis")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================================
# STATISTICAL TEST
# ==========================================================
def statistical_test(results):
    hero = results["S-AutOA"]

    print("\nStatistical Significance (Mann-Whitney U vs S-AutOA)")
    for name, values in results.items():
        if name == "S-AutOA":
            continue

        _, p = stats.mannwhitneyu(hero, values, alternative="less")
        verdict = "WIN" if p < 0.05 else "TIE/LOSS"
        print(f"vs {name:<18}: p={p:.2e} -> {verdict}")


# ==========================================================
# MAIN MULTI-D STUDY
# ==========================================================
def main():

    dimensions = [30, 100, 250]
    all_results = {}

    for D in dimensions:
        results, histories = run_dimension(D)
        all_results[D] = results

        plot_convergence(histories, D)
        plot_boxplot(results, D)
        statistical_test(results)

    plot_scaling(all_results)


if __name__ == "__main__":
    main()

