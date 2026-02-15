"""
Full benchmark for the Gear Train Design Problem (Version 5).

Final methodological corrections for publication:
1.  Statistical Test: Replaced the incorrect Wilcoxon signed-rank test
    (for paired data) with the correct Mann-Whitney U test
    (for independent data) using `scipy.stats.mannwhitneyu`.
2.  Seed Fix: Removed the hardcoded `seed=1` from SOTA DE and CMA-ES
    and replaced it with a random seed for each run. This ensures
    their 30 runs are also stochastic and statistically valid.

This script is now scientifically sound for publication.

Install dependencies:
pip install numpy matplotlib pandas scipy pyswarms cma
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import warnings
from scipy.optimize import differential_evolution
from scipy.stats import mannwhitneyu  # <-- Use Mann-Whitney U for independent samples
import pyswarms as ps
import cma
from cma.evolution_strategy import CMAEvolutionStrategy

# --- Experiment Setup ---

POP_SIZE = 30
MAX_FEVALS = 15000
MAX_RUNS = 30  # Standard for statistical significance

MAX_ITER = int(MAX_FEVALS / POP_SIZE)  # 500 iterations for O(N) algos
MAX_ITER_ABC = int(MAX_ITER / 2)       # 250 for ABC (fair FE budget)

# --- Problem Definition ---

bounds = [(12, 60)] * 4

def round_to_bounds(x):
    return np.clip(np.round(x), 12, 60)

def gear_train_obj_max(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        val = (x[0] * x[1]) / (x[2] * x[3])
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return val

def objective(x):
    x_cont = np.atleast_2d(x)
    x_int = round_to_bounds(x_cont)
    x1, x2, x3, x4 = x_int[:,0], x_int[:,1], x_int[:,2], x_int[:,3]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        val = np.full(x_int.shape[0], float('inf'))
        valid_mask = (x3 != 0) & (x4 != 0)
        if np.any(valid_mask):
            val[valid_mask] = -((x1[valid_mask]*x2[valid_mask]) / (x3[valid_mask]*x4[valid_mask]))
        val[np.isnan(val)] = float('inf')
        val[np.isinf(val)] = float('inf')
    return val[0] if val.size == 1 else val

# --- Optimization Algorithms ---

def abc_optimize(max_iter, pop_size):
    n = 4
    FoodNumber = pop_size
    limit = 100
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    Foods = np.random.uniform(lb, ub, (FoodNumber, n))
    Fitness = np.apply_along_axis(objective, 1, Foods)
    trial = np.zeros(FoodNumber)
    def prob():
        fit_vals = Fitness.copy()
        min_fit = np.min(fit_vals)
        if min_fit <= 0:
            fit_vals += abs(min_fit) + 1e-6
        inv_fitness = 1.0 / fit_vals
        return inv_fitness / np.sum(inv_fitness)
    BestIndex = np.argmin(Fitness)
    Best = Foods[BestIndex]
    BestScore = Fitness[BestIndex]
    convergence = []
    for t in range(max_iter):
        for i in range(FoodNumber):
            k = np.random.randint(FoodNumber)
            while k == i:
                k = np.random.randint(FoodNumber)
            phi = np.random.uniform(-1,1,size=n)
            vi = Foods[i] + phi * (Foods[i] - Foods[k])
            vi = np.clip(vi, lb, ub)
            vi_fitness = objective(vi)
            if vi_fitness < Fitness[i]:
                Foods[i] = vi
                Fitness[i] = vi_fitness
                trial[i] = 0
            else:
                trial[i] += 1
        prob_vals = prob()
        i = 0
        count = 0
        while count < FoodNumber:
            if np.random.rand() < prob_vals[i]:
                k = np.random.randint(FoodNumber)
                while k == i:
                    k = np.random.randint(FoodNumber)
                phi = np.random.uniform(-1,1,size=n)
                vi = Foods[i] + phi*(Foods[i] - Foods[k])
                vi = np.clip(vi, lb, ub)
                vi_fitness = objective(vi)
                if vi_fitness < Fitness[i]:
                    Foods[i] = vi
                    Fitness[i] = vi_fitness
                    trial[i] = 0
                else:
                    trial[i] += 1
                count += 1
            i = (i + 1) % FoodNumber
        max_trial_index = np.argmax(trial)
        if trial[max_trial_index] > limit:
            Foods[max_trial_index] = np.random.uniform(lb, ub)
            Fitness[max_trial_index] = objective(Foods[max_trial_index])
            trial[max_trial_index] = 0
        current_best_index = np.argmin(Fitness)
        if Fitness[current_best_index] < BestScore:
            BestIndex = current_best_index
            Best = Foods[current_best_index]
            BestScore = Fitness[current_best_index]
        convergence.append(-BestScore)
    return round_to_bounds(Best), convergence

def sca_optimize(max_iter, pop_size):
    n = 4
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    Population = np.random.uniform(lb, ub, (pop_size, n))
    fitness = np.apply_along_axis(objective, 1, Population)
    leader_idx = np.argmin(fitness)
    leader_pos = Population[leader_idx].copy()
    leader_score = fitness[leader_idx]
    convergence = []
    for t in range(max_iter):
        r1 = 2 - t*(2/max_iter)
        for i in range(pop_size):
            r2, r3, r4 = 2*np.pi*np.random.rand(), 2*np.random.rand(), np.random.rand()
            if r4 < 0.5:
                Population[i] += r1*np.sin(r2)*abs(r3*leader_pos - Population[i])
            else:
                Population[i] += r1*np.cos(r2)*abs(r3*leader_pos - Population[i])
            Population[i] = np.clip(Population[i], lb, ub)
        fitness = np.apply_along_axis(objective, 1, Population)
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < leader_score:
            leader_score = fitness[current_best_idx]
            leader_pos = Population[current_best_idx].copy()
        convergence.append(-leader_score)
    return round_to_bounds(leader_pos), convergence

def gwo_optimize(max_iter, pop_size):
    n = 4
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    Population = np.random.uniform(lb, ub, (pop_size, n))
    fitness = np.apply_along_axis(objective, 1, Population)
    Alpha_pos, Beta_pos, Delta_pos = np.zeros(n), np.zeros(n), np.zeros(n)
    Alpha_score, Beta_score, Delta_score = float('inf'), float('inf'), float('inf')
    convergence = []
    for t in range(max_iter):
        for i in range(pop_size):
            fit = fitness[i]
            if fit < Alpha_score:
                Delta_score, Delta_pos = Beta_score, Beta_pos.copy()
                Beta_score, Beta_pos = Alpha_score, Alpha_pos.copy()
                Alpha_score, Alpha_pos = fit, Population[i].copy()
            elif fit < Beta_score:
                Delta_score, Delta_pos = Beta_score, Beta_pos.copy()
                Beta_score, Beta_pos = fit, Population[i].copy()
            elif fit < Delta_score:
                Delta_score, Delta_pos = fit, Population[i].copy()
        a = 2 - t*(2/max_iter)
        for i in range(pop_size):
            for j, X in enumerate([Alpha_pos, Beta_pos, Delta_pos]):
                r1, r2 = np.random.rand(n), np.random.rand(n)
                A, C = 2*a*r1 - a, 2*r2
                D = abs(C*X - Population[i])
                X_new = X - A*D
                if j == 0: X1 = X_new
                elif j == 1: X2 = X_new
                else: X3 = X_new
            Population[i] = np.clip((X1 + X2 + X3)/3.0, lb, ub)
        fitness = np.apply_along_axis(objective, 1, Population)
        for i in range(pop_size):
            if fitness[i] < Alpha_score:
                Alpha_score, Alpha_pos = fitness[i], Population[i].copy()
        convergence.append(-Alpha_score)
    return round_to_bounds(Alpha_pos), convergence

def pso_optimize(max_iter, pop_size):
    options = {'c1': 1.49618, 'c2':1.49618, 'w':0.7298}
    bounds_min = np.array([b[0] for b in bounds])
    bounds_max = np.array([b[1] for b in bounds])
    optimizer = ps.single.GlobalBestPSO(n_particles=pop_size, dimensions=4, options=options, bounds=(bounds_min, bounds_max))
    cost, pos = optimizer.optimize(objective, iters=max_iter, verbose=False)
    convergence = [-val for val in optimizer.cost_history]
    return round_to_bounds(pos), convergence

def core_aoa_optimize(max_iter, pop_size):
    n = 4
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    Population = np.random.uniform(lb, ub, (pop_size, n))
    Fitness = np.apply_along_axis(objective, 1, Population)
    BestIndex = np.argmin(Fitness)
    Best = Population[BestIndex].copy()
    BestScore = Fitness[BestIndex]
    convergence = []
    for t in range(max_iter):
        mutation_rate = 0.1
        for i in range(pop_size):
            if i != BestIndex:
                mutant = Population[i] + mutation_rate*(Best - Population[i])*np.random.randn(n)
                mutant = np.clip(mutant, lb, ub)
                mutant_fitness = objective(mutant)
                if mutant_fitness < Fitness[i]:
                    Population[i] = mutant
                    Fitness[i] = mutant_fitness
        current_best_idx = np.argmin(Fitness)
        if Fitness[current_best_idx] < BestScore:
            BestIndex = current_best_idx
            Best = Population[current_best_idx].copy()
            BestScore = Fitness[current_best_idx]
        convergence.append(-BestScore)
    return round_to_bounds(Best), convergence

def adaptive_aoa_optimize(max_iter, pop_size, alpha=0.5):
    n = 4
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    Population = np.random.uniform(lb, ub, (pop_size, n))
    Fitness = np.apply_along_axis(objective, 1, Population)
    BestIndex = np.argmin(Fitness)
    Best = Population[BestIndex].copy()
    BestScore = Fitness[BestIndex]
    convergence = []
    def diversity(pop):
        return np.mean(np.std(pop, axis=0))
    for t in range(max_iter):
        div = diversity(Population)
        if div > 5:
            for i in range(pop_size):
                if i != BestIndex:
                    mutant = Population[i] + alpha*(Best - Population[i])*np.random.randn(n)
                    mutant = np.clip(mutant, lb, ub)
                    mutant_fitness = objective(mutant)
                    if mutant_fitness < Fitness[i]:
                        Population[i] = mutant
                        Fitness[i] = mutant_fitness
        else:
            num_sacrifice = int(alpha * pop_size)
            if num_sacrifice < 1: num_sacrifice = 1
            worst_indices = np.argsort(Fitness)[-num_sacrifice:]
            for idx in worst_indices:
                a,b,c = np.random.choice(pop_size, 3, replace=False)
                mutant = Population[a] + np.random.rand()*(Population[b] - Population[c])
                mutant = np.clip(mutant, lb, ub)
                mutant_fitness = objective(mutant)
                Population[idx] = mutant
                Fitness[idx] = mutant_fitness
        current_best_idx = np.argmin(Fitness)
        if Fitness[current_best_idx] < BestScore:
            BestIndex = current_best_idx
            Best = Population[current_best_idx].copy()
            BestScore = Fitness[current_best_idx]
        convergence.append(-BestScore)
    return round_to_bounds(Best), convergence

def hybrid_aoa_optimize(max_iter, pop_size):
    n = 4
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    Population = np.random.uniform(lb, ub, (pop_size, n))
    Fitness = np.apply_along_axis(objective, 1, Population)
    BestIndex = np.argmin(Fitness)
    Best = Population[BestIndex].copy()
    BestScore = Fitness[BestIndex]
    convergence = []
    def diversity(pop):
        return np.mean(np.std(pop, axis=0))
    for t in range(max_iter):
        if (t//50)%2 == 0:
            mutation_rate = 0.1
            for i in range(pop_size):
                if i != BestIndex:
                    mutant = Population[i] + mutation_rate*(Best - Population[i])*np.random.randn(n)
                    mutant = np.clip(mutant, lb, ub)
                    mutant_fitness = objective(mutant)
                    if mutant_fitness < Fitness[i]:
                        Population[i] = mutant
                        Fitness[i] = mutant_fitness
        else:
            div = diversity(Population)
            alpha = 0.5
            if div > 5:
                for i in range(pop_size):
                    if i != BestIndex:
                        mutant = Population[i] + alpha*(Best - Population[i])*np.random.randn(n)
                        mutant = np.clip(mutant, lb, ub)
                        mutant_fitness = objective(mutant)
                        if mutant_fitness < Fitness[i]:
                            Population[i] = mutant
                            Fitness[i] = mutant_fitness
            else:
                num_sacrifice = int(alpha*pop_size)
                if num_sacrifice < 1: num_sacrifice = 1
                worst_indices = np.argsort(Fitness)[-num_sacrifice:]
                for idx in worst_indices:
                    a,b,c = np.random.choice(pop_size, 3, replace=False)
                    mutant = Population[a] + np.random.rand()*(Population[b] - Population[c])
                    mutant = np.clip(mutant, lb, ub)
                    Population[idx] = mutant
                    Fitness[idx] = objective(mutant)
        current_best_idx = np.argmin(Fitness)
        if Fitness[current_best_idx] < BestScore:
            BestIndex = current_best_idx
            Best = Population[current_best_idx].copy()
            BestScore = Fitness[current_best_idx]
        convergence.append(-BestScore)
    return round_to_bounds(Best), convergence

# --- Tier 2: SOTA Champion Implementations ---

def sota_de_optimize(max_fevals, pop_size_target):
    n = 4
    popsize_multiplier = int(np.ceil(pop_size_target / n))
    actual_pop_size = popsize_multiplier * n
    max_iter_de = int(np.floor(max_fevals / actual_pop_size))
    convergence = []
    def callback(res, *args, **kwargs):
        current_fitness = objective(res)
        convergence.append(-current_fitness)
    result = differential_evolution(objective, bounds, strategy='best1bin',
                                    maxiter=max_iter_de, popsize=popsize_multiplier,
                                    callback=callback, polish=False, disp=False,
                                    seed=np.random.randint(1, 100000))
    if len(convergence) < max_iter_de:
        convergence.extend([convergence[-1]]*(max_iter_de - len(convergence)))
    if not convergence:
        convergence = [-result.fun]*max_iter_de
    return round_to_bounds(result.x), convergence[:max_iter_de]

def cma_es_optimize(max_fevals, pop_size):
    n = 4
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    x0 = np.random.uniform(lb, ub)
    sigma0 = (np.mean(ub) - np.mean(lb)) / 3.
    es = CMAEvolutionStrategy(x0, sigma0, {'bounds': [list(lb), list(ub)], 'popsize':pop_size,
                                           'maxfevals': max_fevals, 'verbose':-9,
                                           'seed':np.random.randint(1, 100000)})
    convergence = []
    max_iter_cma = int(np.floor(max_fevals / pop_size))
    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(s) for s in solutions]
        es.tell(solutions, fitnesses)
        convergence.append(-es.result.fbest)
    best_sol = es.result.xbest
    if len(convergence) < max_iter_cma:
        convergence.extend([convergence[-1]]*(max_iter_cma - len(convergence)))
    if not convergence:
        convergence = [-es.result.fbest]*max_iter_cma
    return round_to_bounds(best_sol), convergence[:max_iter_cma]

# --- Optimizer Dictionary ---

optimizers = {
    "ABC": lambda: abc_optimize(MAX_ITER_ABC, POP_SIZE),
    "SCA": lambda: sca_optimize(MAX_ITER, POP_SIZE),
    "GWO": lambda: gwo_optimize(MAX_ITER, POP_SIZE),
    "PSO (pyswarms)": lambda: pso_optimize(MAX_ITER, POP_SIZE),
    "SOTA DE (best1bin)": lambda: sota_de_optimize(MAX_FEVALS, POP_SIZE),
    "CMA-ES (cma)": lambda: cma_es_optimize(MAX_FEVALS, POP_SIZE),
    "Core AOA": lambda: core_aoa_optimize(MAX_ITER, POP_SIZE),
    "Adaptive AOA t=0.9": lambda: adaptive_aoa_optimize(MAX_ITER, POP_SIZE, alpha=0.9),
    "Adaptive AOA t=0.8": lambda: adaptive_aoa_optimize(MAX_ITER, POP_SIZE, alpha=0.8),
    "Adaptive AOA t=0.7": lambda: adaptive_aoa_optimize(MAX_ITER, POP_SIZE, alpha=0.7),
    "Adaptive AOA t=0.6": lambda: adaptive_aoa_optimize(MAX_ITER, POP_SIZE, alpha=0.6),
    "Adaptive AOA t=0.5": lambda: adaptive_aoa_optimize(MAX_ITER, POP_SIZE, alpha=0.5),
    "Adaptive AOA t=0.4": lambda: adaptive_aoa_optimize(MAX_ITER, POP_SIZE, alpha=0.4),
    "Adaptive AOA t=0.3": lambda: adaptive_aoa_optimize(MAX_ITER, POP_SIZE, alpha=0.3),
    "Adaptive AOA t=0.2": lambda: adaptive_aoa_optimize(MAX_ITER, POP_SIZE, alpha=0.2),
    "Adaptive AOA t=0.1": lambda: adaptive_aoa_optimize(MAX_ITER, POP_SIZE, alpha=0.1),
    "Hybrid AOA": lambda: hybrid_aoa_optimize(MAX_ITER, POP_SIZE)
}

# --- Main Loop & Storage ---

results_all = {name: [] for name in optimizers}
convergence_all = {name: [] for name in optimizers}
runtimes_all = {name: [] for name in optimizers}
raw_objective_vals = {key: [] for key in optimizers.keys()} # For stats

print(f"Starting benchmark: {MAX_RUNS} runs per method, POP_SIZE={POP_SIZE}, FE Budget={MAX_FEVALS}")
print("--"*20)

for run in range(MAX_RUNS):
    print(f"Run {run+1}/{MAX_RUNS}")
    for name, func in optimizers.items():
        start_time = time.time()
        best_sol, convergence = func()
        elapsed = time.time() - start_time

        final_val = gear_train_obj_max(best_sol)
        results_all[name].append(best_sol)
        convergence_all[name].append(convergence)
        runtimes_all[name].append(elapsed)
        raw_objective_vals[name].append(final_val) # Store final value for stats

        print(f"  {name:<20} finished in {elapsed:5.2f}s, Best Value = {final_val:.4f}")

# --- Statistical Summary ---

summary = []
for name in optimizers.keys():
    vals = raw_objective_vals[name] # Use stored values
    summary.append({
        "Optimizer": name,
        "Best": np.max(vals),
        "Worst": np.min(vals),
        "Mean": np.mean(vals),
        "StdDev": np.std(vals),
        "AvgRuntime(s)": np.mean(runtimes_all[name])
    })

df_summary = pd.DataFrame(summary).sort_values(by="Mean", ascending=False)
print("\nBenchmark Summary Over All Runs:")
print(df_summary.to_string(index=False))

# --- CORRECTED: Mann-Whitney U Test ---

print("\n" + "="*30 + " Statistical Test (p-values) " + "="*30)
# --- SET YOUR BEST AOA ALGORITHM HERE ---
# Find the best algo by mean from the summary
CONTROL_ALGORITHM_NAME = df_summary.iloc[0]["Optimizer"]
print(f"Control Algorithm (Best Mean): '{CONTROL_ALGORITHM_NAME}'")
# ---

control_results = raw_objective_vals[CONTROL_ALGORITHM_NAME]
stats_summary = []

for competitor_name in optimizers.keys():
    if competitor_name == CONTROL_ALGORITHM_NAME:
        continue

    competitor_results = raw_objective_vals[competitor_name]

    # Use Mann-Whitney U for independent samples
    # Test if Control > Competitor (alternative='greater')
    try:
        stat, p_value = mannwhitneyu(control_results, competitor_results, alternative='greater')
    except ValueError as e:
        # This can happen if all 30 values are identical
        p_value = 1.0
        print(f"Warning: Could not compute test for {competitor_name}. {e}")

    stats_summary.append({
        "Competitor": competitor_name,
        "p-value (Control > Competitor)": f"{p_value:.4e}",
        "Significant (p < 0.05)": "Yes" if p_value < 0.05 else "No"
    })

df_stats = pd.DataFrame(stats_summary).sort_values(by="p-value (Control > Competitor)")
print(f"Statistical comparison against control: '{CONTROL_ALGORITHM_NAME}'")
print(df_stats.to_string(index=False))


# --- Convergence Plot with Error Bars ---

plt.figure(figsize=(14,9))
for name in optimizers.keys():
    curves = convergence_all[name]
    # Pad all curves to the longest length (MAX_ITER) for a uniform plot
    maxlen = MAX_ITER
    padded = [c + [c[-1]]*(maxlen-len(c)) for c in curves if c] # Handle empty curves if any
    if not padded:
        continue # Skip if an algo failed completely

    avg_curve = np.mean(padded, axis=0)
    std_curve = np.std(padded, axis=0)

    plt.plot(avg_curve[:maxlen], label=name) # Plot up to maxlen
    plt.fill_between(range(maxlen), (avg_curve-std_curve)[:maxlen], (avg_curve+std_curve)[:maxlen], alpha=0.2)

plt.title(f"Average Convergence Curves ({MAX_RUNS} Runs)")
plt.xlabel("Iteration")
plt.ylabel("Best Objective Value (Maximization)")
plt.ylim(0, 30) # Adjust as needed
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='small')
plt.tight_layout()
plt.savefig("gear_train_average_convergence_v5.png")
print("\nSaved convergence plot to 'gear_train_average_convergence_v5.png'")
