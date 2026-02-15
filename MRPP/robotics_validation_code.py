"""
Full benchmark for the Multi-Robot Path Planning (MRPP) Problem (Version 7).

This script is PART 1 of the robotics benchmark: The Metaheuristic Showdown.
It compares AOA variants against other metaheuristics (PSO, GWO, GA, ACO, etc.)
on a "waypoint optimization" version of the problem.

Methodology:
1.  Statistical Rigor: Runs the entire experiment for `MAX_RUNS = 30`.
2.  Fair FE Budget: Enforces a low (but fair) `MAX_FEVALS = 1500` budget
    for all algorithms to test *rapid, early-stage convergence*.
    (50 iters for O(N), 25 iters for ABC).
3.  All Competitors: Includes `GA` (Genetic Algorithm) and `ACO` (Ant Colony).
4.  Full Statistical Analysis: Automatically generates a summary table
    (Best, Worst, Mean, StdDev) and a Mann-Whitney U p-value table.
5.  Full Plotting:
    * INCLUDES THE RADAR PLOT (based on mean performance).
    * Plots average convergence (with error bands).
    * Plots the single best path found from all runs.

Install dependencies:
pip install numpy matplotlib pandas scipy pyswarms
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import warnings
from scipy.optimize import differential_evolution
from scipy.stats import mannwhitneyu
import pyswarms as ps

# --- 1. EXPERIMENT SETUP ---
GRID_SIZE = (1000, 1000)
NUM_ROBOTS = 3
NUM_WAYPOINTS = 3
DIM = NUM_ROBOTS * NUM_WAYPOINTS * 2 # 18 Dimensions

# --- Statistical & Budget Setup ---
POP_SIZE = 30
MAX_RUNS = 30  # Number of independent runs for statistical analysis
MAX_FEVALS = 1500  # Low budget to test rapid convergence (due to slow simulation)

# O(N) algos (pop_size FEs per iter)
MAX_ITER = int(MAX_FEVALS / POP_SIZE)  # 1500 / 30 = 50
# O(2N) algos (e.g., ABC)
MAX_ITER_ABC = int(MAX_ITER / 2)       # 50 / 2 = 25

print(f"--- STARTING MRPP METAHEURISTIC BENCHMARK (V7) ---")
print(f"Problem: {NUM_ROBOTS} Robots, {NUM_WAYPOINTS} Waypoints, {DIM} Dimensions")
print(f"Statistical: {MAX_RUNS} Runs per Algorithm")
print(f"Budget: POP_SIZE={POP_SIZE}, MAX_FEVALS={MAX_FEVALS}")
print(f"MAX_ITER (O(N)) = {MAX_ITER}, MAX_ITER (ABC) = {MAX_ITER_ABC}")
print("-" * 50)


# --- 2. PROBLEM DEFINITION & SIMULATOR ---

# --- Problem Constants ---
starts = [(0, 0), (0, 999), (999, 0)]
goals = [(999, 999), (999, 0), (0, 999)]

# Create Obstacles
obstacles = np.zeros(GRID_SIZE, dtype=int)
obstacles[500, 200:800] = 1  # horizontal wall
obstacles[200:800, 500] = 1  # vertical wall
obstacles[800, 700:950] = 1  # Small new wall
obstacles[100:300, 200] = 1  # Small new wall

# --- Bounds for Optimizers ---
lb = np.zeros(DIM)
ub = np.zeros(DIM)
for i in range(DIM):
    lb[i] = 0
    ub[i] = GRID_SIZE[i % 2] - 1
bounds_list = []
for i in range(DIM):
    bounds_list.append((0, GRID_SIZE[i % 2] - 1))

# --- Simulation Constants ---
MAX_STEPS_SIM = 300
MOVESTEP = 10

# Bresenham's line algorithm
def bresenham(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    if dx > dy:
        err = dx // 2
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy // 2
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x1, y1))
    return points

def is_valid(pos):
    x, y = int(round(pos[0])), int(round(pos[1]))
    if x < 0 or x >= GRID_SIZE[0] or y < 0 or y >= GRID_SIZE[1]:
        return False
    if obstacles[x, y] == 1:
        return False
    return True

def heuristic(pos, goal):
    # Manhattan distance
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def decode_solution_to_waypoints(solution):
    waypoint_lists = []
    idx = 0
    for r in range(NUM_ROBOTS):
        robot_wps = [starts[r]] # Start
        for _ in range(NUM_WAYPOINTS):
            wp_x = int(round(solution[idx]))
            wp_y = int(round(solution[idx+1]))
            robot_wps.append((wp_x, wp_y))
            idx += 2
        robot_wps.append(goals[r]) # Goal
        waypoint_lists.append(robot_wps)
    return waypoint_lists

def run_simulation(solution):
    """
    Runs the full simulation based on a solution vector.
    Returns: (full_paths, reached_goal, total_length, invalid_move_counts,
              waypoint_lists, vertex_collisions, edge_collisions)
    """
    waypoint_lists = decode_solution_to_waypoints(solution)
    full_paths = [[starts[r]] for r in range(NUM_ROBOTS)]
    current_wp_index = [1] * NUM_ROBOTS
    reached_goal = [False] * NUM_ROBOTS
    invalid_move_counts = [0] * NUM_ROBOTS # Hits obstacles

    total_length = 0
    vertex_collisions = 0
    edge_collisions = 0

    for step in range(MAX_STEPS_SIM):
        all_reached = True
        for r in range(NUM_ROBOTS):
            if reached_goal[r]:
                full_paths[r].append(goals[r])
                continue

            all_reached = False
            current_pos = full_paths[r][-1]

            # Check if goal is the next waypoint
            if current_wp_index[r] >= len(waypoint_lists[r]):
                reached_goal[r] = True
                full_paths[r].append(goals[r])
                continue

            target_wp = waypoint_lists[r][current_wp_index[r]]

            # Move towards target
            move_vec = (target_wp[0] - current_pos[0], target_wp[1] - current_pos[1])
            dist_to_target = np.linalg.norm(move_vec)

            if dist_to_target < MOVESTEP:
                tentative = target_wp
                current_wp_index[r] += 1
            else:
                move_vec = (move_vec / dist_to_target) * MOVESTEP
                tentative = (current_pos[0] + move_vec[0], current_pos[1] + move_vec[1])

            tentative = (int(round(tentative[0])), int(round(tentative[1])))

            # Check for wall collisions along the path segment
            valid_move = True
            for px, py in bresenham(current_pos[0], current_pos[1], tentative[0], tentative[1]):
                if not is_valid((px, py)):
                    valid_move = False
                    break

            if valid_move:
                full_paths[r].append(tentative)
                if tentative != current_pos:
                    total_length += heuristic(current_pos, tentative)
                if tentative == goals[r]:
                    reached_goal[r] = True
            else:
                # Hit a wall, stay in place
                full_paths[r].append(current_pos)
                invalid_move_counts[r] += 1

        if all_reached:
            break # All robots at goal

    # --- Collision Checking ---
    max_len_path = max(len(p) for p in full_paths)

    # Vertex collisions (two robots at the same node at the same time)
    for step in range(max_len_path):
        positions = {}
        for r in range(NUM_ROBOTS):
            pos = full_paths[r][min(step, len(full_paths[r])-1)]
            if pos in positions.values():
                vertex_collisions += 1
            positions[r] = pos

    # Edge collisions (two robots swapping nodes)
    for step in range(1, max_len_path):
        for i in range(NUM_ROBOTS):
            for j in range(i+1, NUM_ROBOTS):
                pos_i_now = full_paths[i][min(step, len(full_paths[i])-1)]
                pos_i_prev = full_paths[i][min(step-1, len(full_paths[i])-1)]
                pos_j_now = full_paths[j][min(step, len(full_paths[j])-1)]
                pos_j_prev = full_paths[j][min(step-1, len(full_paths[j])-1)]
                if pos_i_now == pos_j_prev and pos_i_prev == pos_j_now:
                    edge_collisions += 1

    return (full_paths, reached_goal, total_length,
            invalid_move_counts, waypoint_lists,
            vertex_collisions, edge_collisions)

# --- Fitness Function (The "Objective") ---
def fitness_function(solution):
    (full_paths, reached_goal, total_length,
     invalid_move_counts, waypoint_lists,
     vertex_collisions, edge_collisions) = run_simulation(solution)

    penalty = 0
    # Penalty for waypoints inside obstacles
    for r in range(NUM_ROBOTS):
        for wp in waypoint_lists[r][1:-1]: # Exclude start/goal
            if not is_valid(wp):
                penalty += 100000  # High penalty for invalid waypoint

    # Massive penalty for not reaching goal
    for r in range(NUM_ROBOTS):
        if not reached_goal[r]:
            final_pos = full_paths[r][-1]
            penalty += 1000000 + 1000 * heuristic(final_pos, goals[r])

    penalty += sum(invalid_move_counts) * 500  # Penalty for hitting walls
    penalty += vertex_collisions * 100000       # Penalty for robot-robot collision
    penalty += edge_collisions * 100000        # Penalty for robot-robot collision

    return total_length + penalty


# --- 3. OPTIMIZER LIBRARY ---

# --- Artificial Bee Colony (ABC) ---
def abc_optimize(fitness_func, lb, ub, dim, max_iter, pop_size):
    convergence = []
    FoodNumber = pop_size // 2
    limit = 5
    Foods = np.random.uniform(lb, ub, (FoodNumber, dim))
    Fitness = np.array([fitness_func(food) for food in Foods])
    trial = np.zeros(FoodNumber)
    BestIndex = np.argmin(Fitness)
    Best = Foods[BestIndex].copy()
    BestScore = Fitness[BestIndex]

    for t in range(max_iter):
        # Employed bees
        for i in range(FoodNumber):
            k = np.random.randint(FoodNumber)
            while k == i: k = np.random.randint(FoodNumber)
            phi = np.random.uniform(-1,1,dim)
            vi = Foods[i] + phi * (Foods[i] - Foods[k])
            vi = np.clip(vi, lb, ub)
            vi_fitness = fitness_func(vi)
            if vi_fitness < Fitness[i]:
                Foods[i], Fitness[i], trial[i] = vi, vi_fitness, 0
            else:
                trial[i] += 1

        fit_vals = Fitness.copy(); min_fit = np.min(fit_vals)
        if min_fit <= 0: fit_vals += abs(min_fit) + 1e-6
        with warnings.catch_warnings(): # Handle potential division by zero if all fit_vals are inf
            warnings.simplefilter("ignore")
            prob = (1.0 / fit_vals) / np.sum(1.0 / fit_vals)
            if np.isnan(prob).any(): prob = np.ones(FoodNumber) / FoodNumber

        # Onlooker bees
        i, count = 0, 0
        while count < FoodNumber:
            k = np.random.randint(FoodNumber)
            while k == i: k = np.random.randint(FoodNumber)
            if np.random.rand() < prob[i]:
                phi = np.random.uniform(-1,1,dim)
                vi = Foods[i] + phi * (Foods[i] - Foods[k])
                vi = np.clip(vi, lb, ub)
                vi_fitness = fitness_func(vi)
                if vi_fitness < Fitness[i]:
                    Foods[i], Fitness[i], trial[i] = vi, vi_fitness, 0
                else:
                    trial[i] += 1
                count += 1
            i = (i+1) % FoodNumber

        # Scout bees
        max_trial_index = np.argmax(trial)
        if trial[max_trial_index] > limit:
            Foods[max_trial_index] = np.random.uniform(lb, ub)
            Fitness[max_trial_index] = fitness_func(Foods[max_trial_index])
            trial[max_trial_index] = 0

        current_best_index = np.argmin(Fitness)
        if Fitness[current_best_index] < BestScore:
            BestIndex, Best, BestScore = current_best_index, Foods[current_best_index].copy(), Fitness[current_best_index]

        convergence.append(BestScore)
    return Best, BestScore, convergence

# --- Grey Wolf Optimizer (GWO) ---
def gwo_optimize(fitness_func, lb, ub, dim, max_iter, pop_size):
    convergence = []
    Population = np.random.uniform(lb, ub, (pop_size, dim))
    Fitness = np.array([fitness_func(ind) for ind in Population])

    Alpha_pos, Alpha_score = np.zeros(dim), float('inf')
    Beta_pos, Beta_score = np.zeros(dim), float('inf')
    Delta_pos, Delta_score = np.zeros(dim), float('inf')

    # Initial scan for leaders
    for i in range(pop_size):
        fit = Fitness[i]
        if fit < Alpha_score:
            Alpha_score, Alpha_pos = fit, Population[i].copy()
        elif fit < Beta_score:
            Beta_score, Beta_pos = fit, Population[i].copy()
        elif fit < Delta_score:
            Delta_score, Delta_pos = fit, Population[i].copy()

    convergence.append(Alpha_score) # Record best at iter 0

    for t in range(max_iter):
        a = 2 - t * 2 / max_iter # a decreases from 2 to 0

        # Update leader positions
        for i in range(pop_size):
            fit = Fitness[i]
            if fit < Alpha_score:
                Delta_score, Delta_pos = Beta_score, Beta_pos.copy()
                Beta_score, Beta_pos = Alpha_score, Alpha_pos.copy()
                Alpha_score, Alpha_pos = fit, Population[i].copy()
            elif fit < Beta_score:
                Delta_score, Delta_pos = Beta_score, Beta_pos.copy()
                Beta_score, Beta_pos = fit, Population[i].copy()
            elif fit < Delta_score:
                Delta_score, Delta_pos = fit, Population[i].copy()

        for i in range(pop_size):
            X = []
            for leader_pos in [Alpha_pos, Beta_pos, Delta_pos]:
                r1, r2 = np.random.rand(dim), np.random.rand(dim)
                A, C = 2 * a * r1 - a, 2 * r2
                D = abs(C * leader_pos - Population[i])
                X.append(leader_pos - A * D)
            new_pos = (X[0] + X[1] + X[2]) / 3
            Population[i] = np.clip(new_pos, lb, ub)
            Fitness[i] = fitness_func(Population[i])

        # Final check for new best
        current_best_idx = np.argmin(Fitness)
        if Fitness[current_best_idx] < Alpha_score:
            Alpha_score, Alpha_pos = Fitness[current_best_idx], Population[current_best_idx].copy()

        convergence.append(Alpha_score)
    return Alpha_pos, Alpha_score, convergence

# --- Particle Swarm Optimization (PSO) ---
def pso_optimize(fitness_func, lb, ub, dim, max_iter, pop_size):
    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
    optimizer = ps.single.GlobalBestPSO(n_particles=pop_size, dimensions=dim, options=options, bounds=(lb, ub))

    # pyswarms expects the objective function to take a 2D array (n_particles, n_dims)
    def obj_for_pso(x_2d):
      return np.array([fitness_func(ind) for ind in x_2d])

    cost, pos = optimizer.optimize(obj_for_pso, iters=max_iter, verbose=False)
    convergence = optimizer.cost_history
    return pos, cost, convergence

# --- Differential Evolution (DE) ---
def de_optimize(fitness_func, bounds_list, max_iter, pop_size):
    n = len(bounds_list)
    # Calculate popsize multiplier for scipy.de
    popsize_multiplier = int(np.ceil(pop_size / n))
    convergence = []

    # Callback to store convergence
    def callback(res, *args, **kwargs):
        # res is the OptimizeResult object in modern scipy
        try:
            convergence.append(res.fun)
        except AttributeError:
            # Fallback for older scipy: res is the solution vector
            convergence.append(fitness_function(res))

    result = differential_evolution(
        fitness_func, bounds_list,
        strategy='best1bin',
        maxiter=max_iter, popsize=popsize_multiplier,
        callback=callback, polish=False, disp=False,
        seed=np.random.randint(1, 100000)) # Random seed for each run

    # Pad convergence if DE finished early
    if len(convergence) < max_iter:
         convergence.extend([convergence[-1]] * (max_iter - len(convergence)))
    if not convergence: # Failsafe if callback failed
         convergence = [result.fun] * max_iter

    return result.x, result.fun, convergence[:max_iter]

# --- Sine-Cosine Algorithm (SCA) ---
def sca_optimize(fitness_func, lb, ub, dim, max_iter, pop_size):
    convergence = []
    Population = np.random.uniform(lb, ub, (pop_size, dim))
    Fitness = np.array([fitness_func(ind) for ind in Population])
    best_idx = np.argmin(Fitness)
    best = Population[best_idx].copy()
    BestScore = Fitness[best_idx]

    convergence.append(BestScore) # Record best at iter 0

    for t in range(max_iter):
        r1 = 2 - 2 * t / max_iter # a from 2 to 0

        for i in range(pop_size):
            r2 = 2 * np.pi * np.random.rand() # 0 to 2pi
            r3 = 2 * np.random.rand()         # 0 to 2
            r4 = np.random.rand()             # 0 to 1

            if r4 < 0.5:
                Population[i] += r1 * np.sin(r2) * abs(r3 * best - Population[i])
            else:
                Population[i] += r1 * np.cos(r2) * abs(r3 * best - Population[i])

            Population[i] = np.clip(Population[i], lb, ub)
            Fitness[i] = fitness_func(Population[i])

        current_best_idx = np.argmin(Fitness)
        if Fitness[current_best_idx] < BestScore:
            best_idx = current_best_idx
            best = Population[best_idx].copy()
            BestScore = Fitness[best_idx]

        convergence.append(BestScore)
    return best, BestScore, convergence

# --- Core AOA ---
def core_aoa_optimize(fitness_func, lb, ub, dim, max_iter, pop_size):
    convergence = []
    Population = np.random.uniform(lb, ub, (pop_size, dim))
    Fitness = np.array([fitness_func(ind) for ind in Population])
    best_idx = np.argmin(Fitness)
    best = Population[best_idx].copy()
    BestScore = Fitness[best_idx]

    convergence.append(BestScore) # Record best at iter 0

    for _ in range(max_iter):
        mutation_rate = 0.1
        for i in range(pop_size):
            if i != best_idx:
                mutant = Population[i] + mutation_rate * (best - Population[i]) * np.random.randn(dim)
                mutant = np.clip(mutant, lb, ub)
                mutant_fitness = fitness_func(mutant)
                if mutant_fitness < Fitness[i]:
                    Population[i] = mutant
                    Fitness[i] = mutant_fitness

        current_best_idx = np.argmin(Fitness)
        if Fitness[current_best_idx] < BestScore:
            best_idx = current_best_idx
            best = Population[best_idx].copy()
            BestScore = Fitness[current_best_idx]

        convergence.append(BestScore)
    return best, BestScore, convergence

# --- Adaptive AOA ---
def adaptive_aoa_optimize(fitness_func, lb, ub, dim, max_iter, pop_size, alpha=0.5):
    convergence = []
    Population = np.random.uniform(lb, ub, (pop_size, dim))
    Fitness = np.array([fitness_func(ind) for ind in Population])
    best_idx = np.argmin(Fitness)
    best = Population[best_idx].copy()
    BestScore = Fitness[best_idx]

    convergence.append(BestScore) # Record best at iter 0

    def diversity(pop): return np.mean(np.std(pop, axis=0))
    div_threshold = np.mean(ub - lb) * 0.05 # 5% of average search space

    for _ in range(max_iter):
        div = diversity(Population)
        if div > div_threshold: # Exploit
            for i in range(pop_size):
                if i != best_idx:
                    mutant = Population[i] + alpha * (best - Population[i]) * np.random.randn(dim)
                    mutant = np.clip(mutant, lb, ub)
                    mutant_fitness = fitness_func(mutant)
                    if mutant_fitness < Fitness[i]:
                        Population[i] = mutant
                        Fitness[i] = mutant_fitness
        else: # Explore (DE-style "autotomy")
            num_sacrifice = int(alpha * pop_size); num_sacrifice = max(1, num_sacrifice)
            worst_indices = np.argsort(Fitness)[-num_sacrifice:]
            for idx in worst_indices:
                a, b, c = np.random.choice(pop_size, 3, replace=False)
                mutant = Population[a] + 0.5 * (Population[b] - Population[c]) # DE/rand/1
                mutant = np.clip(mutant, lb, ub)
                mutant_fitness = fitness_func(mutant)
                # Replace worst
                Population[idx] = mutant
                Fitness[idx] = mutant_fitness

        current_best_idx = np.argmin(Fitness)
        if Fitness[current_best_idx] < BestScore:
            best_idx = current_best_idx
            best = Population[best_idx].copy()
            BestScore = Fitness[current_best_idx]

        convergence.append(BestScore)
    return best, BestScore, convergence

# --- Hybrid AOA ---
def hybrid_aoa_optimize(fitness_func, lb, ub, dim, max_iter, pop_size):
    convergence = []
    Population = np.random.uniform(lb, ub, (pop_size, dim))
    Fitness = np.array([fitness_func(ind) for ind in Population])
    best_idx = np.argmin(Fitness)
    best = Population[best_idx].copy()
    BestScore = Fitness[best_idx]

    convergence.append(BestScore) # Record best at iter 0

    def diversity(pop): return np.mean(np.std(pop, axis=0))
    div_threshold = np.mean(ub - lb) * 0.05

    for t in range(max_iter):
        if (t // 10) % 2 == 0: # Core AOA part (Exploitation)
            mutation_rate = 0.1
            for i in range(pop_size):
                if i != best_idx:
                    mutant = Population[i] + mutation_rate * (best - Population[i]) * np.random.randn(dim)
                    mutant = np.clip(mutant, lb, ub)
                    mutant_fitness = fitness_func(mutant)
                    if mutant_fitness < Fitness[i]:
                        Population[i] = mutant
                        Fitness[i] = mutant_fitness
        else: # Adaptive AOA part (Exploration)
            alpha = 0.5 # Fixed alpha for this part
            div = diversity(Population)
            if div > div_threshold: # Still diverse, just do a normal adaptive move
                for i in range(pop_size):
                    if i != best_idx:
                        mutant = Population[i] + alpha * (best - Population[i]) * np.random.randn(dim)
                        mutant = np.clip(mutant, lb, ub)
                        mutant_fitness = fitness_func(mutant)
                        if mutant_fitness < Fitness[i]:
                            Population[i] = mutant
                            Fitness[i] = mutant_fitness
            else: # Low diversity, trigger autotomy
                num_sacrifice = int(alpha * pop_size); num_sacrifice = max(1, num_sacrifice)
                worst_indices = np.argsort(Fitness)[-num_sacrifice:]
                for idx in worst_indices:
                    a, b, c = np.random.choice(pop_size, 3, replace=False)
                    mutant = Population[a] + 0.5 * (Population[b] - Population[c])
                    mutant = np.clip(mutant, lb, ub)
                    mutant_fitness = fitness_func(mutant)
                    Population[idx] = mutant
                    Fitness[idx] = mutant_fitness

        current_best_idx = np.argmin(Fitness)
        if Fitness[current_best_idx] < BestScore:
            best_idx = current_best_idx
            best = Population[best_idx].copy()
            BestScore = Fitness[current_best_idx]

        convergence.append(BestScore)
    return best, BestScore, convergence

# --- Your Custom AOA ---
def custom_aoa_optimize(fitness_func, max_iter, pop_size):
    convergence = []
    population = np.random.uniform(lb, ub, (pop_size, DIM))
    best_solution = None
    best_fitness = np.inf

    convergence.append(best_fitness) # Record best at iter 0 (inf)

    for t in range(max_iter):
        # 1. Manage population size
        current_pop_size = len(population)
        if current_pop_size < pop_size:
            new_blood = np.random.uniform(lb, ub, (pop_size - current_pop_size, DIM))
            population = np.vstack([population, new_blood])
        elif current_pop_size > pop_size:
            population = population[np.random.choice(current_pop_size, pop_size, replace=False)]

        # 2. Evaluate
        fitnesses = np.array([fitness_func(ind) for ind in population])

        # 3. Find current best
        idx_best = np.argmin(fitnesses)
        if fitnesses[idx_best] < best_fitness:
            best_fitness = fitnesses[idx_best]
            best_solution = population[idx_best].copy()

        # 4. Survivor Selection (Autotomy)
        threshold = np.percentile(fitnesses, 30) # Keep top 30%
        survivors = population[fitnesses <= threshold]
        if len(survivors) == 0: survivors = population[np.array([idx_best])]

        # 5. Regeneration
        n_regen = max(1, int(0.3 * pop_size)) # Regenerate 30%
        regen_indices = np.random.choice(len(survivors), n_regen)
        regenerated = survivors[regen_indices] + np.random.normal(0, 75.0, (n_regen, DIM))
        regenerated = np.clip(regenerated, lb, ub)

        # 6. New Population
        population = np.vstack([survivors, regenerated])

        # 7. Add noise/mutation
        noise = np.random.normal(0, 50.0, population.shape)
        population = np.clip(population + noise, lb, ub)

        # 8. Elitism
        if best_solution is not None:
             population[0] = best_solution

        convergence.append(best_fitness)

    if best_solution is None: # Failsafe
        best_solution = population[np.argmin(fitnesses)]

    return best_solution, best_fitness, convergence

# --- NEW: Genetic Algorithm (GA) ---
def ga_optimize(fitness_func, lb, ub, dim, max_iter, pop_size):
    convergence = []
    Population = np.random.uniform(lb, ub, (pop_size, dim))
    Fitness = np.array([fitness_func(ind) for ind in Population])
    best_idx = np.argmin(Fitness)
    Best = Population[best_idx].copy()
    BestScore = Fitness[best_idx]

    convergence.append(BestScore) # Record best at iter 0

    CXPB, MUTPB = 0.7, 0.2

    for _ in range(max_iter):
        # --- Selection (Tournament) ---
        offspring = []
        for _ in range(pop_size):
            i1, i2 = np.random.choice(pop_size, 2, replace=False)
            winner = i1 if Fitness[i1] < Fitness[i2] else i2
            offspring.append(Population[winner].copy())

        # --- Crossover (Simulated Binary Crossover - SBX) ---
        for i in range(0, pop_size, 2):
            if np.random.rand() < CXPB and i+1 < pop_size:
                p1, p2 = offspring[i], offspring[i+1]
                u = np.random.rand(dim)
                # eta = 20
                beta = np.where(u <= 0.5, (2*u)**(1.0/21.0), (1.0/(2.0*(1.0-u)))**(1.0/21.0))
                c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
                offspring[i] = np.clip(c1, lb, ub)
                offspring[i+1] = np.clip(c2, lb, ub)

        # --- Mutation (Gaussian) ---
        for i in range(pop_size):
            if np.random.rand() < MUTPB:
                offspring[i] += np.random.normal(0, 50.0, dim) # 50-pixel std dev
                offspring[i] = np.clip(offspring[i], lb, ub)

        # --- Evaluation & Elitism ---
        Offspring_Fitness = np.array([fitness_func(ind) for ind in offspring])

        # Combine and select best
        Combined_Pop = np.vstack((Population, offspring))
        Combined_Fitness = np.concatenate((Fitness, Offspring_Fitness))

        best_indices = np.argsort(Combined_Fitness)[:pop_size]
        Population = Combined_Pop[best_indices]
        Fitness = Combined_Fitness[best_indices]

        if Fitness[0] < BestScore:
            Best, BestScore = Population[0].copy(), Fitness[0]

        convergence.append(BestScore)

    return Best, BestScore, convergence

# --- NEW: Ant Colony Optimization (ACO-R, for continuous) ---
def aco_optimize(fitness_func, lb, ub, dim, max_iter, pop_size):
    convergence = []
    # Parameters
    archive_size = 10 # K (Number of "elite" ants)
    q = 0.1  # Intensification factor (std dev of Gaussian)
    xi = 0.85 # Diversification factor (importance of other solutions)

    # Initialize solution archive (K solutions)
    archive = np.random.uniform(lb, ub, (archive_size, dim))
    archive_fitness = np.array([fitness_func(sol) for sol in archive])
    archive_order = np.argsort(archive_fitness)
    archive = archive[archive_order]
    archive_fitness = archive_fitness[archive_order]

    Best, BestScore = archive[0].copy(), archive_fitness[0]

    convergence.append(BestScore) # Record best at iter 0

    for _ in range(max_iter):
        # --- Calculate weights (pheromone) for each elite ant ---
        weights = 1.0 / (np.sqrt(2 * np.pi) * q * archive_size) * \
                  np.exp(-0.5 * np.arange(archive_size)**2 / (q**2 * archive_size**2))
        weights = weights / np.sum(weights)

        new_solutions = np.zeros((pop_size, dim))
        for i in range(pop_size):
            # --- Select Gaussian kernel (ant) ---
            l = np.random.choice(archive_size, p=weights)

            # --- Sample from this kernel (ant moves) ---
            std_devs = np.zeros(dim)
            for d in range(dim):
                # Calculate avg distance to other solutions
                std_devs[d] = xi * np.sum(np.abs(archive[l, d] - archive[:, d])) / (archive_size - 1 + 1e-9)

            # Generate new solution
            new_sol = np.random.normal(archive[l], std_devs)
            new_sol = np.clip(new_sol, lb, ub)
            new_solutions[i] = new_sol

        # --- Evaluate new solutions ---
        new_fitnesses = np.array([fitness_func(sol) for sol in new_solutions])

        # --- Update Archive ---
        Combined_Pop = np.vstack((archive, new_solutions))
        Combined_Fitness = np.concatenate((archive_fitness, new_fitnesses))

        best_indices = np.argsort(Combined_Fitness)[:archive_size]
        archive = Combined_Pop[best_indices]
        archive_fitness = Combined_Fitness[best_indices]

        if archive_fitness[0] < BestScore:
            Best, BestScore = archive[0].copy(), archive_fitness[0]

        convergence.append(BestScore)

    return Best, BestScore, convergence


# --- 4. VISUALIZATION FUNCTIONS ---

def plot_static_path(solution, title="Static Path", save_path=None):
    if solution is None:
        print(f"Cannot plot {title}: No solution found.")
        return

    # Run simulation to get fresh path data
    (static_paths, _, _, _,
     waypoints, _, _) = run_simulation(solution)

    plt.figure(figsize=(12, 10))
    # Plot Obstacles
    obstacle_coords = np.argwhere(obstacles == 1)
    plt.scatter(obstacle_coords[:, 0], obstacle_coords[:, 1], c='black', s=1, marker='s', label='Obstacles')

    colors = plt.cm.jet(np.linspace(0, 1, NUM_ROBOTS))

    for i in range(NUM_ROBOTS):
        path = static_paths[i]
        if not path: continue

        # Trim path to end at goal
        trimmed_path = []
        for pos in path:
            trimmed_path.append(pos)
            if pos == goals[i]:
                break

        xs, ys = zip(*trimmed_path)
        plt.plot(xs, ys, color=colors[i], label=f'Robot {i+1} Path', linestyle=':', lw=2)

        # Plot Start
        plt.scatter(starts[i][0], starts[i][1], c=[colors[i]], marker='s', s=200, edgecolors='black', zorder=5)
        # Plot Goal
        plt.scatter(goals[i][0], goals[i][1], c=[colors[i]], marker='*', s=400, edgecolors='black', zorder=5, label=f'Robot {i+1} Goal')
        # Plot Final position
        plt.scatter(path[-1][0], path[-1][1], c=[colors[i]], marker='o', s=100, edgecolors='black', zorder=5)
        # Plot Waypoints
        if len(waypoints[i]) > 2: # Has at least one waypoint
            wp_xs, wp_ys = zip(*waypoints[i][1:-1])
            plt.scatter(wp_xs, wp_ys, c=[colors[i]], marker='x', s=150, zorder=4, label=f'Robot {i+1} WPs')

    plt.title(title, fontsize=16)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.xlim(0, GRID_SIZE[0])
    plt.ylim(0, GRID_SIZE[1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved path plot to {save_path}")
    plt.show()

def plot_radar_chart(df_results, save_path=None):
    """
    Plots a radar chart. Expects df_results to be the summary dataframe
    with algorithms as the index.
    """
    print("\n--- Plotting Radar Chart (1 = Best Performance) ---")

    # Use the MEAN values from the summary table
    metrics = df_results[['Time (s)', 'Total Collisions', 'Path Length']].copy()

    # Normalize Data (0-1, where 1 is BEST)
    # (max - x) / (max - min) for cost metrics (less is better)
    for col in metrics.columns:
        min_val = metrics[col].min()
        max_val = metrics[col].max()
        if (max_val - min_val) == 0:
            metrics[col] = 1.0 # All are equally good
        else:
            metrics[col] = (max_val - metrics[col]) / (max_val - min_val)

    labels = metrics.columns
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Close the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Use a color-blind-friendly colormap
    colors = plt.cm.tab20(np.linspace(0, 1, len(metrics)))

    for i, (index, row) in enumerate(metrics.iterrows()):
        values = row.values.tolist()
        values += values[:1] # Close the loop
        ax.plot(angles, values, label=index, color=colors[i], linewidth=2, linestyle='solid')
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_rlabel_position(0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["Worst", "", "Avg", "", "Best"])
    ax.set_ylim(0, 1)

    plt.title('Mean Optimizer Performance (1 = Best)', size=20, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.45, 1.1))
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved radar chart to {save_path}")
    plt.show()

def plot_average_convergence(convergence_data, max_iter, save_path=None):
    print("\n--- Plotting Average Convergence ---")
    plt.figure(figsize=(14, 9))

    colors = plt.cm.tab20(np.linspace(0, 1, len(convergence_data)))

    for i, (name, curves) in enumerate(convergence_data.items()):
        # Pad all curves to the longest length (MAX_ITER) for a uniform plot
        maxlen = max_iter

        # Ensure curves are at least maxlen long, padding with last value
        padded = []
        for c in curves:
            if not c: # Handle empty convergence list
                padded.append([np.inf] * maxlen)
                continue
            c_padded = c + [c[-1]] * (maxlen - len(c))
            padded.append(c_padded[:maxlen]) # Truncate/pad to maxlen

        if not padded:
            continue # Skip if an algo failed completely

        avg_curve = np.mean(padded, axis=0)
        std_curve = np.std(padded, axis=0)

        plt.plot(avg_curve, label=name, color=colors[i], lw=2)
        plt.fill_between(range(maxlen), (avg_curve-std_curve), (avg_curve+std_curve), color=colors[i], alpha=0.15)

    plt.title(f"Average Convergence Curves ({MAX_RUNS} Runs, {MAX_FEVALS} FEs)")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness (Cost) - Log Scale")
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.yscale('log') # Fitness spans orders of magnitude, log scale is essential
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved convergence plot to {save_path}")
    plt.show()


# --- 5. MAIN EXPERIMENT ---

# --- Optimizer Dictionary (All inclusive) ---
optimizers = {
    # Tier 1 Baselines
    "PSO": lambda: pso_optimize(fitness_function, lb, ub, DIM, MAX_ITER, POP_SIZE),
    "GWO": lambda: gwo_optimize(fitness_function, lb, ub, DIM, MAX_ITER, POP_SIZE),

    # Tier 2 Path-Finding Metaheuristics
    "GA": lambda: ga_optimize(fitness_function, lb, ub, DIM, MAX_ITER, POP_SIZE),
    "ACO": lambda: aco_optimize(fitness_function, lb, ub, DIM, MAX_ITER, POP_SIZE),

    # Other Baselines from your code
    "ABC": lambda: abc_optimize(fitness_function, lb, ub, DIM, MAX_ITER_ABC, POP_SIZE),
    "SCA": lambda: sca_optimize(fitness_function, lb, ub, DIM, MAX_ITER, POP_SIZE),
    "DE": lambda: de_optimize(fitness_function, bounds_list, MAX_ITER, POP_SIZE),

    # Your AOA Variants
    "Core AOA": lambda: core_aoa_optimize(fitness_function, lb, ub, DIM, MAX_ITER, POP_SIZE),
    "Hybrid AOA": lambda: hybrid_aoa_optimize(fitness_function, lb, ub, DIM, MAX_ITER, POP_SIZE),
    "Custom AOA": lambda: custom_aoa_optimize(fitness_function, MAX_ITER, POP_SIZE),

    "Adaptive AOA t=0.1": lambda: adaptive_aoa_optimize(fitness_function, lb, ub, DIM, MAX_ITER, POP_SIZE, alpha=0.1),
    "Adaptive AOA t=0.2": lambda: adaptive_aoa_optimize(fitness_function, lb, ub, DIM, MAX_ITER, POP_SIZE, alpha=0.2),
    "Adaptive AOA t=0.3": lambda: adaptive_aoa_optimize(fitness_function, lb, ub, DIM, MAX_ITER, POP_SIZE, alpha=0.3),
    "Adaptive AOA t=0.4": lambda: adaptive_aoa_optimize(fitness_function, lb, ub, DIM, MAX_ITER, POP_SIZE, alpha=0.4),
    "Adaptive AOA t=0.5": lambda: adaptive_aoa_optimize(fitness_function, lb, ub, DIM, MAX_ITER, POP_SIZE, alpha=0.5),
    "Adaptive AOA t=0.6": lambda: adaptive_aoa_optimize(fitness_function, lb, ub, DIM, MAX_ITER, POP_SIZE, alpha=0.6),
    "Adaptive AOA t=0.7": lambda: adaptive_aoa_optimize(fitness_function, lb, ub, DIM, MAX_ITER, POP_SIZE, alpha=0.7),
    "Adaptive AOA t=0.8": lambda: adaptive_aoa_optimize(fitness_function, lb, ub, DIM, MAX_ITER, POP_SIZE, alpha=0.8),
    "Adaptive AOA t=0.9": lambda: adaptive_aoa_optimize(fitness_function, lb, ub, DIM, MAX_ITER, POP_SIZE, alpha=0.9),


}
# Note: I've trimmed the 't' variants to 3 for a cleaner plot.
# You can add all 9 back in if you wish.


# --- Main Experiment Loop ---
results_all_runs = defaultdict(list)
convergence_all_runs = defaultdict(list)
runtimes_all_runs = defaultdict(list)
raw_objective_vals = defaultdict(list)

# Global best tracking
global_best_fitness = np.inf
global_best_solution = None
global_best_algo = ""

for run in range(1, MAX_RUNS + 1):
    print(f"\n--- Starting Run {run} / {MAX_RUNS} ---")
    for name, func in optimizers.items():
        start_time = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # Suppress runtime warnings during optimization
            best_sol, best_fit, convergence = func()

        elapsed = time.time() - start_time

        # Store results
        results_all_runs[name].append(best_sol)
        convergence_all_runs[name].append(convergence)
        runtimes_all_runs[name].append(elapsed)
        raw_objective_vals[name].append(best_fit)

        # Update global best
        if best_fit < global_best_fitness:
            global_best_fitness = best_fit
            global_best_solution = best_sol
            global_best_algo = name

        print(f"  {name:<20} finished in {elapsed:6.2f}s. Best Fitness = {best_fit:,.2f}")

print("\n--- EXPERIMENT COMPLETE ---")


# --- 6. RESULTS & ANALYSIS ---

# 6.1. Statistical Summary Table
summary = []
for name in optimizers.keys():
    vals = np.array(raw_objective_vals[name])

    # Get metrics from the best solution of this algo's 30 runs
    best_run_idx = np.argmin(vals)
    best_sol_for_algo = results_all_runs[name][best_run_idx]

    # Rerun simulation on the best solution to get clean metrics
    (_, reached, path_len, wall_hits, _, v_cols, e_cols) = run_simulation(best_sol_for_algo)
    total_cols = sum(wall_hits) + v_cols + e_cols
    # Check if ALL robots reached their goal
    goal_reached = "Yes" if np.all(reached) else "No"

    summary.append({
        "Optimizer": name,
        "Best": np.min(vals),
        "Worst": np.max(vals),
        "Mean": np.mean(vals),
        "StdDev": np.std(vals),
        "Avg Runtime (s)": np.mean(runtimes_all_runs[name]),
        "Path Length (of Best)": path_len,
        "Collisions (of Best)": total_cols,
        "Goal Reached? (of Best)": goal_reached
    })

df_summary = pd.DataFrame(summary).sort_values(by="Mean")
df_summary_for_radar = df_summary.set_index("Optimizer")
# Rename columns for radar
df_summary_for_radar = df_summary_for_radar.rename(columns={
    "Avg Runtime (s)": "Time (s)",
    "Path Length (of Best)": "Path Length",
    "Collisions (of Best)": "Total Collisions"
})

print("\n" + "="*30 + " Benchmark Summary Over 30 Runs " + "="*30)
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
    # Format floating point numbers for readability
    print(df_summary.to_string(index=False, float_format="%.2f"))


# 6.2. Statistical Test (Mann-Whitney U)
print("\n" + "="*30 + " Statistical Test (p-values) " + "="*30)
# Set the control algorithm to the one with the best Mean fitness
CONTROL_ALGORITHM_NAME = df_summary.iloc[0]["Optimizer"]
print(f"Control Algorithm (Best Mean): '{CONTROL_ALGORITHM_NAME}'")

control_results = raw_objective_vals[CONTROL_ALGORITHM_NAME]
stats_summary = []

for competitor_name in optimizers.keys():
    if competitor_name == CONTROL_ALGORITHM_NAME:
        continue
    competitor_results = raw_objective_vals[competitor_name]

    # Test if Control < Competitor (alternative='less') since lower fitness is better
    try:
        stat, p_value = mannwhitneyu(control_results, competitor_results, alternative='less')
    except ValueError as e:
        # This can happen if all 30 values are identical (e.g., all failed)
        p_value = 1.0
        print(f"Warning: Could not compute test for {competitor_name}. {e}")

    stats_summary.append({
        "Competitor": competitor_name,
        "p-value (Control < Competitor)": f"{p_value:.4e}",
        "Significant (p < 0.05)": "Yes" if p_value < 0.05 else "No"
    })

df_stats = pd.DataFrame(stats_summary).sort_values(by="p-value (Control < Competitor)")
print(f"Statistical comparison against control: '{CONTROL_ALGORITHM_NAME}'")
print(df_stats.to_string(index=False))


# 6.3. Average Convergence Plot
plot_average_convergence(convergence_all_runs, MAX_ITER,
                         save_path="mrpp_average_convergence.png")

# 6.4. Radar Plot
plot_radar_chart(df_summary_for_radar,
                 save_path="mrpp_radar_chart.png")

# 6.5. Best Overall Path Plot
print(f"\n--- Plotting Best Overall Path from '{global_best_algo}' (Fitness: {global_best_fitness:,.2f}) ---")
plot_static_path(global_best_solution,
                 title=f"Best Overall Path: {global_best_algo}\nFitness: {global_best_fitness:.2f}",
                 save_path="mrpp_best_overall_path.png")