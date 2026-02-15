# AutOA: Adaptive Autotomy-Based Optimization Algorithm
---
This repository provides the complete experimental framework used in the ESWA submission:

**AutOA: Adaptive Autotomy-Based Optimization with Diversity-Driven Mode Switching**

It contains the full implementation of Core, Hybrid, and Adaptive AutOA, along with all benchmark environments and baseline algorithms used in the experimental evaluation.

---

## Repository Structure

Gear_problem_code/   → Gear Train Design benchmark (engineering optimization)
MRPP/                → Multi-Robot Path Planning benchmark (robotics optimization)
HPO_run_codes/       → CNN Hyperparameter Optimization benchmark (machine learning)
README.md

---

## Implemented Algorithms

The repository includes:

- Core AutOA
- Hybrid AutOA
- Adaptive AutOA (t = 0.1 – 0.9 variants)

Baseline metaheuristics:

- PSO (Particle Swarm Optimization)
- GWO (Grey Wolf Optimizer)
- ABC (Artificial Bee Colony)
- SCA (Sine-Cosine Algorithm)
- DE (Differential Evolution)
- GA (Genetic Algorithm)
- ACO (Ant Colony Optimization)
- CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

All algorithms are executed under identical function evaluation (FE) budgets and population sizes to ensure fair comparison.

---

## Experimental Protocol

All experiments strictly follow the methodology described in the manuscript:

- 30 independent runs per algorithm
- Fixed function evaluation (FE) budget
- Equal population size across methods
- Mann–Whitney U statistical test for independent samples
- Convergence curves reported using mean ± standard deviation
- Log-scale visualization where appropriate

Independent random seeds were used for each run.

Due to the stochastic nature of metaheuristic algorithms, exact numerical values may vary slightly across executions. However, repeated independent runs consistently preserve the relative performance ranking and statistical conclusions reported in the manuscript.
All baseline algorithms were implemented or configured using commonly recommended parameter settings from the literature. No additional tuning was performed beyond these standard configurations.

---

## Installation

Python 3.9+ is recommended.

Install required packages:

``bash
pip install numpy scipy pandas matplotlib pyswarms cma tensorflow scikit-learn
Running the Benchmarks
1. Gear Train Design Benchmark
python Gear_problem_code/<gear_train_script_name>.py
This script will:

Execute 30 independent runs per method

Print statistical summary tables

Perform Mann–Whitney U tests

Generate convergence plots

2. Multi-Robot Path Planning (MRPP)
python MRPP/<mrpp_script_name>.py
This will:

Run the full multi-algorithm benchmark

Generate average convergence curves

Produce radar comparison charts

Save best-path visualizations

3. CNN Hyperparameter Optimization (HPO)
Place the dataset file:

PRSA_data_2010.1.1-2014.12.31.csv
inside the HPO_run_codes directory.

Then run:

python HPO_run_codes/<hpo_script_name>.py
This will:

Perform 30-run HPO benchmarking

Compare Adaptive AutOA variants against baselines

Train final CNN models

Output RMSE, MAE, MSE, R², and MAPE metrics

Generate convergence and prediction plots

---

## Reproducibility Statement:
All algorithms are evaluated under identical evaluation budgets and parameter settings to ensure fairness.

Although individual runs may produce slightly different numerical results due to stochastic initialization and sampling effects, repeated executions consistently preserve the performance ranking and statistical significance conclusions reported in the manuscript.

This repository contains all necessary implementation details to reproduce the experimental framework described in the paper.


---

## Hardware Notes:
Runtime may vary depending on CPU/GPU availability.
The CNN HPO benchmark is computationally heavier due to neural network training.

---

## License:
This project is licensed under the MIT License. See the LICENSE file for details.

