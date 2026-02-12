import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import warnings
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import mannwhitneyu

# --- 0. CONFIGURATION ---
MAX_RUNS = 10        # Run 10 times for a statistical average (30 is better if you have time)
MAX_FEVALS = 50      # Give each optimizer 50 evaluations (50 evals / 10 pop = 5 iterations)
POP_SIZE = 10
HPO_EPOCHS = 20      # Train for 20 epochs during the HPO race
TIME_STEPS = 24 * 3

FINAL_EPOCHS = 100   # Train the final model for 100 epochs
FINAL_PATIENCE = 10  # Stop if validation loss doesn't improve for 10 epochs


DATA_FILE = 'PRSA_data_2010.1.1-2014.12.31.csv'
PLOTS_DIR = 'cnn_benchmark_plots'
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

HPO_BOUNDS_LOW = [0.0] * 5
HPO_BOUNDS_HIGH = [1.0] * 5
HPO_DIM = 5

# --- HPO Parameter Decoding ---
def decode_params(cont_params):
    """Converts a continuous vector [0,1]^5 into a discrete set of hyperparameters."""
    num_layers = 1 if cont_params[0] < 0.5 else 2
    if cont_params[1] < 0.33:
        num_filters = 16
    elif cont_params[1] < 0.66:
        num_filters = 32
    else:
        num_filters = 64

    if cont_params[2] < 0.33:
        kernel_size = 3
    elif cont_params[2] < 0.66:
        kernel_size = 5
    else:
        kernel_size = 7

    if cont_params[3] < 0.33:
        batch_size = 16
    elif cont_params[3] < 0.66:
        batch_size = 32
    else:
        batch_size = 64

    min_lr, max_lr = 1e-4, 1e-2
    learning_rate = 10**(np.log10(min_lr) + cont_params[4] * (np.log10(max_lr) - np.log10(min_lr)))

    return {
        "num_layers": int(num_layers),
        "num_filters": int(num_filters),
        "kernel_size": int(kernel_size),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate)
    }

# --- DATA PREPROCESSING ---
def create_timeseries_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def load_and_preprocess_data():
    print("Loading and preprocessing data...")
    if not os.path.exists(DATA_FILE):
        print(f"Data file '{DATA_FILE}' missing, downloading...")
        try:
            # Note: This URL is a placeholder and won't work.
            # You must have the 'PRSA_data_2010.1.1-2014.12.31.csv' file locally.
            url = "PRSA_data_2010.1.1-2014.12.31.csv"
            df = pd.read_csv(url)
            df.to_csv(DATA_FILE, index=False)
            print(f"Downloaded '{DATA_FILE}' successfully.")
        except Exception as e:
            print(f"Download failed: {e}")
            raise FileNotFoundError(f"Please provide {DATA_FILE} in the same directory.")
    
    df = pd.read_csv(DATA_FILE)
    df['pm2.5'].bfill(inplace=True)
    df['pm2.5'].ffill(inplace=True)
    df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])
    df.set_index('datetime', inplace=True)
    df.drop(['No','year','month','day','hour','cbwd'], axis=1, inplace=True)
    TARGET_COL = 'pm2.5'
    FEATURES = [col for col in df.columns if col != TARGET_COL]
    
    # Split data for HPO (short train/val) and Final (long train/test)
    hpo_train_df = df.loc['2010-01-01':'2011-12-31']
    hpo_val_df = df.loc['2012-01-01':'2012-12-31']
    final_train_df = df.loc['2010-01-01':'2013-12-31']
    final_test_df = df.loc['2014-01-01':'2014-12-31']
    
    # Scale HPO data
    feature_scaler = StandardScaler()
    hpo_train_scaled_features = feature_scaler.fit_transform(hpo_train_df[FEATURES])
    hpo_val_scaled_features = feature_scaler.transform(hpo_val_df[FEATURES])
    
    target_scaler = StandardScaler()
    hpo_train_scaled_target = target_scaler.fit_transform(hpo_train_df[[TARGET_COL]])
    hpo_val_scaled_target = target_scaler.transform(hpo_val_df[[TARGET_COL]])
    
    hpo_train_scaled = pd.DataFrame(hpo_train_scaled_features, columns=FEATURES, index=hpo_train_df.index)
    hpo_train_scaled[TARGET_COL] = hpo_train_scaled_target
    hpo_val_scaled = pd.DataFrame(hpo_val_scaled_features, columns=FEATURES, index=hpo_val_df.index)
    hpo_val_scaled[TARGET_COL] = hpo_val_scaled_target

    X_hpo_train, y_hpo_train = create_timeseries_dataset(hpo_train_scaled[FEATURES], hpo_train_scaled[TARGET_COL], TIME_STEPS)
    X_hpo_val, y_hpo_val = create_timeseries_dataset(hpo_val_scaled[FEATURES], hpo_val_scaled[TARGET_COL], TIME_STEPS)
    print(f"Shapes - X_train: {X_hpo_train.shape}, X_val: {X_hpo_val.shape}")

    # Return HPO data, Final data, and the fitted target_scaler
    return (X_hpo_train, y_hpo_train), (X_hpo_val, y_hpo_val), (final_train_df, final_test_df), target_scaler

# --- CNN MODEL BUILDING ---
def build_model(params, time_steps, num_features):
    model = Sequential()
    model.add(Input(shape=(time_steps, num_features)))
    for _ in range(params['num_layers']):
        model.add(Conv1D(filters=params['num_filters'], kernel_size=params['kernel_size'], activation='relu', padding='causal'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse')
    return model, model.count_params()

# --- FITNESS FUNCTIONS ---
_FITNESS_FUNC_DATA = {} # Global dict to hold HPO data

def objective_function(cont_params):
    """The main fitness function. Takes a [0,1]^5 vector and returns validation loss."""
    global _FITNESS_FUNC_DATA
    
    # 1. Decode vector into model parameters
    params = decode_params(cont_params)
    
    try:
        # 2. Build model
        model, _ = build_model(params, TIME_STEPS, _FITNESS_FUNC_DATA['n_features'])
        
        # 3. Train model (for few epochs)
        history = model.fit(_FITNESS_FUNC_DATA['X_train'], _FITNESS_FUNC_DATA['y_train'],
                            validation_data=(_FITNESS_FUNC_DATA['X_val'], _FITNESS_FUNC_DATA['y_val']),
                            epochs=HPO_EPOCHS, batch_size=params['batch_size'], verbose=0)
        
        # 4. Get final validation loss
        val_loss = history.history['val_loss'][-1]
        
        if np.isnan(val_loss) or np.isinf(val_loss):
            return np.inf # Penalize models that diverge
        return val_loss
    except Exception:
        # Penalize models that fail to build (e.g., bad parameter combination)
        return np.inf

# --- OPTIMIZER RUNNERS ---
def run_optimizer(optimizer_func, run_num):
    """
    A generic wrapper to run any optimizer function.
    It handles timing, error catching, and history padding.
    """
    start_time = time.time()
    convergence_history = [] # This list is PASSED BY REFERENCE to the optimizer_func
    
    try:
        # This function (e.g., run_my_aoa) MUST populate the convergence_history list
        best_params_vector, best_loss = optimizer_func(convergence_history)
    except Exception as e:
        print(f"  ERROR in optimizer: {e}")
        best_params_vector = None
        best_loss = np.inf

    elapsed = time.time() - start_time
    print(f"  Run {run_num+1:02d} complete in {elapsed:.2f}s. Best Loss: {best_loss:.4f}")
    
    # Pad convergence history to exactly MAX_FEVALS for plotting
    if len(convergence_history) < MAX_FEVALS:
        pad_value = convergence_history[-1] if convergence_history else np.inf
        convergence_history += [pad_value] * (MAX_FEVALS - len(convergence_history))
        
    return best_loss, best_params_vector, convergence_history[:MAX_FEVALS], elapsed

# ==============================================================================
# ==============================================================================
#
# ---  ✅ PLACEHOLDER: ADD YOUR OPTIMIZER IMPLEMENTATIONS HERE ---
#
# Hybrid AOA
def run_hybrid_aoa(history_list):
    lb, ub, dim = HPO_BOUNDS_LOW, HPO_BOUNDS_HIGH, HPO_DIM
    pop_size = POP_SIZE
    max_iter = int(MAX_FEVALS / pop_size)

    # Helper function to calculate population diversity
    def diversity(pop): 
        return np.mean(np.std(pop, axis=0))

    # --- Initialization ---
    Population = np.random.uniform(lb, ub, (pop_size, dim))
    Fitness = np.full(pop_size, np.inf)

    best_pos = np.zeros(dim)
    best_score = float('inf')

    # Initial evaluation
    for i in range(pop_size):
        Fitness[i] = objective_function(Population[i])
        if Fitness[i] < best_score:
            best_score = Fitness[i]
            best_pos = Population[i].copy()
        history_list.append(best_score) # Add best-so-far

    # --- Main Loop ---
    div_threshold = 0.05 # Diversity threshold
    alpha = 0.5          # Parameter for adaptive phase
    mutation_rate = 0.1  # Parameter for core phase

    for t in range(max_iter - 1): # -1 for initial eval
        new_population = Population.copy()
        new_fitness = Fitness.copy()

        # Strategy 1: Core AOA (Exploration Phase)
        # Runs for 10 iterations, then switches
        if (t // 10) % 2 == 0: 
            for i in range(pop_size):
                mutant = Population[i] + mutation_rate * (best_pos - Population[i]) * np.random.randn(dim)
                mutant = np.clip(mutant, lb, ub)
                mutant_fitness = objective_function(mutant)

                if mutant_fitness < Fitness[i]:
                    new_population[i], new_fitness[i] = mutant, mutant_fitness
                
                if mutant_fitness < best_score:
                    best_score = mutant_fitness
                    best_pos = mutant.copy()
                history_list.append(best_score)

        # Strategy 2: Adaptive AOA (Exploitation Phase)
        else:
            div = diversity(Population)
            if div > div_threshold:
                # High diversity strategy (adaptive mutation)
                for i in range(pop_size):
                    mutant = Population[i] + alpha * (best_pos - Population[i]) * np.random.randn(dim)
                    mutant = np.clip(mutant, lb, ub)
                    mutant_fitness = objective_function(mutant)
                    if mutant_fitness < Fitness[i]:
                        new_population[i], new_fitness[i] = mutant, mutant_fitness
                    
                    if mutant_fitness < best_score:
                        best_score = mutant_fitness
                        best_pos = mutant.copy()
                    history_list.append(best_score)
            else:
                # Low diversity strategy (DE-like exploitation)
                num_sacrifice = max(1, int(alpha * pop_size))
                worst_indices = np.argsort(Fitness)[-num_sacrifice:]
                
                for i in range(pop_size):
                    if i in worst_indices:
                        a, b, c = np.random.choice(pop_size, 3, replace=False)
                        mutant = Population[a] + 0.5 * (Population[b] - Population[c])
                    else:
                        mutant = Population[i] + 0.1 * (best_pos - Population[i]) * np.random.randn(dim)
                    
                    mutant = np.clip(mutant, lb, ub)
                    mutant_fitness = objective_function(mutant)
                    new_population[i], new_fitness[i] = mutant, mutant_fitness

                    if mutant_fitness < best_score:
                        best_score = mutant_fitness
                        best_pos = mutant.copy()
                    history_list.append(best_score)
        
        # Update population for next iteration
        Population = new_population
        Fitness = new_fitness

    return best_pos, best_score
#
# ==============================================================================
#
# --- EXAMPLE: Random Search (follows the template) ---
# Note: This is a simple case where 1 eval = 1 iteration
#
def run_random_search(history_list):
    best_loss = np.inf
    best_params = None
    
    for _ in range(MAX_FEVALS):
        # 1. Generate a solution
        cont_params = np.random.rand(HPO_DIM)
        
        # 2. Evaluate it
        loss = objective_function(cont_params)
        
        # 3. Update best
        if loss < best_loss:
            best_loss = loss
            best_params = cont_params
            
        # 4. Append the *current best* to history
        history_list.append(best_loss) 
        
    # 5. Return final best
    return best_params, best_loss

# ==============================================================================
# ==============================================================================


# --- FINAL VALIDATION ---
def run_final_validation(optimizer_name, best_cont_params, final_train_df, final_test_df, target_scaler):
    """
    Trains the best model found on the *full* dataset and tests it.
    """
    print(f"\n--- Final Validation: {optimizer_name} ---")
    
    # 1. Decode best parameters
    params = decode_params(best_cont_params)
    print(f"  Params: {params}")
    
    # 2. Prepare full dataset
    TARGET_COL = 'pm2.5'
    FEATURES = [col for col in final_train_df.columns if col != TARGET_COL]
    
    feature_scaler_final = StandardScaler()
    final_train_scaled_features = feature_scaler_final.fit_transform(final_train_df[FEATURES])
    final_test_scaled_features = feature_scaler_final.transform(final_test_df[FEATURES])
    
    # Use the *original* target_scaler from HPO data to prevent data leakage
    final_train_scaled_target = target_scaler.transform(final_train_df[[TARGET_COL]])
    final_test_scaled_target = target_scaler.transform(final_test_df[[TARGET_COL]])
    
    final_train_scaled = pd.DataFrame(final_train_scaled_features, columns=FEATURES, index=final_train_df.index)
    final_train_scaled[TARGET_COL] = final_train_scaled_target
    final_test_scaled = pd.DataFrame(final_test_scaled_features, columns=FEATURES, index=final_test_df.index)
    final_test_scaled[TARGET_COL] = final_test_scaled_target
    
    X_train, y_train = create_timeseries_dataset(final_train_scaled[FEATURES], final_train_scaled[TARGET_COL], TIME_STEPS)
    X_test, y_test = create_timeseries_dataset(final_test_scaled[FEATURES], final_test_scaled[TARGET_COL], TIME_STEPS)
    print(f"  Final shapes: X_train {X_train.shape}, X_test {X_test.shape}")
    
    # 3. Build and train final model
    model, total_params = build_model(params, TIME_STEPS, len(FEATURES))
    early_stopping = EarlyStopping(monitor='val_loss', patience=FINAL_PATIENCE, restore_best_weights=True)
    
    start_time = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=FINAL_EPOCHS,
                        batch_size=params['batch_size'], verbose=1, callbacks=[early_stopping])
    train_time = time.time() - start_time
    
    # 4. Evaluate
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform to get real values
    y_test_real = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_real = target_scaler.inverse_transform(y_pred_scaled)

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        if not np.any(mask): return 0.0
        y_true_masked, y_pred_masked = y_true[mask].flatten(), y_pred[mask].flatten()
        if y_true_masked.size == 0: return 0.0
        return np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100

    metrics = {
        "Optimizer": optimizer_name,
        "RMSE": np.sqrt(mean_squared_error(y_test_real, y_pred_real)),
        "MAE": mean_absolute_error(y_test_real, y_pred_real),
        "MSE": mean_squared_error(y_test_real, y_pred_real),
        "R2": r2_score(y_test_real, y_pred_real),
        "MAPE": mean_absolute_percentage_error(y_test_real, y_pred_real),
        "Train Time (s)": train_time,
        "Total Params": total_params,
        "Best HPO Params": str(params)
    }
    print(f"  Final Validation Complete. RMSE: {metrics['RMSE']:.4f}, R2: {metrics['R2']:.4f}")

    # 5. Plot and save final results
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss Curve - {optimizer_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, f"FINAL_LOSS_{optimizer_name}.png"))
    plt.close()

    plt.figure(figsize=(15,7))
    plt.plot(y_test_real[0:500], label='Actual PM2.5', color='blue', alpha=0.7)
    plt.plot(y_pred_real[0:500], label='Predicted PM2.5', color='red', linestyle='--')
    plt.title(f'Actual vs Predicted PM2.5 - {optimizer_name}')
    plt.xlabel('Time (First 500 Hours)')
    plt.ylabel('PM2.5 Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, f"FINAL_PRED_{optimizer_name}.png"))
    plt.close()

    return metrics

# --- MAIN EXECUTION ---
def main():
    print("="*70)
    print(f"PART 1: STARTING HPO RACE ({MAX_RUNS} runs, {MAX_FEVALS} evals each)")
    print("="*70)

    try:
        (X_hpo_train, y_hpo_train), (X_hpo_val, y_hpo_val), (final_train_df, final_test_df), target_scaler = load_and_preprocess_data()
    except FileNotFoundError:
        print("Data file missing. Exiting.")
        return

    # Set the global data for the objective function
    global _FITNESS_FUNC_DATA
    _FITNESS_FUNC_DATA = {
        'X_train': X_hpo_train, 'y_train': y_hpo_train,
        'X_val': X_hpo_val, 'y_val': y_hpo_val,
        'n_features': X_hpo_train.shape[2]
    }

    # --- 💡 ADD YOUR OPTIMIZERS HERE ---
    # The key must be the optimizer's name (for plots/tables).
    # The value must be the function name (e.g., `run_my_optimizer_name`).
    optimizers = {
        "Random Search": run_random_search,
        "Hybrid AOA": run_hybrid_aoa,
        # "My_AOA_v1": run_my_AOA_v1,
        # "My_AOA_v2": run_my_AOA_v2,
    }
    
    # --- From here, the script is fully automatic ---

    hpo_run_results = defaultdict(list)
    hpo_convergence_curves = defaultdict(list)
    hpo_best_params = {} # Stores the best param vector from ALL runs

    # --- PART 1: HPO RACE ---
    for run in range(MAX_RUNS):
        print(f"\n--- HPO Race Run {run+1}/{MAX_RUNS} ---")
        for name, func in optimizers.items():
            print(f" Optimizing: {name} ...")
            # Use a new seed for each optimizer run
            np.random.seed(int(time.time() * 1000) % 2**32) 
            
            best_loss, best_params_vec, history, _ = run_optimizer(func, run_num=run)
            
            hpo_run_results[name].append(best_loss)
            hpo_convergence_curves[name].append(history)
            
            # Check if this run is the best-ever for this optimizer
            if best_loss < hpo_best_params.get(name, (np.inf, None))[0]:
                hpo_best_params[name] = (best_loss, best_params_vec)

    print("\n" + "="*70)
    print("PART 1: HPO RACE RESULTS")
    print("="*70)

    # --- HPO Summary Table ---
    hpo_summary = []
    for name, losses in hpo_run_results.items():
        hpo_summary.append({
            "Optimizer": name,
            "Best": np.min(losses),
            "Worst": np.max(losses),
            "Mean": np.mean(losses),
            "StdDev": np.std(losses)
        })
    df_hpo_summary = pd.DataFrame(hpo_summary).sort_values(by="Mean")
    print("--- HPO Performance Summary (Lower is better) ---")
    print(df_hpo_summary.to_string(index=False))

    # --- Statistical Test ---
    if len(optimizers) > 1:
        print("\n--- Statistical Significance Test (Mann-Whitney U) ---")
        control_name = df_hpo_summary.iloc[0]["Optimizer"]
        control_results = hpo_run_results[control_name]
        stats_summary = []
        for name, results in hpo_run_results.items():
            if name == control_name:
                continue
            try:
                stat, p_value = mannwhitneyu(control_results, results, alternative='less')
            except ValueError:
                p_value = 1.0
            stats_summary.append({
                "Competitor": name,
                f"p-value ({control_name} < Competitor)": f"{p_value:.4e}",
                "Significant (p < 0.05)": "Yes" if p_value < 0.05 else "No"
            })
        df_stats = pd.DataFrame(stats_summary).sort_values(by=f"p-value ({control_name} < Competitor)")
        print(f"Control Algorithm: '{control_name}'")
        print(df_stats.to_string(index=False))

    # --- Plot Average Convergence Curves ---
    plt.figure(figsize=(14,9))
    colors = plt.cm.tab20(np.linspace(0, 1, len(optimizers)))
    for i, (name, curves) in enumerate(hpo_convergence_curves.items()):
        # Ensure all curves are MAX_FEVALS long
        padded_curves = []
        for c in curves:
            if not c: c = [np.inf]
            if len(c) < MAX_FEVALS:
                c.extend([c[-1]] * (MAX_FEVALS - len(c)))
            padded_curves.append(c[:MAX_FEVALS])
        
        avg_curve = np.mean(padded_curves, axis=0)
        std_curve = np.std(padded_curves, axis=0)
        plt.plot(avg_curve, label=name, color=colors[i], lw=2)
        plt.fill_between(range(MAX_FEVALS), avg_curve-std_curve, avg_curve+std_curve, color=colors[i], alpha=0.1)
    
    plt.title(f'HPO Convergence ({MAX_RUNS} runs, {MAX_FEVALS} evals)')
    plt.xlabel('Function Evaluations')
    plt.ylabel('Best Validation Loss (MSE)')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small')
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'HPO_RACE_Convergence.png'))
    plt.close()

    # --- PART 2: FINAL VALIDATION ---
    print("\n" + "="*70)
    print("PART 2: FINAL VALIDATION")
    print("="*70)

    final_metrics_list = []
    for name, (best_loss, best_params_vec) in hpo_best_params.items():
        if best_params_vec is None or best_loss == np.inf:
            print(f"Skipping {name} due to invalid parameters or infinite loss.")
            continue
        metrics = run_final_validation(name, best_params_vec, final_train_df, final_test_df, target_scaler)
        final_metrics_list.append(metrics)

    # --- Final Metrics Table ---
    df_final_metrics = pd.DataFrame(final_metrics_list).sort_values(by='RMSE')
    cols = ["Optimizer", "RMSE", "MAE", "MSE", "R2", "MAPE", "Train Time (s)", "Total Params", "Best HPO Params"]
    df_final_metrics = df_final_metrics.reindex(columns=[c for c in cols if c in df_final_metrics.columns])

    print("\n--- Final Model Performance Summary (Sorted by RMSE) ---")
    print(df_final_metrics.to_string(index=False))

    print(f"\nBenchmark Complete. All plots and tables saved to '{PLOTS_DIR}' directory.")

if __name__ == "__main__":
    main()

