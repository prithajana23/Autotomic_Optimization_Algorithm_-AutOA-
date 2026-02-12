import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import warnings
import random
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import mannwhitneyu
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# --- 0. CONFIGURATION ---
MAX_RUNS = 10        # Run 10 times for a statistical average
MAX_FEVALS = 50      # Give each optimizer 50 evaluations
HPO_EPOCHS = 20      # Train for 20 epochs during the HPO race
TIME_STEPS = 24 * 3

FINAL_EPOCHS = 100   # Train the final model for 100 epochs
FINAL_PATIENCE = 10  # Stop if validation loss doesn't improve for 10 epochs

DATA_FILE = 'PRSA_data_2010.1.1-2014.12.31.csv'
PLOTS_DIR = 'cnn_benchmark_plots'
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

warnings.filterwarnings('ignore')
# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')

HPO_BOUNDS_LOW = [0.0] * 5
HPO_BOUNDS_HIGH = [1.0] * 5
HPO_DIM = 5

# --- HELPER: Global Seeding ---
def set_global_seed(seed):
    """Sets the seed for Python, NumPy, and TensorFlow to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# --- HPO Parameter Decoding ---
def decode_params(cont_params):
    """Converts a continuous vector [0,1]^5 into a discrete set of hyperparameters."""
    # 1. Number of Layers
    num_layers = 1 if cont_params[0] < 0.5 else 2
    
    # 2. Number of Filters
    if cont_params[1] < 0.33:
        num_filters = 16
    elif cont_params[1] < 0.66:
        num_filters = 32
    else:
        num_filters = 64

    # 3. Kernel Size
    if cont_params[2] < 0.33:
        kernel_size = 3
    elif cont_params[2] < 0.66:
        kernel_size = 5
    else:
        kernel_size = 7

    # 4. Batch Size
    if cont_params[3] < 0.33:
        batch_size = 16
    elif cont_params[3] < 0.66:
        batch_size = 32
    else:
        batch_size = 64

    # 5. Learning Rate (Log Scale)
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
            # Placeholder URL; typically requires local file
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"
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
_FITNESS_FUNC_DATA = {} 

def objective_function(cont_params):
    """The main fitness function for continuous optimizers. Takes a [0,1]^5 vector."""
    global _FITNESS_FUNC_DATA
    params = decode_params(cont_params)
    
    try:
        model, _ = build_model(params, TIME_STEPS, _FITNESS_FUNC_DATA['n_features'])
        history = model.fit(_FITNESS_FUNC_DATA['X_train'], _FITNESS_FUNC_DATA['y_train'],
                            validation_data=(_FITNESS_FUNC_DATA['X_val'], _FITNESS_FUNC_DATA['y_val']),
                            epochs=HPO_EPOCHS, batch_size=params['batch_size'], verbose=0)
        
        val_loss = history.history['val_loss'][-1]
        if np.isnan(val_loss) or np.isinf(val_loss): return np.inf
        return val_loss
    except Exception:
        return np.inf

# --- OPTIMIZER RUNNERS ---
def run_optimizer(optimizer_func, run_num):
    """Generic wrapper. Handles timing, errors, and history padding."""
    start_time = time.time()
    convergence_history = [] 
    
    try:
        # best_params_solution can be a vector OR a dict (for TPE)
        best_params_solution, best_loss = optimizer_func(convergence_history)
    except Exception as e:
        print(f"  ERROR in optimizer: {e}")
        import traceback
        traceback.print_exc()
        best_params_solution = None
        best_loss = np.inf

    elapsed = time.time() - start_time
    print(f"  Run {run_num+1:02d} complete in {elapsed:.2f}s. Best Loss: {best_loss:.4f}")
    
    if len(convergence_history) < MAX_FEVALS:
        pad_value = convergence_history[-1] if convergence_history else np.inf
        convergence_history += [pad_value] * (MAX_FEVALS - len(convergence_history))
        
    return best_loss, best_params_solution, convergence_history[:MAX_FEVALS], elapsed

# ==============================================================================
# --- SPECIFIC OPTIMIZERS ---
# ==============================================================================

# --- TPE (Hyperopt) Configuration ---
# We define the lists here so we can map indices back to values manually
TPE_OPTS = {
    'num_layers': [1, 2],
    'num_filters': [16, 32, 64],
    'kernel_size': [3, 5, 7],
    'batch_size': [16, 32, 64]
}

TPE_SPACE = {
    "num_layers": hp.choice("num_layers", TPE_OPTS['num_layers']),
    "num_filters": hp.choice("num_filters", TPE_OPTS['num_filters']),
    "kernel_size": hp.choice("kernel_size", TPE_OPTS['kernel_size']),
    "batch_size": hp.choice("batch_size", TPE_OPTS['batch_size']),
    "learning_rate": hp.loguniform("learning_rate", np.log(1e-4), np.log(1e-2))
}

def objective_function_tpe(params):
    """Specific objective function for Hyperopt. Receives REAL values."""
    global _FITNESS_FUNC_DATA
    try:
        # Params come in as numpy types sometimes, cast to python native
        clean_params = {
            'num_layers': int(params['num_layers']),
            'num_filters': int(params['num_filters']),
            'kernel_size': int(params['kernel_size']),
            'batch_size': int(params['batch_size']),
            'learning_rate': float(params['learning_rate'])
        }
        model, _ = build_model(clean_params, TIME_STEPS, _FITNESS_FUNC_DATA['n_features'])
        history = model.fit(_FITNESS_FUNC_DATA['X_train'], _FITNESS_FUNC_DATA['y_train'],
                            validation_data=(_FITNESS_FUNC_DATA['X_val'], _FITNESS_FUNC_DATA['y_val']),
                            epochs=HPO_EPOCHS, batch_size=clean_params['batch_size'], verbose=0)
        val_loss = history.history['val_loss'][-1]
        return {"loss": val_loss if np.isfinite(val_loss) else np.inf, "status": STATUS_OK}
    except Exception:
        return {"loss": np.inf, "status": STATUS_OK}

def run_tpe(history_list):
    trials = Trials()
    best_indices = fmin(fn=objective_function_tpe, space=TPE_SPACE, algo=tpe.suggest, 
                        max_evals=MAX_FEVALS, trials=trials, verbose=0)
    
    # ✅ FIX 1: Reconstruct the EXACT dictionary from indices
    # fmin returns indices for hp.choice, but actual value for hp.loguniform
    best_params_real = {
        'num_layers': int(TPE_OPTS['num_layers'][best_indices['num_layers']]),
        'num_filters': int(TPE_OPTS['num_filters'][best_indices['num_filters']]),
        'kernel_size': int(TPE_OPTS['kernel_size'][best_indices['kernel_size']]),
        'batch_size': int(TPE_OPTS['batch_size'][best_indices['batch_size']]),
        'learning_rate': float(best_indices['learning_rate'])
    }
    
    best_loss = trials.best_trial['result']['loss']
    
    # Populate history
    current_best = np.inf
    for loss in trials.losses():
        if loss < current_best: current_best = loss
        history_list.append(current_best)
        
    return best_params_real, best_loss  # Returning dict, not vector!

# --- Random Search ---
def run_random_search(history_list):
    best_loss = np.inf
    best_params = None
    for _ in range(MAX_FEVALS):
        cont_params = np.random.rand(HPO_DIM)
        loss = objective_function(cont_params)
        if loss < best_loss:
            best_loss = loss
            best_params = cont_params
        history_list.append(best_loss) 
    return best_params, best_loss

# ==============================================================================
# --- FINAL VALIDATION ---
# ==============================================================================

def run_final_validation(optimizer_name, best_params_solution, final_train_df, final_test_df, target_scaler):
    """Trains the best model on full data. Handles both Vector and Dict inputs."""
    print(f"\n--- Final Validation: {optimizer_name} ---")
    
    # ✅ FIX 1 (Part 2): Check if input is Dict (TPE) or Vector (Others)
    if isinstance(best_params_solution, dict):
        params = best_params_solution
        print(f"  Params (Direct): {params}")
    else:
        params = decode_params(best_params_solution)
        print(f"  Params (Decoded): {params}")
    
    # Prepare Full Data
    TARGET_COL = 'pm2.5'
    FEATURES = [col for col in final_train_df.columns if col != TARGET_COL]
    
    feature_scaler_final = StandardScaler()
    final_train_scaled_features = feature_scaler_final.fit_transform(final_train_df[FEATURES])
    final_test_scaled_features = feature_scaler_final.transform(final_test_df[FEATURES])
    
    final_train_scaled_target = target_scaler.transform(final_train_df[[TARGET_COL]])
    final_test_scaled_target = target_scaler.transform(final_test_df[[TARGET_COL]])
    
    final_train_scaled = pd.DataFrame(final_train_scaled_features, columns=FEATURES, index=final_train_df.index)
    final_train_scaled[TARGET_COL] = final_train_scaled_target
    final_test_scaled = pd.DataFrame(final_test_scaled_features, columns=FEATURES, index=final_test_df.index)
    final_test_scaled[TARGET_COL] = final_test_scaled_target
    
    X_train, y_train = create_timeseries_dataset(final_train_scaled[FEATURES], final_train_scaled[TARGET_COL], TIME_STEPS)
    X_test, y_test = create_timeseries_dataset(final_test_scaled[FEATURES], final_test_scaled[TARGET_COL], TIME_STEPS)
    
    # Train
    model, total_params = build_model(params, TIME_STEPS, len(FEATURES))
    early_stopping = EarlyStopping(monitor='val_loss', patience=FINAL_PATIENCE, restore_best_weights=True)
    
    start_time = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=FINAL_EPOCHS,
                        batch_size=params['batch_size'], verbose=1, callbacks=[early_stopping])
    train_time = time.time() - start_time
    
    # Metrics
    y_pred_scaled = model.predict(X_test)
    y_test_real = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_real = target_scaler.inverse_transform(y_pred_scaled)

    def mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        if not np.any(mask): return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    metrics = {
        "Optimizer": optimizer_name,
        "RMSE": np.sqrt(mean_squared_error(y_test_real, y_pred_real)),
        "MAE": mean_absolute_error(y_test_real, y_pred_real),
        "R2": r2_score(y_test_real, y_pred_real),
        "MAPE": mape(y_test_real, y_pred_real),
        "Train Time (s)": train_time,
        "Total Params": total_params,
        "Best HPO Params": str(params)
    }
    print(f"  Final Validation Complete. RMSE: {metrics['RMSE']:.4f}")

    # Plotting
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss Curve - {optimizer_name}')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, f"FINAL_LOSS_{optimizer_name}.png"))
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

    global _FITNESS_FUNC_DATA
    _FITNESS_FUNC_DATA = {
        'X_train': X_hpo_train, 'y_train': y_hpo_train,
        'X_val': X_hpo_val, 'y_val': y_hpo_val,
        'n_features': X_hpo_train.shape[2]
    }

    optimizers = {
        "Random Search": run_random_search,
        "TPE (Bayesian)": run_tpe, 
    }

    hpo_run_results = defaultdict(list)
    hpo_convergence_curves = defaultdict(list)
    hpo_best_params = {}

    # --- PART 1: HPO RACE ---
    for run in range(MAX_RUNS):
        print(f"\n--- HPO Race Run {run+1}/{MAX_RUNS} ---")
        for name, func in optimizers.items():
            print(f" Optimizing: {name} ...")
            
            # ✅ FIX 3: Robust Seeding (Python + Numpy + TF)
            seed = run * 100 + hash(name) % 100
            set_global_seed(seed)
            
            best_loss, best_params_sol, history, _ = run_optimizer(func, run_num=run)
            
            hpo_run_results[name].append(best_loss)
            hpo_convergence_curves[name].append(history)
            
            if best_loss < hpo_best_params.get(name, (np.inf, None))[0]:
                hpo_best_params[name] = (best_loss, best_params_sol)

    # --- RESULTS & STATS ---
    print("\n" + "="*70)
    print("PART 1: HPO RACE RESULTS")
    print("="*70)

    hpo_summary = []
    for name, losses in hpo_run_results.items():
        hpo_summary.append({
            "Optimizer": name,
            "Best": np.min(losses),
            "Mean": np.mean(losses),
            "StdDev": np.std(losses)
        })
    df_hpo_summary = pd.DataFrame(hpo_summary).sort_values(by="Mean")
    print(df_hpo_summary.to_string(index=False))

    if len(optimizers) > 1:
        print("\n--- Statistical Significance Test (Mann-Whitney U) ---")
        control_name = df_hpo_summary.iloc[0]["Optimizer"]
        control_results = hpo_run_results[control_name]
        stats_summary = []
        for name, results in hpo_run_results.items():
            if name == control_name: continue
            
            # ✅ FIX 2: Statistical Safety Check
            if len(control_results) < 5 or len(results) < 5:
                p_value = np.nan
            else:
                try:
                    stat, p_value = mannwhitneyu(control_results, results, alternative='less')
                except ValueError:
                    p_value = 1.0
            
            stats_summary.append({
                "Competitor": name,
                f"p-value ({control_name} < Competitor)": f"{p_value:.4e}",
                "Significant (p < 0.05)": "Yes" if p_value < 0.05 else "No"
            })
        print(f"Control Algorithm: '{control_name}'")
        print(pd.DataFrame(stats_summary).to_string(index=False))

    # --- PLOTTING ---
    plt.figure(figsize=(14,9))
    colors = plt.cm.tab20(np.linspace(0, 1, len(optimizers)))
    for i, (name, curves) in enumerate(hpo_convergence_curves.items()):
        padded_curves = []
        for c in curves:
            if not c: c = [np.inf]
            if len(c) < MAX_FEVALS: c.extend([c[-1]] * (MAX_FEVALS - len(c)))
            padded_curves.append(c[:MAX_FEVALS])
        
        avg_curve = np.mean(padded_curves, axis=0)
        plt.plot(avg_curve, label=name, color=colors[i], lw=2)
    
    plt.title(f'HPO Convergence ({MAX_RUNS} runs)')
    plt.xlabel('Function Evaluations')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, 'HPO_RACE_Convergence.png'))
    plt.close()

    # --- PART 2: FINAL VALIDATION ---
    print("\n" + "="*70)
    print("PART 2: FINAL VALIDATION")
    print("="*70)

    final_metrics_list = []
    for name, (best_loss, best_params_sol) in hpo_best_params.items():
        if best_params_sol is None: continue
        metrics = run_final_validation(name, best_params_sol, final_train_df, final_test_df, target_scaler)
        final_metrics_list.append(metrics)

    df_final = pd.DataFrame(final_metrics_list).sort_values(by='RMSE')
    print("\n--- Final Model Performance Summary ---")
    print(df_final.to_string(index=False))
    print(f"\nBenchmark Complete. Results in '{PLOTS_DIR}' directory.")

if __name__ == "__main__":
    main()
