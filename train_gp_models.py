# scripts/04_train_gp_models.py

import matplotlib
matplotlib.use('Agg') 

import pandas as pd
import numpy as np
import os
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error
from joblib import Parallel, delayed
from sklearn.compose import TransformedTargetRegressor

# --- Configuration ---
DATA_FILE_PATH = os.path.join('sweep_sparse_merged.csv')
OUTPUT_DIR = os.path.join('outputs')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'saved_models')

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("--- Starting Final Gaussian Process Surrogate Modeling ---")

# --- 1. Load and Prepare Data ---
df = pd.read_csv(DATA_FILE_PATH)
X = df[['p3', 'a', 'b']]
y = df[['F1', 'F2', 'F3', 'F4']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler object saved to '{scaler_path}'")

# --- 2. Define a function for processing a single objective ---
def process_objective(objective_name, X_scaled, y_series):
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    kernel = ConstantKernel(1.0, (1e-3, 1e6)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e2))
    cv_splitter = KFold(n_splits=10, shuffle=True, random_state=42)

    # Base GP model
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42, alpha=1e-10)

    # **THE FIX**: Wrap the GP in a regressor that automatically scales the target variable (y).
    # This standardizes the y-values, making the GP fitting process much more stable.
    model_with_y_scaling = TransformedTargetRegressor(regressor=gp, transformer=StandardScaler())

    y_pred_cv = cross_val_predict(model_with_y_scaling, X_scaled, y_series, cv=cv_splitter, n_jobs=1)
    
    r2_cv = r2_score(y_series, y_pred_cv)
    rmse_cv = np.sqrt(mean_squared_error(y_series, y_pred_cv))
    
    model_with_y_scaling.fit(X_scaled, y_series)
    
    model_path = os.path.join(MODEL_DIR, f'gp_model_{objective_name}.joblib')
    joblib.dump(model_with_y_scaling, model_path)
    
    y_pred_final = model_with_y_scaling.predict(X_scaled)
    residuals = y_series - y_pred_final
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_series, y_pred_final, alpha=0.6)
    plt.plot([y_series.min(), y_series.max()], [y_series.min(), y_series.max()], '--r', lw=2)
    plt.xlabel("True Values"); plt.ylabel("Predicted Values")
    plt.title(f"GP Model - {objective_name}: Predicted vs. True (R²={r2_cv:.3f})")
    plt.savefig(os.path.join(OUTPUT_DIR, f"GP_{objective_name}_predicted_vs_true.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred_final, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel("Predicted Values"); plt.ylabel("Residuals (True - Predicted)")
    plt.title(f"GP Model - {objective_name}: Residual Plot")
    plt.savefig(os.path.join(OUTPUT_DIR, f"GP_{objective_name}_residuals.png"), dpi=300)
    plt.close()

    return {'objective': objective_name, 'r2': r2_cv, 'rmse': rmse_cv}

# --- 3. Run the processing in Parallel ---
print(f"\n🚀 Starting parallel model training for {len(y.columns)} objectives...")
results = Parallel(n_jobs=-1)(
    delayed(process_objective)(col, X_scaled, y[col]) for col in tqdm(y.columns, desc="Processing Objectives")
)

# --- 4. Save all GP Cross-Validation Scores to a CSV ---
gp_cv_df = pd.DataFrame(results)
gp_scores_path = os.path.join(OUTPUT_DIR, 'gp_cv_scores.csv')
gp_cv_df.to_csv(gp_scores_path, index=False)
print(f"\n✅ All GP cross-validation results saved to '{gp_scores_path}'")

print("\n--- Final Script Finished ---")