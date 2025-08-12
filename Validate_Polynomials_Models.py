# scripts/03_validate_polynomial_models.py

import matplotlib
# IMPORTANT: Set a non-interactive backend BEFORE importing pyplot
# This prevents the script from trying to open a GUI window and causing the error.
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error

# --- Configuration ---
DATA_FILE_PATH = os.path.join('sweep_sparse_merged.csv')
OUTPUT_DIR = os.path.join('outputs_ValidatePolyModels')

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("--- Starting Polynomial Model Validation ---")

# --- 1. Load and Prepare Data ---
try:
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"✅ Successfully loaded dataset. Rows: {len(df)}")
except FileNotFoundError:
    print(f"❌ ERROR: Data file not found at '{DATA_FILE_PATH}'.")
    exit()

X = df[['p3', 'a', 'b']]
y = df[['F1', 'F2', 'F3', 'F4']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. Systematically Train, Validate, and Plot ---

degrees_to_test = [2, 3, 4, 5]
cv_splitter = KFold(n_splits=10, shuffle=True, random_state=42)
all_cv_results = []

for objective_name in y.columns:
    print(f"\n{'='*20} Processing Objective: {objective_name} {'='*20}")
    
    objective_results = []
    for degree in degrees_to_test:
        model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
        
        y_pred_cv = cross_val_predict(model, X_scaled, y[objective_name], cv=cv_splitter)
        
        r2_cv = r2_score(y[objective_name], y_pred_cv)
        rmse_cv = np.sqrt(mean_squared_error(y[objective_name], y_pred_cv))
        
        objective_results.append({'degree': degree, 'r2': r2_cv, 'rmse': rmse_cv})
        print(f"  Degree {degree}: Cross-Validated R² = {r2_cv:.4f} | RMSE = {rmse_cv:.4f}")

    best_model_info = max(objective_results, key=lambda x: x['r2'])
    best_degree = best_model_info['degree']
    all_cv_results.extend([{'objective': objective_name, **res} for res in objective_results])
    
    print(f"  🏆 Best Degree for {objective_name} is {best_degree} (R² = {best_model_info['r2']:.4f})")
    
    final_model = make_pipeline(PolynomialFeatures(degree=best_degree, include_bias=False), LinearRegression())
    final_model.fit(X_scaled, y[objective_name])
    y_pred_final = final_model.predict(X_scaled)
    residuals = y[objective_name] - y_pred_final

    # --- Plot 1: Predicted vs. True Values ---
    plt.figure(figsize=(8, 8))
    plt.scatter(y[objective_name], y_pred_final, alpha=0.6)
    plt.plot([y[objective_name].min(), y[objective_name].max()], [y[objective_name].min(), y[objective_name].max()], '--r', lw=2)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{objective_name}: Predicted vs. True (Degree {best_degree}, R²={best_model_info['r2']:.3f})")
    plot_path_pred = os.path.join(OUTPUT_DIR, f"{objective_name}_predicted_vs_true.png")
    plt.savefig(plot_path_pred, dpi=300)
    plt.close()
    print(f"  ✅ Saved Predicted vs. True plot to '{plot_path_pred}'")

    # --- Plot 2: Residual vs. Predicted Values ---
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred_final, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (True - Predicted)")
    plt.title(f"{objective_name}: Residual Plot (Degree {best_degree})")
    plot_path_res = os.path.join(OUTPUT_DIR, f"{objective_name}_residuals.png")
    plt.savefig(plot_path_res, dpi=300)
    plt.close()
    print(f"  ✅ Saved Residual plot to '{plot_path_res}'")
    
    # --- (Optional) 4. Save Model Coefficients ---
    try:
        coeffs = final_model.named_steps['linearregression'].coef_
        coeff_path = os.path.join(OUTPUT_DIR, f'model_coeffs_{objective_name}.txt')
        np.savetxt(coeff_path, coeffs, fmt='%.6f')
        print(f"  ✅ Saved model coefficients to '{coeff_path}'")
    except Exception as e:
        print(f"  ⚠️ Could not save coefficients: {e}")

# --- 5. Save all Cross-Validation Scores to a CSV ---
cv_df = pd.DataFrame(all_cv_results)
cv_scores_path = os.path.join(OUTPUT_DIR, 'polynomial_cv_scores.csv')
cv_df.to_csv(cv_scores_path, index=False)
print(f"\n✅ All cross-validation results saved to '{cv_scores_path}'")

print("\n--- Script Finished ---")