# scripts/05_find_pareto_front.py

import numpy as np
import pandas as pd
import joblib
import os
from tqdm import tqdm

print("--- Starting Surrogate-Based Multi-Objective Optimization ---")
OUTPUT_DIR_NAME = 'outputsFinal' 

# --- 1. Load the Scaler and Final "Champion" Models ---
MODEL_DIR = os.path.join('outputs', 'saved_models')
OUTPUT_DIR = os.path.join('..', OUTPUT_DIR_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)


try:
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    # Load the hybrid set of champion models we selected
    model_f1 = joblib.load(os.path.join(MODEL_DIR, 'gp_model_F1.joblib'))
    # The best model for F2 was the polynomial one
    poly_f2_path = os.path.join(MODEL_DIR, 'poly_model_F2.joblib')
    if os.path.exists(poly_f2_path):
        model_f2 = joblib.load(poly_f2_path)
    else: # Fallback in case you only have GP models saved
        model_f2 = joblib.load(os.path.join(MODEL_DIR, 'gp_model_F2.joblib'))
    
    model_f3 = joblib.load(os.path.join(MODEL_DIR, 'gp_model_F3.joblib'))
    # The best model for F4 was the polynomial one
    poly_f4_path = os.path.join(MODEL_DIR, 'poly_model_F4.joblib')
    if os.path.exists(poly_f4_path):
        model_f4 = joblib.load(poly_f4_path)
    else:
        model_f4 = joblib.load(os.path.join(MODEL_DIR, 'gp_model_F4.joblib'))

    print("✅ All models and scaler loaded successfully.")

except FileNotFoundError as e:
    print(f" ERROR: A model file was not found. Make sure you have run the training scripts. Details: {e}")
    exit()


# --- 2. Wrap Models into a Single Prediction Function ---
def predict_objectives(p3, a, b):
    """
    Takes arrays of p3, a, and b, and returns predictions for F1, F2, F3, and F4.
    """
    # Create the input array and scale it
    X_input = np.column_stack([p3, a, b])
    X_scaled = scaler.transform(X_input)
    
    # Predict each objective using its champion model
    pred_f1 = model_f1.predict(X_scaled)
    pred_f2 = model_f2.predict(X_scaled)
    pred_f3 = model_f3.predict(X_scaled)
    pred_f4 = model_f4.predict(X_scaled)
    
    return pred_f1, pred_f2, pred_f3, pred_f4

print("✅ Prediction function is ready.")


# --- 3. Run Optimization via Large-Scale Random Sampling ---
N_SAMPLES = 100000  # Generate 100,000 random configurations
print(f"\n🚀 Generating and evaluating {N_SAMPLES} random parameter configurations...")

# Generate random samples for [p3, a, b] within their valid bounds [0.01, 1.0]
# We use a helper function to avoid p1 > p2 constraint violations later
def generate_valid_samples(n_samples):
    p3 = np.random.uniform(0.01, 1.0, n_samples)
    a = np.random.uniform(0.01, 1.0, n_samples)
    b = np.random.uniform(a, 1.0, n_samples) # Ensure b is always >= a
    return p3, a, b

p3_samples, a_samples, b_samples = generate_valid_samples(N_SAMPLES)

# Evaluate all samples using our fast surrogate models
f1_preds, f2_preds, f3_preds, f4_preds = predict_objectives(p3_samples, a_samples, b_samples)

# Combine inputs and predictions into a single DataFrame
results_df = pd.DataFrame({
    'p3': p3_samples, 'a': a_samples, 'b': b_samples,
    'F1_Delivery': f1_preds, 'F2_Energy': f2_preds,
    'F3_Latency': f3_preds, 'F4_Fairness': f4_preds
})
print(f"✅ Evaluated {len(results_df)} configurations.")


# --- 4. Identify the Pareto Front ---
print("\n🔍 Identifying the Pareto-optimal front from the results...")

# Our goals are to MAXIMIZE F1 and MINIMIZE F2, F3, F4.
# A point is "dominated" if another point is better or equal in all objectives,
# and strictly better in at least one. The Pareto front is the set of non-dominated points.

def find_pareto_front(df):
    pareto_indices = []
    # Create a NumPy array for faster comparison
    # We negate F1 because we want to maximize it (minimizing -F1 is equivalent)
    values = df[['F1_Delivery', 'F2_Energy', 'F3_Latency', 'F4_Fairness']].to_numpy()
    values[:, 0] = -values[:, 0] # Negate F1 for minimization
    
    for i in tqdm(range(len(values)), desc="Finding Pareto Front"):
        # Assume the current point is on the Pareto front
        is_dominated = False
        for j in range(len(values)):
            if i == j:
                continue
            # Check if point j dominates point i
            if np.all(values[j] <= values[i]) and np.any(values[j] < values[i]):
                is_dominated = True
                break # Another point dominates this one, no need to check further
        
        if not is_dominated:
            pareto_indices.append(i)
            
    return df.iloc[pareto_indices]

pareto_df = find_pareto_front(results_df)

# --- 5. Save the Results ---
output_path = os.path.join(OUTPUT_DIR, 'pareto_front_solutions.csv')
pareto_df.to_csv(output_path, index=False)

print(f"\n✅ Pareto front with {len(pareto_df)} optimal solutions saved to '{output_path}'")
print("\n--- Script Finished ---")