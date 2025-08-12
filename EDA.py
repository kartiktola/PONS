# scripts/01_eda.py

import matplotlib
# IMPORTANT: Set a non-interactive backend BEFORE importing pyplot
# This prevents the script from trying to open a GUI window.
matplotlib.use('Agg') 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Configuration ---
DATA_FILE_PATH = os.path.join('sweep_sparse_merged.csv')
OUTPUT_DIR = os.path.join('..', 'outputs')

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("--- Starting Exploratory Data Analysis ---")

# --- 1. Load the Dataset ---
try:
    df = pd.read_csv(DATA_FILE_PATH)
    print(f" Successfully loaded dataset from '{DATA_FILE_PATH}'. Rows: {len(df)}, Cols: {len(df.columns)}")
except FileNotFoundError:
    print(f" ERROR: Data file not found at '{DATA_FILE_PATH}'.")
    exit()

# --- 2. Dataset Overview ---
print("\n=== Dataset Info ===")
df.info()

# Define the objective columns that are already in your CSV
obj_cols = ['F1', 'F2', 'F3', 'F4']
print("\n=== Descriptive Statistics (Objectives) ===")
print(df[obj_cols].describe())

# --- 3. Correlation Heatmap ---
print("\n Generating correlation heatmap...")
# Define all columns relevant for the heatmap
heatmap_columns = ['p3', 'a', 'b', 'p1', 'p2'] + obj_cols

plt.figure(figsize=(12, 10))
correlation_matrix = df[heatmap_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f')
plt.title('Correlation Matrix of Input Parameters and Objectives', fontsize=16)
plt.tight_layout()

heatmap_path = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
plt.savefig(heatmap_path, dpi=300)
print(f"Saved correlation heatmap to '{heatmap_path}'")
plt.close()

# --- 4. Pair Plots ---
print(" Generating pair plots...")
input_vars = ['p3', 'a', 'b']

pair_plot = sns.pairplot(
    df,
    x_vars=input_vars,
    y_vars=obj_cols,
    kind='reg',
    plot_kws={'line_kws': {'color': 'red', 'lw': 2}, 'scatter_kws': {'alpha': 0.6}}
)
pair_plot.fig.suptitle('Input Parameters vs. Output Objectives', y=1.02, fontsize=16)

pairplot_path = os.path.join(OUTPUT_DIR, 'input_output_pairplot.png')
plt.savefig(pairplot_path, dpi=300)
print(f" Saved pair plots to '{pairplot_path}'")
plt.close()

print("\n--- EDA Script Finished ---")
print(" Check the 'outputs/' folder for your plots.")