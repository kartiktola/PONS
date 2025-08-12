import matplotlib
# CRITICAL FIX: Set a non-interactive backend BEFORE importing pyplot
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

print("--- Starting Pareto Front Visualization ---")

# --- 1. Configuration and Loading Data ---
DATA_FILE_PATH = os.path.join('outputsFinal', 'pareto_front_solutions.csv')
OUTPUT_DIR = os.path.join('outputsFinal')

try:
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"✅ Successfully loaded Pareto front data with {len(df)} optimal solutions.")
except FileNotFoundError:
    print(f"❌ ERROR: Data file not found at '{DATA_FILE_PATH}'. Make sure you have run the optimization script first.")
    exit()

# --- 2. Create a Pair Plot of the Objectives ---
print("📊 Generating Pair Plot of objectives...")

# We only want to plot the objective columns
objective_cols = ['F1_Delivery', 'F2_Energy', 'F3_Latency', 'F4_Fairness']

pair_plot = sns.pairplot(df[objective_cols], kind='scatter', diag_kind='kde', plot_kws={'alpha': 0.6})
pair_plot.fig.suptitle('2D Projections of the Pareto Front Trade-offs', y=1.03)

# Save the plot
pairplot_path = os.path.join(OUTPUT_DIR, 'pareto_front_pairplot.png')
plt.savefig(pairplot_path, dpi=300)
print(f"✅ Saved Pair Plot to '{pairplot_path}'")
plt.close()


# --- 3. Create a Parallel Coordinates Plot ---
print("📊 Generating Parallel Coordinates Plot...")

# This plot helps visualize the parameters that lead to certain objective values.
# It's interactive, so we save it as an HTML file.

# We rename columns for a cleaner plot legend
plot_df = df.rename(columns={
    'p3': 'Param: p3', 'a': 'Param: a', 'b': 'Param: b',
    'F1_Delivery': 'Obj: Delivery (F1)', 'F2_Energy': 'Obj: Energy (F2)',
    'F3_Latency': 'Obj: Latency (F3)', 'F4_Fairness': 'Obj: Fairness (F4)'
})

# We want to highlight solutions with the highest delivery probability
color_by = 'Obj: Delivery (F1)'

fig = px.parallel_coordinates(
    plot_df,
    color=color_by,
    dimensions=['Param: p3', 'Param: a', 'Param: b', 'Obj: Delivery (F1)', 'Obj: Energy (F2)', 'Obj: Latency (F3)', 'Obj: Fairness (F4)'],
    color_continuous_scale=px.colors.sequential.Viridis,
    title="Parallel Coordinates Plot of Pareto-Optimal Solutions"
)

# Save the interactive plot to an HTML file
parallel_plot_path = os.path.join(OUTPUT_DIR, 'pareto_front_parallel_plot.html')
fig.write_html(parallel_plot_path)
print(f"✅ Saved interactive Parallel Coordinates Plot to '{parallel_plot_path}'")

print("\n--- Script Finished ---")