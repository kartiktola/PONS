import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

# read each file and pull out the mean delivery‐prob, energy, latency
def summarize(file):
    df = pd.read_csv(file)
    return {
      'F1': df['F1'].mean(),
      'F2': df['F2'].mean(),
      'F3': df['F3'].mean()
    }

data = {
  '24h': summarize('conv_24h.csv'),
  '12h': summarize('conv_12h.csv'),
  '6h' : summarize('conv_6h.csv'),
}

# turn into DataFrame
df = pd.DataFrame(data).T
print(df)

# plot
for metric in ['F1','F2','F3']:
    plt.figure()
    plt.plot(df.index, df[metric], marker='o')
    plt.title(f'Convergence of {metric}')
    plt.xlabel('Simulation horizon')
    plt.ylabel(metric)
    plt.grid(True)
plt.savefig("delivery_probability.png", bbox_inches="tight")
plt.close()