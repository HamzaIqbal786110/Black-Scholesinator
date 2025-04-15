import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, '../outputs')

FILENAME_PATTERN = re.compile(r'black_scholes_(\w+)_p(\d+)_t(\d+)\.out')
results = []

# Parse each output file
for file in os.listdir(OUTPUT_DIR):
    match = FILENAME_PATTERN.match(file)
    if match:
        impl, p, t = match.groups()
        p, t = int(p), int(t)
        with open(os.path.join(OUTPUT_DIR, file)) as f:
            text = f.read()

        times = re.findall(r'TIME TAKEN\s*=\s*([\d.]+)\s*ms', text)
        times = [float(t) for t in times]

        if times:
            results.append({
                'Implementation': impl,
                'PSteps': p,
                'TSteps': t,
                'MeanTime': sum(times) / len(times),
                'StdTime': pd.Series(times).std(),
                'RunCount': len(times)
            })

df = pd.DataFrame(results)
df.to_csv(os.path.join(BASE_DIR, 'parsed_results.csv'), index=False)

sns.set(style="whitegrid")

# 2D Plot with fixed variable (TSteps or PSteps)
def plot_fixed_variable(df, fixed_col, variable_col, values, title_prefix, fname_prefix):
    for val in values:
        subset = df[df[fixed_col] == val]
        if subset.empty:
            continue

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=subset, x=variable_col, y='MeanTime', hue='Implementation', marker='o')
        plt.title(f"{title_prefix} (Fixed {fixed_col} = {val})")
        plt.xlabel(variable_col)
        plt.ylabel('Mean Execution Time (ms)')
        plt.legend(title='Implementation')
        plt.tight_layout()
        fname = f"{fname_prefix}_fixed_{fixed_col}_{val}.png"
        plt.savefig(os.path.join(BASE_DIR, fname))
        plt.close()

# Choose representative fixed values: low, mid, high
def get_representative_values(series):
    unique_vals = sorted(series.unique())
    if len(unique_vals) < 3:
        return unique_vals
    return [unique_vals[0], unique_vals[len(unique_vals)//2], unique_vals[-1]]

# Plot: Varying TSteps with fixed PSteps
pstep_vals = get_representative_values(df['PSteps'])
plot_fixed_variable(df, 'PSteps', 'TSteps', pstep_vals, 'Mean Time vs TSteps', 'mean_time_vs_tsteps')

# Plot: Varying PSteps with fixed TSteps
tstep_vals = get_representative_values(df['TSteps'])
plot_fixed_variable(df, 'TSteps', 'PSteps', tstep_vals, 'Mean Time vs PSteps', 'mean_time_vs_psteps')

# 3D Surface Plot per implementation
def plot_3d_surface(df, impl):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    subset = df[df['Implementation'] == impl]
    X = subset['PSteps']
    Y = subset['TSteps']
    Z = subset['MeanTime']
    ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(f'3D Surface Plot - {impl}')
    ax.set_xlabel('PSteps')
    ax.set_ylabel('TSteps')
    ax.set_zlabel('Mean Time (ms)')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f'3d_surface_{impl}.png'))
    plt.close()

for impl in df['Implementation'].unique():
    plot_3d_surface(df, impl)

# Combined 3D Scatter Plot
def plot_combined_3d(df):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    for impl in df['Implementation'].unique():
        subset = df[df['Implementation'] == impl]
        ax.scatter(subset['PSteps'], subset['TSteps'], subset['MeanTime'], label=impl, s=60)
    ax.set_title('3D Scatter Plot: Mean Time across PSteps & TSteps (All Implementations)')
    ax.set_xlabel('PSteps')
    ax.set_ylabel('TSteps')
    ax.set_zlabel('Mean Time (ms)')
    ax.legend(title='Implementation')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '3d_combined_all_implementations.png'))
    plt.close()

plot_combined_3d(df)
