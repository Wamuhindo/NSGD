import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

wanted_service = 'Exponential'  # Specify the service you want to plot
# Load the CSV
df = pd.read_csv("resultsMixedAll.csv")  # replace with your actual filename

df = df[df['Service'].isin([wanted_service])]  # Filter out unwanted methods
# Compute mean and std dev
grouped = df.groupby(['arrival', 'Service', 'Method']).agg(
    cost_mean=('cost', 'mean'),
    cost_std=('cost', 'std')
).reset_index()

# Assign color and hatch styles
palette = sns.color_palette("Set2", n_colors=df['Method'].nunique())
services = df['Method'].unique()
service_colors = dict(zip(services, palette))

exp_styles = {
    "Exponential": "",
    "Deterministic": "//"
}

# Plotting
fig, ax = plt.subplots(figsize=(8, 7))

# Width and offsets for grouped bars
bar_width = 0.2
arrival_levels = sorted(df['arrival'].unique())
x_locs = np.arange(len(arrival_levels))
offset = 0

# Track handles for legend
legend_handles = {}

for i, (service, service_df) in enumerate(grouped.groupby("Service")):
    for j, (method, sub_df) in enumerate(service_df.sort_values(by='Method', ascending=False).groupby("Method", sort=False)):
        means = []
        stds = []
        xs = []

        for idx, arrival in enumerate(arrival_levels):
            row = sub_df[sub_df['arrival'] == arrival]
            if not row.empty:
                means.append(row['cost_mean'].values[0])
                stds.append(row['cost_std'].values[0])
                xs.append(x_locs[idx] + offset)
        
        bar = ax.bar(xs, means, yerr=stds, width=bar_width,
                     label=f'{method}' if f'{method}' not in legend_handles else "",
                     color=service_colors[method],
                     #hatch=exp_styles[method],
                     edgecolor='black',
                     capsize=5)
        
        # Add value labels on top of each bar
        for rect, mean, std in zip(bar, means, stds):
            top = mean + std
            ax.text(rect.get_x() + rect.get_width() / 2, top + 0.01 * top,
                    f'{top:.1f}', ha='center', va='bottom', fontsize=13)
        
        legend_handles[f'{method}'] = bar

        offset += bar_width + 0.01

# Axes and formatting
ax.set_xticks(x_locs + (offset - bar_width - 0.01) / 2)
ax.set_xticklabels([str(a) for a in arrival_levels], fontsize=14)
plt.tick_params(axis='y', labelsize=14)
ax.set_ylabel("Cost", fontsize=16)
plt.xlabel(r"$\lambda/N$",fontsize=16)
#ax.set_title("Cost vs Arrival by Service and Expiration", fontsize=14)
ax.legend(fontsize=14,title_fontsize=14, ncol=1)
plt.tight_layout()
plt.savefig(f"cost_vs_method_vs_arrival_{wanted_service}_v.png", dpi=100)
plt.show()
