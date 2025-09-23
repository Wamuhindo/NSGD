import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the CSV
df = pd.read_csv("resultsOurAlgoAll.csv")  # replace with your actual filename

# Compute mean and std dev
grouped = df.groupby(['arrival', 'Service', 'Expiration']).agg(
    cost_mean=('cost', 'mean'),
    cost_std=('cost', 'std')
).reset_index()

# Assign color and hatch styles
palette = sns.color_palette("Set2", n_colors=df['Service'].nunique())
services = df['Service'].unique()
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
    for j, (expiration, sub_df) in enumerate(service_df.groupby("Expiration")):
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
                     label=f'{service} - {expiration}' if f'{service}-{expiration}' not in legend_handles else "",
                     color=service_colors[service],
                     hatch=exp_styles[expiration],
                     edgecolor='black',
                     capsize=5)
        

        # Add value labels on top of each bar
        for rect, mean, std in zip(bar, means, stds):
            top = mean + std
            #ax.text(rect.get_x() + rect.get_width() / 2, top + 0.01 * top,
            #        f'{mean:.0f}', ha='center', va='bottom', fontsize=13)
        
        legend_handles[f'{service}-{expiration}'] = bar

        offset += bar_width + 0.01

# Axes and formatting
ax.set_xticks(x_locs + (offset - bar_width - 0.01) / 2)
ax.set_xticklabels([str(a) for a in arrival_levels], fontsize=14)
plt.tick_params(axis='y', labelsize=16)
ax.set_ylabel("Cost", fontsize=18)
plt.xlabel(r"$\lambda/N$",fontsize=16)
#ax.set_title("Cost vs Arrival by Service and Expiration", fontsize=14)
ax.legend(title="Service Time - Expiration Time",fontsize=14,title_fontsize=14, ncol=1)
plt.tight_layout()
plt.savefig("cost_vs_arrival.png", dpi=300)
plt.show()
