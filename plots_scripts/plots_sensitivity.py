import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Your data

# Create DataFrame
df = pd.read_csv("sensitivity_data.csv")

# Melt to long format for plotting
df_long = df.melt(id_vars=["method", "arrival"], var_name="Requests", value_name="Value")

# Convert Requests column to numeric for proper plotting
df_long["Requests"] = df_long["Requests"].astype(int)

# Plot
g = sns.FacetGrid(df_long, col="arrival", hue="method", col_wrap=2, height=4, sharey=False)
g.map(sns.lineplot, "Requests", "Value", marker="o")
#g.add_legend(loc="upper center", fontsize=12)
g.fig.set_size_inches(12, 10)
for ax, title in zip(g.axes.flat, g.col_names):
    if title == 0.7:
        ax.legend(loc="upper left", title="method")


for ax in g.axes.flatten():
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

g.set_axis_labels("Weight(Init-reserved)", "Cost", fontsize=14)
g.set_titles("Arrival = {col_name}", size=14)
plt.tight_layout()
plt.savefig("sensitivity.png",dpi=100)
plt.show()