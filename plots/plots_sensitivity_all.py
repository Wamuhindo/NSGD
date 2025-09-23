import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Your data
# Create DataFrame
df = pd.read_csv("sensitivity_data.csv")
# Melt to long format
df_long = df.melt(id_vars=["method", "arrival","ord"], var_name="Requests", value_name="Value")
df_long["Requests"] = df_long["Requests"].astype(int)

# Unique arrival values
arrivals = sorted(df_long["arrival"].unique())

# Create a figure for each arrival
for arrival in arrivals:
    plt.figure(figsize=(8, 7))
    subset = df_long[df_long["arrival"] == arrival].sort_values(by="ord",ascending=False)
    sns.lineplot(data=subset, x="Requests", y="Value", hue="method", marker="o",linewidth=2.5,markersize=8)
    #plt.title(fr"$\lambda/N = {arrival}$", fontsize=16)
    plt.xlabel(fr"Weight $w_4$", fontsize=20)
    plt.ylabel("Cost", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend( loc="upper left", fontsize=18, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(f"sensitivity_{arrival}.png", dpi=100)
    plt.show()
