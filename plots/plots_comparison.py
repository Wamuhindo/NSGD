import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

arrivals = ["0.1", "0.2", "0.3", "0.7"]

arrivals_dict = {
    "0.1":"5",
    "0.2":"10",
    "0.3":"15",
    "0.7":"35"
}


fsz_label = 16
fsz_value = 14
fsz_legend = 13

# Collect "Our Algorithm" data
all_data = []
for arrival in arrivals:
    filepath = f"costs/costs_function_min_{arrivals_dict[arrival]}.txt"
    df = pd.read_csv(filepath, names=["theta", "costp", "costm", "cost"], sep=";")
    for cost in df["cost"]:
        all_data.append({"arrival": arrival, "cost": cost, "Service": "Exponential", "Expiration":"Exponential"})

for arrival in arrivals:
    filepath = f"costs/costs_function_min_{arrivals_dict[arrival]}_dt.txt"
    df = pd.read_csv(filepath, names=["theta", "costp", "costm", "cost"], sep=";")
    for cost in df["cost"]:
        all_data.append({"arrival": arrival, "cost": cost, "Service": "Exponential", "Expiration":"Deterministic"})

for arrival in arrivals:
    filepath = f"costs/costs_function_min_{arrivals_dict[arrival]}_pareto.txt"
    df = pd.read_csv(filepath, names=["theta", "costp", "costm", "cost"], sep=";")
    for cost in df["cost"]:
        all_data.append({"arrival": arrival, "cost": cost, "Service": "Pareto", "Expiration":"Exponential"})


for arrival in arrivals:
    filepath = f"costs/costs_function_min_{arrivals_dict[arrival]}_dt_pareto.txt"
    df = pd.read_csv(filepath, names=["theta", "costp", "costm", "cost"], sep=";")
    for cost in df["cost"]:
        all_data.append({"arrival": arrival, "cost": cost, "Service": "Pareto", "Expiration":"Deterministic"})


# Combine and plot
plot_df = pd.DataFrame(all_data)
plot_df.to_csv("zzzOurAlgoAll.csv", index=False)

plt.figure(figsize=(8, 7))

plt.legend(fontsize=fsz_legend)
plt.xlabel(r"$\lambda/N$",fontsize=fsz_label)
plt.ylabel("Cost",fontsize=fsz_label)
plt.tick_params(axis='x', labelsize=fsz_value)
plt.tick_params(axis='y', labelsize=fsz_value)
plt.grid(True, linestyle='-', alpha=0.5)
plt.tight_layout()
#plt.savefig("box_plot_all_ppo_dt_old.png", dpi=100)
plt.show()



