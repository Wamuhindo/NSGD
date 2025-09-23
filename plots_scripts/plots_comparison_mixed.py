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

# PPO result directories
ppo_dirs = {
    "0.1": "results/results_ppo_final_5/PPO_Environment_2025-05-19_06-41-24.766754",
    "0.2": "results/results_ppo_final_10/PPO_Environment_2025-05-19_18-04-58.840352",
    "0.3": "results/results_ppo_final_15/PPO_Environment_2025-05-29_12-18-17.571635",
    "0.7": "results/results_ppo_final_35/PPO_Environment_2025-05-29_13-22-46.270037",
}



ppo_dirs_old = {

    "0.1": "res/res5/PPO_Environment_2025-05-12_23-04-30.238007", 
    "0.2": "res/res10/PPO_Environment_2025-05-20_07-50-58.927823",
    "0.3": "res/res15/PPO_Environment_2025-05-20_07-52-30.445730",
    "0.7": "res/res35/PPO_Environment_2025-05-29_21-46-29.405341",
}

ppo_dirs_pareto_old= {

    "0.1": "res_pareto/res_5/PPO_Environment_2025-06-08_11-08-45.595920", 
    "0.2": "res_pareto/res_10/PPO_Environment_2025-06-08_10-45-31.975945",
    "0.3": "res_pareto/res_15/PPO_Environment_2025-06-08_10-49-44.724934",
    "0.7":"res_pareto/res_35/PPO_Environment_2025-06-08_10-50-47.868157"
}


ppo_dirs_pareto = {

    "0.1": "results/results_ppo_final_pareto_5/PPO_Environment_2025-06-08_16-25-04.324402", 
    "0.2": "results/results_ppo_final_pareto_10/PPO_Environment_2025-06-08_11-19-13.181329",
    "0.3": "results/results_ppo_final_pareto_15/PPO_Environment_2025-06-08_19-53-12.410117",
    "0.7":"results/results_ppo_final_pareto_35/PPO_Environment_2025-06-08_16-42-28.597453"
}


fsz_label = 16
fsz_value = 14
fsz_legend = 13

# Collect "Our Algorithm" data
all_data = []

for arrival, directory in ppo_dirs.items():
    filepath = f"{directory}/costs100.csv"
    df = pd.read_csv(filepath, header=0, sep=",")
    for cost in df["cost"]:
        all_data.append({"arrival": arrival, "cost": cost, "Method": "LSTM-PPO"})
        all_data.append({"arrival": arrival, "cost": cost, "Service": "Exponential", "Expiration":"Exponential", "Method": "LSTM-PPO-COST"})

for arrival, directory in ppo_dirs_pareto.items():
    filepath = f"{directory}/costs100.csv"
    df = pd.read_csv(filepath, header=0, sep=",")
    for cost in df["cost"]:
        all_data.append({"arrival": arrival, "cost": cost, "Method": "LSTM-PPO"})
        all_data.append({"arrival": arrival, "cost": cost, "Service": "Pareto", "Expiration":"Exponential", "Method": "LSTM-PPO-COST"})


for arrival, directory in ppo_dirs_old.items():
    filepath = f"{directory}/costs100.csv"
    df = pd.read_csv(filepath, header=0, sep=",")
    for cost in df["cost"]:
        all_data.append({"arrival": arrival, "cost": cost, "Method": "LSTM-PPO"})
        all_data.append({"arrival": arrival, "cost": cost, "Service": "Exponential", "Expiration":"Exponential", "Method": "LSTM-PPO"})

for arrival, directory in ppo_dirs_pareto_old.items():
    filepath = f"{directory}/costs100.csv"
    df = pd.read_csv(filepath, header=0, sep=",")
    for cost in df["cost"]:
        all_data.append({"arrival": arrival, "cost": cost, "Method": "LSTM-PPO"})
        all_data.append({"arrival": arrival, "cost": cost, "Service": "Pareto", "Expiration":"Exponential", "Method": "LSTM-PPO"})



for arrival in arrivals:
    filepath = f"costs/costs_function_min_{arrivals_dict[arrival]}.txt"
    df = pd.read_csv(filepath, names=["theta", "costp", "costm", "cost"], sep=";")
    for cost in df["cost"]:
        all_data.append({"arrival": arrival, "cost": cost, "Service": "Exponential", "Expiration":"Exponential", "Method": "NSGD"})


for arrival in arrivals:
    filepath = f"costs/costs_function_min_{arrivals_dict[arrival]}_pareto.txt"
    df = pd.read_csv(filepath, names=["theta", "costp", "costm", "cost"], sep=";")
    for cost in df["cost"]:
        all_data.append({"arrival": arrival, "cost": cost, "Service": "Pareto", "Expiration":"Exponential", "Method": "NSGD"})


# Combine and plot
plot_df = pd.DataFrame(all_data)
plot_df.to_csv("resultsMixedAll.csv", index=False)

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



