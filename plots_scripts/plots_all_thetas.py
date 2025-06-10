import pandas as pd
import matplotlib.pyplot as plt
import datetime
from matplotlib.ticker import MaxNLocator, MultipleLocator


# Load data



date = "08_06_2025" 
power_tau = 4

arrival = "5"

parent_folder = f"simulation_parallel_{arrival}"


# colors = ['blue','blue','black', 'black',"red","red"]
# linestyle = ['solid','dashed','solid', 'dashed',"solid","dashed"]
# labels = ['', '', '', '','','']
# markers = ['x','x','x','x','x','x']

# scenarios = [""]

colors={
    "4":"blue",
    "2":"black",
    "10":"red",
    "0":"orange",
}

colors = ["#1f77b4","black","#2ca02c","blue","red","#ff7f0e"]
linestyles={
    "2":"-",
    "4":":",
    "10":":",
    "0":"-"
}

markers ={
    "1":"o",
    "2":"v",
    "0.1":"^",
    "0.5":"x",
    "0.2":"<",
    "0.01":"x",
    "5":"x",
    "3":"^",
    "10":"<",
}

metrics=["theta","theta_min", "gamma"]
#metrics=[ "gamma"]
dict_metrics = {
    "theta":r"$\theta_{{stock}}$",
    "theta_min":r"$\theta_{{idle}}$",
    "gamma":r"$\theta_{{exp}}\cdot 10^{3}$",
}

metrics_names={
    "theta":"theta_init",
    "theta_min":"theta_idle",
    "gamma":"theta_exp",
               }

K = 1

fsz_label = 16
fsz_title = 16
fsz_legend = 14
fsz_ticks = 14

# 0.15 0.15 , 0.3 0.3 _sss , 0.5 0.5 _s, 1

for metric in metrics:

    fig, ax = plt.subplots(figsize=(8, 7))

    opt_inits = [[1,1,5],[1,5,5],[3,4,10],[10,2,1]] #[2,5,5],
    #opt_inits = [[1,1,5],[10,2,1]]
    #opt_inits = [[2,4,5]]
    i=0
    
    for opt_init in opt_inits:

        root = f"{parent_folder}/zexperiment_PAR_{'_'.join(str(x) for x in opt_init)}_{power_tau}_{date}/"
            
        dfs = []
        for k in range(K):
            file_path = f"{parent_folder}{suf}/zexperiment_PAR_{'_'.join(str(x) for x in opt_init)}_{power_tau}_{date}/theta_costs_plus_{k}.csv" #_{theta_min}
            file_path_minus = f"{parent_folder}{suf}/zexperiment_PAR_{'_'.join(str(x) for x in opt_init)}_{power_tau}_{date}/theta_costs_minus_{k}.csv"

            print(file_path,file_path_minus)
            try:
                df_plus = pd.read_csv(file_path, header=0)
            except pd.errors.EmptyDataError:
                df_plus = pd.DataFrame(columns=["time", "mode", "theta", "theta_min", "gamma", "avg_plus", "avg_minus"])
            try:
                df_minus = pd.read_csv(file_path_minus, header=0)
            except pd.errors.EmptyDataError:
                df_minus = pd.DataFrame(columns=["time", "mode", "theta", "theta_min", "gamma", "avg_plus", "avg_minus"])
            dfs.append(df_plus)
            dfs.append(df_minus)

        df = pd.concat(dfs, ignore_index=True)
        
        new_row = pd.DataFrame({
            'time': [0],
            'mode': ['plus'],
            'theta': [opt_init[0]],
            'theta_min': [opt_init[1]],
            'gamma': [opt_init[2]],
            'gamma_bf': [opt_init[2]],
            'avg_plus': [0],
            'avg_minus': [0]
            })
        
        df = pd.concat([new_row,df], ignore_index=True)

        df = df.sort_values(by="time")
        
        if "gamma_bf" not in df.columns:
            df["gamma_bf"] = df["gamma"]
        
        df.to_csv(f"{root}all_metrics.csv",index=False)

        # Plot
       
        minx, maxx = 0, 83
        rnd=2 
        if metric == "theta":
            rnd=3
        elif metric == "gamma":
            rnd=3
        else:
            rnd=3 
        x = list(range(len(df["time"])))[minx:maxx]

        y = df[f"{metric}_bf" if "gamma" in metric and "gamma_bf" in df.columns else metric].round(rnd)[minx:maxx]

        # Plot Theta
        ax.plot( x,y, label=fr"$\theta_{{stock}}^{{0}} = {opt_init[0]}$, $\theta_{{idle}}^{{0}} = {opt_init[1]}$, $\theta_{{exp}}^0 = {opt_init[2]}\cdot10^{{-3}}$",   marker="x") #marker=markers[f"{opt_init[2]}"] color=colors[f"{opt_init[0]}"], # linestyle=linestyles[f"{opt_init[1]}"],
        i+=1



    ax.spines["left"].set_position(("data", 0)) 

    ax.yaxis.tick_left()  # Ensure y-axis ticks are on the left
    ax.xaxis.set_ticks_position("bottom") 
    ax.set_xlim([min(x), max(x)+1])  # Set x limits to the data range
    if metric == "theta":
        ax.set_ylim([0,11])
    elif metric == "gamma":
        ax.set_ylim([0,11])
    else:
        ax.set_ylim([0,6])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_tick_params(labelsize=fsz_ticks)
    ax.xaxis.set_tick_params(labelsize=fsz_ticks)
    # Labels and Legend
    plt.xlabel("n",fontsize=fsz_label)
    plt.ylabel(dict_metrics[metric],fontsize=fsz_label)
    plt.title(fr"$\tau = 10^{power_tau}$, $\lambda = {arrival}$, $\mu = 1$, $\alpha = 0.1$",fontsize=fsz_title)
    plt.legend(fontsize=fsz_legend)

    # Show the plot
    plt.tight_layout()
    plt.grid()

    plt.savefig(f'zzzplots5/plots_{metrics_names[metric]}_{arrival.replace(".","")}_{power_tau}_pareto.png', dpi=100, bbox_inches='tight') 

    plt.show()