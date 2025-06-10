import pandas as pd
import matplotlib.pyplot as plt
import datetime
from matplotlib.ticker import MultipleLocator, FixedLocator
import numpy as np

# Load data
idle_ = ""
directory = f"zzzcosts_both{idle_}"
arrivals = ["015", "03", "05", "1"]
arrivals = ["5"]
idle_mins = [0]

colors = {
    "015":"black",
    "03":"blue",
    "05":"pink",
    "1":"red"
}

markers = {
    0: "x",
    1:"x",
    2:"x",
    3:"x",
    4:"x"
}

linestyles = {
    0: '-',
    1: '--',
    2: '-.',
    3: ':',
    4: (0, (3, 5, 1, 5))  # custom dash pattern
}

linestyles = {
    0: '-',
    1: '-',
    2: '-',
    3: '-',
    4: '-' # custom dash pattern
}

fsz_label = 16
fsz_value = 14
fsz_legend = 13
N=50


for arrival in arrivals:
    fig, ax = plt.subplots(figsize=(8, 7))
    ymax = 0
    ymin = 25
    for idle_min in idle_mins:
        file_path= f"zzsingle_parallel_cost_paper_thetainit5_dn/zexperiment_PAR_test_all_adam_{arrival}/costs_function_min_{arrival}.txt"
        df1 = pd.read_csv(file_path, header=None,sep=";",names=["theta", "avg_plus", "avg_minus", "avg"]) 

        # Plot
        df1["opt_array"] = df1["theta"].apply(lambda s: np.fromstring(s.strip('[]'), sep=' '))
        
        print(df1["opt_array"])
        x = df1["opt_array"]
        minx = 0
        maxx = 36

        x = x[minx:maxx]
        y = df1["avg"][minx:maxx]

        ymax = max(max(y),ymax)
        ymin = min(min(y),ymin)

        # Plot Theta
        ax.plot( x, y,   marker=markers[idle_min], linestyle=linestyles[idle_min]) 


    ax.spines["left"].set_position(("data", min(x))) 

    ax.yaxis.tick_left()  # Ensure y-axis ticks are on the left
    ax.xaxis.set_ticks_position("bottom") 
    ax.set_xlim([min(x), max(x)+0])  # Set x limits to the data range
    ax.set_ylim([50, ymax+5])  # Set x limits to the data range
    # Labels and Legend
    plt.xlabel(fr"$\theta_{{stock}}$",fontsize=fsz_label)
    plt.ylabel(r"Cost function",fontsize=fsz_label)
    arrival = arrival.replace('0','0.')
    plt.title(fr"$\lambda/N = {int(arrival)/N}$, $\mu = 1$, $\alpha = 0.1$, $\theta_{{idle}} = {idle_min}$, $\theta_{{exp}} = 5\cdot 10^{{-3}}$",fontsize=fsz_label)

    ax = plt.gca()
    #ax.xaxis.set_major_locator(MultipleLocator(5)) 

    ticks = np.arange(0, max(x) + 5, 5)
    if 1 not in ticks:
        ticks = np.sort(np.append(ticks, 1))

    ax.xaxis.set_major_locator(FixedLocator(ticks))

    y_ticks = ax.get_yticks()  # current y-ticks

    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.xaxis.set_tick_params(labelsize=fsz_value)
    ax.yaxis.set_tick_params(labelsize=fsz_value)

    # Show the plot
    plt.tight_layout()
    plt.grid()
    plt.savefig(f'zzzplots5/theta_cost_init_{arrival}{idle_}.png', dpi=100, bbox_inches='tight') 
    plt.show()

