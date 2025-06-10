import pandas as pd
import matplotlib.pyplot as plt
import datetime
from matplotlib.ticker import MultipleLocator

# Load data
arrivals = ["03", "015"]

for arrival in arrivals:
    file_path1= f"costs_function_{arrival}.txt"
    file_path2 = f"costs_function_min1_{arrival}.txt"
    file_path3 = f"costs_function_min2_{arrival}.txt"
    file_path4 = f"costs_function_min3_{arrival}.txt"
    df1 = pd.read_csv(file_path1, header=None,sep=";",names=["theta", "avg_plus", "avg_minus", "avg"]) #, names=["time", "mode", "theta", "avg_plus", "avg_minus"])
    df2 = pd.read_csv(file_path2, header=None,sep=";",names=["theta", "avg_plus", "avg_minus", "avg"]) #, names=["time", "mode", "theta", "avg_plus", "avg_minus"])
    df3 = pd.read_csv(file_path3, header=None, sep=";",names=["theta", "avg_plus", "avg_minus", "avg"])
    df4 = pd.read_csv(file_path4, header=None, sep=";",names=["theta", "avg_plus", "avg_minus", "avg"])

    # Convert timestamps to readable datetime
    #df_par2["time"] = pd.to_datetime(df_par2["time"], unit="s")
    #df_par10["time"] = pd.to_datetime(df_par10["time"], unit="s")
    #df_par2["time"] = df_par2["time"].apply(lambda x: datetime.datetime.fromtimestamp(x))
    #df_par10["time"] = df_par10["time"].apply(lambda x: datetime.datetime.fromtimestamp(x))

    # Sort data by time
    #df_par2 #.sort_values(by="time")
    #df_par10 #.sort_values(by="time")
    # df.to_csv(f'{file_path_par2.split("/")[0]}/theta_costs_v.csv', index=False)
    # df10.to_csv(f'{file_path_par10.split("/")[0]}/theta_costs_v.csv', index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    x = df1["theta"]

    # Plot Theta
    ax.plot( df1["theta"], df1["avg"], label=r"$idle_min = 0$", color="blue", marker="x")
    ax.plot( df2["theta"], df2["avg"], label=r"$idle_min = 1$", color="black",linestyle="dashed", marker="x")
    ax.plot( df3["theta"], df3["avg"], label=r"$idle_min = 2$", color="magenta",linestyle="solid", marker="x")
    ax.plot( df4["theta"], df4["avg"], label=r"$idle_min = 3$", color="red",linestyle="dashed", marker="x")
    #plt.plot( x,df2["theta"], label=r"Sequential $\theta_0 = 2$", color="black",linestyle="dashed", marker="x")

    # Plot avg_plus and avg_minus
    #plt.plot( df["avg_plus"], label="Avg Plus", color="green", linestyle="dashed")
    #plt.plot(df["avg_minus"], label="Avg Minus", color="red", linestyle="dotted")

    ax.spines["left"].set_position(("data", 0)) 
    #ax.spines["right"].set_color("none")  # Hide right spine
    #ax.spines["top"].set_color("none")    # Hide top spine
    ax.yaxis.tick_left()  # Ensure y-axis ticks are on the left
    ax.xaxis.set_ticks_position("bottom") 
    ax.set_xlim([min(x), max(x)+1])  # Set x limits to the data range
    # Labels and Legend
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"Cost function")
    arrival = arrival.replace('0','0.')
    plt.title(fr"$\lambda = {arrival}$, $\mu = 1$, $\beta = 0.1$, $\gamma = 0.01$")
    plt.legend()
    #plt.xticks(rotation=45)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(5)) 
    # Show the plot
    plt.tight_layout()
    plt.grid()

    plt.show()

    plt.savefig(f'zzplots/theta_cosst_{arrival}.png', dpi=300, bbox_inches='tight') 