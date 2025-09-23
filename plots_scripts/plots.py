import json
import numpy as np
import matplotlib.pyplot as plt

# Open the file and read line by line

costs = []
costs2 = []
weights = [0,1,1,5,500,1000]

costs_theta = []

directory = "results/results_ppo_final/PPO_Environment_2025-04-08_01-15-06.654411/evaluation"

directories = {
    "1":"results/results_ppo_final_1/PPO_Environment_2025-05-05_22-49-42.721083/evaluation",
    "05":"results/results_ppo_final_05/PPO_Environment_2025-05-05_18-18-47.076901/evaluation",
    "030":"results/results_ppo_final_03/PPO_Environment_2025-05-05_12-17-14.243220/evaluation",
    "03":"results/results_ppo_final/PPO_Environment_2025-04-08_01-15-06.654411/evaluation",
    "015":"results/results_ppo_final_015/PPO_Environment_2025-05-05_09-59-34.583632/evaluation"
}

arrival = "03"

file_costs = f"costs_function_min_{arrival}.txt"

directory = directories[arrival]

with open(f"{directory}/all_costs.csv", "r") as f:
    costs_theta = [float(line.strip()) for line in f]

with open(f'{directory}/all_states', 'r') as f:
    for line in f:
        # Parse the JSON object
        data = json.loads(line.strip())  # `strip()` to remove extra spaces/newlines
        
        # Extract elements
        state = data['state']
        job_rejected = data['job_rejected']

        if job_rejected == False:
            state.append(0)
        else:
            state.append(1)

        cost = np.dot(state, weights)

        costs.append(cost)

if arrival == "03":
    with open(f'{directories["030"]}/all_states', 'r') as f:
        for line in f:
            # Parse the JSON object
            data = json.loads(line.strip())  # `strip()` to remove extra spaces/newlines
            
            # Extract elements
            state = data['state']
            job_rejected = data['job_rejected']

            if job_rejected == False:
                state.append(0)
            else:
                state.append(1)

            cost = np.dot(state, weights)

            costs2.append(cost)

        # Print the extracted data (or you can process it as needed)
        #print(f"State: {state}, Job Rejected: {job_rejected}")

num = 10000

costs_t = np.array(costs[0:len(costs_theta)]) 
if arrival == "03":
    costs_t = (np.array(costs[0:len(costs_theta)]) + np.array(costs2[0:len(costs_theta)]))/2
fig, ax = plt.subplots(figsize=(8, 7))

tot, avg = np.array(costs_t).sum(),np.array(costs_t).mean()
tot_t, avg_t = np.array(costs_theta).sum(),np.array(costs_theta).mean()
print("PPO",tot, avg )
print("theta",tot_t, avg_t )
ax.plot(np.convolve(costs_t,np.ones(num)/num,'same'), label=f"PPO-LSTM",color="orange")# (average={round(avg,2)})
ax.plot(np.convolve(costs_theta,np.ones(num)/num,'same'), label=f"Our algorithm")# (average={round(avg_t,2)})
#plt.ylim(0,100)
plt.xlabel('Steps')
plt.ylabel('Cost')
plt.title('Cost Over Time')
plt.tight_layout()
plt.legend()
plt.savefig("zcost010_fake.png")
plt.show()
