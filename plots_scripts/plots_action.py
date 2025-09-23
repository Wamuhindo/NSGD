import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Open the file and read line by line

actions = []
replicas = []
weights = [0,1,1,5,500,1000]

costs_theta = []

with open('results/results_ppo/PPO_Environment_2025-04-01_13-55-35.162548/simulation_log.txt', 'r') as f:
    for line in f:
        # Parse the JSON object
        data = json.loads(line.strip())  # `strip()` to remove extra spaces/newlines
        
        # Extract elements
        action = int(data['action'])
        replica = data['replicas']

        actions.append(action)
        replicas.append(replica)

        # Print the extracted data (or you can process it as needed)
        #print(f"State: {state}, Job Rejected: {job_rejected}")

fig, ax = plt.subplots(figsize=(12, 8))
num = 1
#plt.plot(np.convolve(replicas,np.ones(num)/num,'same'), label="PPO")
#plt.plot(np.convolve(replicas,np.ones(num)/num,'same'))
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.plot(actions)

plt.xlabel('Steps')
plt.ylabel('Action')
plt.title('Actions Over Time')
plt.tight_layout()
plt.legend()
plt.grid()
plt.show()
plt.savefig("test_actions.png")

fig, ax = plt.subplots(figsize=(12, 8))
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.plot(replicas)
plt.xlabel('Steps')
plt.ylabel('Replicas')
plt.title('Replicas Over Time')
plt.tight_layout()
plt.legend()
plt.grid()
plt.show()
plt.savefig("test_replica.png")