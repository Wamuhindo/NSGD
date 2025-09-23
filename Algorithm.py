from utils import SystemState
import numpy as np
import time

class AutoScalingAlgorithm:
    def __init__(self, N, k_delta, k_gamma,theta_init,tau, K, T=1e9):
        self.N = N
        self.k_delta = k_delta
        self.k_gamma = k_gamma
        self.Tmax = T
        self.theta = theta_init
        self.theta_step = theta_init
        self.state_elements_count = 5
        self.state = []
        self.weights = []
        self.tau = tau
        self.init_state()
        self.init_weights()
        self.n = 1
        self.t = 0
        self.K = K
        self.costs = []
        self.thetas = []
        self.states = []
        self.all_costs = []
        self.k = 0
        self.costs_avg_plus = 0
        self.costs_avg_minus = 0
        self.all_states = []
        self.all_thetas = []
        self.all_costs = []
        self.has_rejected_job=False
        self.last_chackpoint = time.time()

    def init_state(self):
        
        self.state = [0] * self.state_elements_count
        self.state[SystemState.COLD.value] = self.N
        self.state[SystemState.IDLE_ON.value] = 0
        self.state[SystemState.INITIALIZING.value] = 0
        self.state[SystemState.BUSY.value] = 0
        self.state[SystemState.INIT_RESERVED.value] = 0 
        
    def init_weights(self):
        
        self.weights = [0] * self.state_elements_count
        self.weights[SystemState.COLD.value] = 0
        self.weights[SystemState.IDLE_ON.value] = 1
        self.weights[SystemState.INITIALIZING.value] = 5.0
        self.weights[SystemState.BUSY.value] = 1.0
        self.weights[SystemState.INIT_RESERVED.value] = 500
        self.w_rej = 1e3 
        
    def set_has_rejected_job(self,has_rejected_job):
        self.has_rejected_job = has_rejected_job

    def set_weights(self, w_cold, w_idle_on, w_init_free, w_busy, w_reserved, w_rej):
        self.weights[SystemState.COLD.value] = w_cold
        self.weights[SystemState.IDLE_ON.value] = w_idle_on
        self.weights[SystemState.INITIALIZING.value] = w_init_free
        self.weights[SystemState.BUSY.value] = w_busy
        self.weights[SystemState.INIT_RESERVED.value] = w_reserved
        self.w_rej = w_rej
    
    def set_state(self, cold, idle_on, init_free, busy, init_reserved):
        self.state[SystemState.COLD.value] = cold
        self.state[SystemState.IDLE_ON.value] = idle_on
        self.state[SystemState.INITIALIZING.value] = init_free
        self.state[SystemState.BUSY.value] = busy
        self.state[SystemState.INIT_RESERVED.value] = init_reserved
    
    def get_state(self):
        return self.state
    
    def get_theta_step(self):
        #return 2
        return self.theta_step
    
    def running_condition(self):
        
        return self.t < self.Tmax
    
    
    def simulate_step(self, state, simulator):

        self.t += 1
        current_checkpoint = time.time()
        time_diff = current_checkpoint - self.last_chackpoint
        self.last_chackpoint = current_checkpoint
        cost = self.compute_cost(state)
        simulator.job_rejected = False
        
        #save all the costs and states
        self.all_states.append(state)
        self.all_costs.append(cost)
        
        gamma_n = self.k_gamma / (self.n ** 1.0)
        delta_n = self.k_delta / (self.n ** (2.0 / 3.0))
        tau_n = int(self.tau * (1 + np.log10(self.n)))
        
        # update theta to theta_step = theta + delta_n
        if self.k < self.K * tau_n:
            if self.k == 0 and self.n == 1:
                self.thetas.append(self.theta)
                self.costs.append(0)
            np.random.seed(self.n + self.K)
            theta_step = min(np.random.choice([np.floor(self.theta + delta_n), np.floor(self.theta + delta_n) + 1]), self.N)
            self.theta_step = theta_step
            self.costs_avg_plus += cost
            self.k += 1
        # update theta to theta_step = theta - delta_n
        elif self.k < 2 * self.K * tau_n:
            np.random.seed(self.n + 2 * self.K)
            theta_step = max(np.random.choice([np.floor(self.theta - delta_n), np.floor(self.theta - delta_n) + 1]), 1)
            self.theta_step = theta_step
            self.costs_avg_minus += cost
            self.k += 1
        # update theta with the gradient descent
        else:
            self.costs_avg_plus /= (self.K * tau_n)
            self.costs_avg_minus /= (self.K * tau_n)
            self.theta = min(max(self.theta - gamma_n * (self.costs_avg_plus - self.costs_avg_minus) / (2.0 * delta_n), 1.0), self.N)
            self.theta_step = self.theta
            self.thetas.append(self.theta)
            self.costs.append((self.costs_avg_plus + self.costs_avg_minus) / 2.0)
            self.states.append(self.state.copy())
            self.n += 1
            self.k = 0
            total_req, served_req = simulator.get_request_stats_between(simulator.last_t, simulator.t)
            served = simulator.total_finished - simulator.last_total_finished
            simulator.last_total_finished = simulator.total_finished
            simulator.last_t = simulator.t
            cost = (self.costs_avg_plus + self.costs_avg_minus) / 2.0
            average_resource_usage = simulator.get_average_resource_usage()
            with open("theta_costs.txt", "a") as file:
                file.write(f"{self.theta};{self.costs_avg_plus};{self.costs_avg_minus};{cost};{total_req};{served_req};{served};{average_resource_usage}\n")
            self.costs_avg_plus = 0
            self.costs_avg_minus = 0
        time_after_algo = time.time()
        algo_time = time_after_algo - current_checkpoint    
        #with open("time_steps.txt", "a") as file:
        #    file.write(f"{time_diff};{algo_time}\n")'''
            
            
        
       
    def compute_cost(self, state=None):
        if state is None:
            raise Exception("State is not provided")
        self.state = state.copy()
        return np.dot(self.state, self.weights) + self.w_rej * self.has_rejected_job

