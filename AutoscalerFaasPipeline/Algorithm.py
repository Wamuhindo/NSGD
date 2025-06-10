from AutoscalerFaasVectoriel.utils import SystemState
import numpy as np
import time

class AutoScalingAlgorithm:
    def __init__(self, N, k_delta, k_gamma,theta_init,tau, K, T=1e9, log_dir=""):
        self.N = N
        self.k_delta = k_delta
        self.k_gamma = k_gamma
        self.Tmax = T
        self.theta = theta_init
        self.theta_init = theta_init
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
        self.costs_avg_plus = np.zeros(len(theta_init))
        self.costs_avg_minus = np.zeros(len(theta_init))
        self.all_states = []
        self.all_thetas = []
        self.all_costs = []
        self.has_rejected_job=[False]*len(theta_init)
        self.last_chackpoint = time.time()
        self.log_dir = log_dir
        self.distribution_delta = {}
        self.rang = np.random.default_rng(1234)
        self.rang_delta = np.random.default_rng(1234)

    def init_state(self):
        
        self.state = [0] * self.state_elements_count
        self.state[SystemState.COLD.value] = self.N
        self.state[SystemState.IDLE_ON.value] = [0]*len(self.theta_init)
        self.state[SystemState.INITIALIZING.value] = [0]*len(self.theta_init)
        self.state[SystemState.BUSY.value] = [0]*len(self.theta_init)
        self.state[SystemState.INIT_RESERVED.value] = [0]*len(self.theta_init)
        
    def init_weights(self):
        
        self.weights = [0] * self.state_elements_count
        self.weights[SystemState.COLD.value] = 0
        self.weights[SystemState.IDLE_ON.value] = [1.0]*len(self.theta_init)
        self.weights[SystemState.BUSY.value] = [1.0]*len(self.theta_init)
        self.weights[SystemState.INITIALIZING.value] =[5.0]*len(self.theta_init)
        self.weights[SystemState.INIT_RESERVED.value] = [500]*len(self.theta_init)
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
    
    
    def simulate_step2(self, state, simulator):

        self.t += 1
        current_checkpoint = time.time()
        time_diff = current_checkpoint - self.last_chackpoint
        self.last_chackpoint = current_checkpoint
        cost = self.compute_cost(state)
        simulator.job_rejected = [False]*len(self.theta_init)

        
        #save all the costs and states
        self.all_states.append(state)
        self.all_costs.append(cost)
        
        gamma_n = self.k_gamma / (self.n ** 1.0)
        #delta_n = self.k_delta / (self.n ** (2.0 / 3.0))
        simplex = np.random.dirichlet(np.ones(len(self.theta)))
        delta_n = simplex * (self.k_delta / (self.n ** (2.0 / 3.0)))
        tau_n = int(self.tau * (1 + np.log10(self.n)))

        # with open(f"{self.log_dir}/all_costs_realtime.txt", "a") as file:
        #         file.write(f"{cost};{state};{simplex}\n")
        
        
        # update theta to theta_step = theta + delta_n
        if self.k < self.K * tau_n:
            if self.k == 0 and self.n == 1:
                self.thetas.append(self.theta)
                self.costs.append([0]*len(self.theta))
            np.random.seed(self.n + self.K)
            #print("theta", self.theta, delta_n)

            theta_plus_delta = np.array(self.theta) + np.array(delta_n)
            floor_vals = np.floor(theta_plus_delta)
            ceil_vals = floor_vals + 1

            # Randomly pick floor or ceil for each element
            rand_choices = np.random.rand(len(theta_plus_delta))  # uniform [0,1)
            theta_step = np.where(rand_choices < 0.5, floor_vals, ceil_vals)

            # Make sure sum does not exceed self.N (optional depending on your logic)
            # Cap individual values at self.N if needed
            theta_step = np.minimum(theta_step, self.N)
            #theta_step = np.minimum(np.random.choice([np.floor(np.array(self.theta) + np.array(delta_n)), np.floor(np.array(self.theta) + np.array(delta_n)) + 1],size=len(self.theta)), self.N)
            self.theta_step = theta_step
            self.costs_avg_plus += cost
            self.k += 1
            
        # update theta to theta_step = theta - delta_n
        elif self.k < 2 * self.K * tau_n:
            np.random.seed(self.n + 2 * self.K)

            theta_minus_delta = np.array(self.theta) - np.array(delta_n)
            floor_vals = np.floor(theta_minus_delta)
            ceil_vals = floor_vals + 1

            # Randomly choose floor or ceil for each element
            rand_choices = np.random.rand(len(theta_minus_delta))
            theta_step = np.where(rand_choices < 0.5, floor_vals, ceil_vals)

            # Ensure minimum value is 1
            theta_step = np.maximum(theta_step, 1)

            #theta_step = np.maximum(np.random.choice([np.floor(np.array(self.theta) - np.array(delta_n)), np.floor(np.array(self.theta) - np.array(delta_n)) + 1], size=len(self.theta)), 1)
            self.theta_step = theta_step
            self.costs_avg_minus += cost
            self.k += 1

        # update theta with the gradient descent
        else:
            self.costs_avg_plus /= (self.K * tau_n)
            self.costs_avg_minus /= (self.K * tau_n)
            gradient = (self.costs_avg_plus - self.costs_avg_minus) / (2.0 * np.array(delta_n))
            self.theta = np.minimum(np.maximum(np.array(self.theta) - gamma_n * gradient, 1.0), self.N)
            self.theta_step = self.theta
            self.thetas.append(self.theta)
            self.costs.append((self.costs_avg_plus + self.costs_avg_minus) / 2.0)
            state_to_save = self.state.copy()
            state_to_save[0] = np.array([state_to_save[0]]*len(self.theta))
            self.states.append(state_to_save)
            self.n += 1
            self.k = 0
            total_req, served_req = simulator.get_request_stats_between(simulator.last_t, simulator.t)
            served = np.array(simulator.total_finished) - np.array(simulator.last_total_finished)
            simulator.last_total_finished = simulator.total_finished
            simulator.last_t = simulator.t
            cost = (self.costs_avg_plus + self.costs_avg_minus) / 2.0
            average_resource_usage = simulator.get_average_resource_usage()
            
            #print(f"COST_P = {self.costs_avg_plus}, COST_M = {self.costs_avg_minus}, tau = {tau_n}")
            with open(f"{self.log_dir}/theta_costs_v.txt", "a") as file:
                file.write(f"{self.theta};{self.costs_avg_plus};{self.costs_avg_minus};{cost};{total_req};{served_req};{served};{average_resource_usage};{self.state}\n")
            self.costs_avg_plus = np.zeros(len(self.theta_init))
            self.costs_avg_minus = np.zeros(len(self.theta_init))
        time_after_algo = time.time()
        algo_time = time_after_algo - current_checkpoint    
        #with open("time_steps.txt", "a") as file:
        #    file.write(f"{time_diff};{algo_time}\n")'''
            
    def init_simulator_state(self,simulator,seed):
        running_function_count = [1]*len(self.theta)
        idle_function_count = [0]*len(self.theta)
        init_free_function_count = [1]*len(self.theta)
        init_reserved_function_count = [0]*len(self.theta)

        simulator._init_processes_and_transition(simulator.functions,seed)
        simulator.initialiaze_system(simulator.t, running_function_count, idle_function_count, init_free_function_count, init_reserved_function_count)

    def simulate_step(self, state, simulator):
        self.t += 1
        current_checkpoint = time.time()
        time_diff = current_checkpoint - self.last_chackpoint
        self.last_chackpoint = current_checkpoint

        cost = self.compute_cost(state)
        simulator.job_rejected = [False] * len(self.theta_init)

        # Log states and costs
        self.all_states.append(state)
        self.all_costs.append(cost)

        gamma_n = self.k_gamma / self.n

        # if f'{self.n}' not in self.distribution_delta.keys():
        #     self.distribution_delta[f'{self.n}'] = np.random.dirichlet(np.ones(len(self.theta)))
        
        tau_n = int(self.tau * (1 + np.log10(self.n)))


        if self.k < self.K * tau_n:

            if self.k == 0 and self.n == 1:
                self.thetas.append(self.theta)
                self.costs.append([0] * len(self.theta))
                #np.random.seed(n+K)
                self.rang = np.random.default_rng(self.n+self.K)
                self.rang_delta = np.random.default_rng(self.n+self.K)

                self.init_simulator_state(simulator,self.n+self.K)



            distribution = self.rang_delta.dirichlet(np.ones(len(self.theta)))
            delta_n =  distribution * (self.k_delta / self.n ** (2.0 / 3.0))
            with open(f"{self.log_dir}/delta.txt", "a") as file:
                file.write(f"plus,{distribution}\n")
            

            #np.random.seed(self.n + self.K)
            theta_step = np.where(
                self.rang.rand(len(self.theta)) < 0.5,
                np.floor(self.theta + delta_n),
                np.floor(self.theta + delta_n) + 1
            )
            self.theta_step = np.minimum(theta_step, self.N)
            self.costs_avg_plus += cost
            self.k += 1

            if self == self.K * tau_n:
                self.rang = np.random.default_rng(100*self.n+2*self.K)
                self.rang_delta = np.random.default_rng(100*self.n+2*self.K)
                self.init_simulator_state(simulator,100*self.n+2*self.K)

        elif self.k < 2 * self.K * tau_n:

            #np.random.seed(self.n + 2 * self.K)
            distribution = self.rang_delta.dirichlet(np.ones(len(self.theta)))
            delta_n =  distribution * (self.k_delta / self.n ** (2.0 / 3.0))
            with open(f"{self.log_dir}/delta.txt", "a") as file:
                file.write(f"minus,{distribution}\n")
            theta_step = np.where(
                self.rang.rand(len(self.theta)) < 0.5,
                np.floor(self.theta - delta_n),
                np.floor(self.theta - delta_n) + 1
            )
            self.theta_step = np.maximum(theta_step, 1)
            self.costs_avg_minus += cost
            self.k += 1

            if self.k == (2 * self.K * tau_n):
                print(f"{self.k}")
                self.costs_avg_plus /= (self.K * tau_n)
                self.costs_avg_minus /= (self.K * tau_n)

                gradient = (self.costs_avg_plus - self.costs_avg_minus) / (2.0 * delta_n)
                self.theta = np.clip(self.theta - gamma_n * gradient, 1.0, self.N)
                self.theta_step = self.theta
                self.thetas.append(self.theta)
                #self.costs.append((self.costs_avg_plus + self.costs_avg_minus) / 2.0)

                # Prepare and store state
                state_to_save = state.copy()
                state_to_save[0] = np.full(len(self.theta), state[0])
                self.states.append(state_to_save)

                # Update stats
                self.n += 1
                self.k = 0
                self.rang = np.random.default_rng(self.n+self.K)

                self.init_simulator_state(simulator,self.n+self.K)

                total_req, served_req = simulator.get_request_stats_between(simulator.last_t, simulator.t)
                served = np.array(simulator.total_finished) - np.array(simulator.last_total_finished)
                simulator.last_total_finished = simulator.total_finished
                simulator.last_t = simulator.t
                average_resource_usage = simulator.get_average_resource_usage()
                cost_avg = (self.costs_avg_plus + self.costs_avg_minus) / 2.0

                with open(f"{self.log_dir}/theta_costs_v.txt", "a") as file:
                    file.write(f"{self.theta};{self.costs_avg_plus};{self.costs_avg_minus};{cost_avg};{total_req};{served_req};{served};{average_resource_usage};{self.state}\n")

                self.costs_avg_plus = np.zeros(len(self.theta_init))
                self.costs_avg_minus = np.zeros(len(self.theta_init))

        # Log algorithm time
        algo_time = time.time() - current_checkpoint
        # Optionally write to time file
        # with open("time_steps.txt", "a") as file:
        #     file.write(f"{time_diff};{algo_time}\n")
   
        
       
    def compute_cost_scalar(self, state=None):
        if state is None:
            raise Exception("State is not provided")
        self.state = state.copy()
        return np.dot(self.state, self.weights) + self.w_rej * self.has_rejected_job
    

    def compute_cost(self, state=None):
        if state is None:
            raise Exception("State is not provided")

        self.state = state.copy()

        # Extract weights (wcold is always 0, so we ignore it)
        _, widle, wbusy, winitializing, winit_reserved = self.weights

        # Extract state components
        cold, idle, busy, initializing, init_reserved = self.state

        #print(cold, idle, busy, initializing, init_reserved, type(idle),isinstance(idle, list))

        
        # Compute weighted sum for each component (handling varying sizes)
        cost_idle = np.multiply(idle, widle) if isinstance(idle, list) or isinstance(idle, np.ndarray) else 0
        cost_busy = np.multiply(busy, wbusy) if isinstance(busy, list) or isinstance(idle, np.ndarray) else 0
        cost_initializing = np.multiply(initializing, winitializing) if isinstance(initializing, list) or isinstance(initializing, np.ndarray) else 0
        cost_init_reserved = np.multiply(init_reserved, winit_reserved) if isinstance(init_reserved, list) or isinstance(init_reserved, np.ndarray) else 0

        # Compute total cost for each vector element
        total_cost = np.array(cost_idle) + np.array(cost_busy) + np.array(cost_initializing) + np.array(cost_init_reserved)
        
        
        # Add rejection penalty element-wise
        total_cost += self.w_rej * np.array(self.has_rejected_job, dtype=int)

        # print("TOTAL COST",total_cost)
        # import sys
        # if hasattr(self,"numo"):
        #     if self.numo ==2:
        #         sys.exit()
        #     else:
        #         self.numo +=1
        # else:
        #     self.numo = 1

        return total_cost

