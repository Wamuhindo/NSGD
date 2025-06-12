from AutoscalerFaas.utils import SystemState
import numpy as np
import time

class AutoScalingAlgorithm:
    def __init__(self, N, k_delta, k_gamma,theta_init,tau, K, T=1e9, log_dir=""):
        self.N = N
        self.k_delta = k_delta
        self.k_gamma = k_gamma
        self.Tmax = T
        self.opt = theta_init
        self.opt_init = theta_init
        self.opt_step = theta_init
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
        self.log_dir = log_dir
        self.m = 0.0
        self.v = 0.0
        self.beta = np.array([0.9,0.9,0.9])
        self.grad_avg_sq = np.zeros(len(theta_init))
        self.beta1 = np.array([0.9 ,0.85,0.9])
        self.beta2 = np.array([0.999,0.999,0.999])
        self.epsilon = 1e-8

    def set_params(self, **kwargs):
        
        self.K_exp = kwargs.get('K_exp',1000)
        self.gamma_min = kwargs.get('gamma_min',1)
        self.exp_lr = kwargs.get('exp_lr',[1,1,1])
        self.prtb = kwargs.get('prtb',[
            [-0.5, 0.5],   # for element 0
            [-0.5, 0.5],   # for element 1
            [-1, 1]    # for element 2
        ])
        self.grad_norm = kwargs.get('grad_norm', 1e6)
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
        self.weights[SystemState.BUSY.value] = 1.0
        self.weights[SystemState.INITIALIZING.value] = 5.0
        self.weights[SystemState.INIT_RESERVED.value] = 50
        self.w_rej = 100
        
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
        return self.opt_step
    
    def running_condition(self):
        
        return self.t < self.Tmax


    def simulate_step(self, state, simulator):

        self.t += 1
        current_checkpoint = time.time()

        self.last_chackpoint = current_checkpoint
        cost = self.compute_cost(state)
        simulator.job_rejected = False
        
        
    
        digit_round = 6
        gamma_round = 6
        theta_round = 6

        # with open(f"{self.log_dir}/all_costs_realtime.txt", "a") as file:
        #         file.write(f"{cost}\n")#;{state}
        
        #save all the costs and states
        self.all_states.append(state)
        self.all_costs.append(cost)
        
        gamma_n = self.k_gamma / (self.n ** 1.0)
        delta_n = self.k_delta / (self.n ** (2.0 / 3.0))
        tau_n = int(self.tau * (1 + np.log10(self.n)))
        
        # update theta to theta_step = theta + delta_n
        if self.k < self.K * tau_n:
            if self.k == 0 and self.n == 1:
                self.thetas.append(self.opt)
                self.costs.append(np.array([0]))
                #np.random.seed(self.n + self.K)
                
                all_choices = np.array(self.prtb)
                
                self.rang_perturb = np.random.default_rng(simulator.seed+0*10)

                # Use random indices (0 or 1) for each row
                random_indices =  self.rang_perturb.integers(0, 2, size=all_choices.shape[0])
                # Select independently
                self.perturbation = all_choices[np.arange(all_choices.shape[0]), random_indices]

                self.perturbations =  self.perturbation * delta_n 
                simulator.rang_delta_plus = np.random.default_rng(simulator.seed)
                #simulator.rang_delta_minus = np.random.default_rng(simulator.seed)
                simulator.rang_delta_min_plus = np.random.default_rng(simulator.seed)
                #simulator.rang_delta_min_minus = np.random.default_rng(simulator.seed)
                
            #np.random.seed(self.n + self.K)
            opt_delta = np.array(self.opt) + self.perturbations #sim.autoscaler.theta + delta_n
                    
            theta_delta = opt_delta[0]
            theta_min_delta = opt_delta[1]
            p_theta = theta_delta - np.floor(theta_delta)
            theta_step = round(min(simulator.rang_delta_plus.choice([np.floor(theta_delta), np.floor(theta_delta) + 1], p=[1-p_theta,  p_theta]), self.N),theta_round)
            p_theta_min = theta_min_delta - np.floor(theta_min_delta)
            theta_min_step = round(min(simulator.rang_delta_min_plus.choice([np.floor(theta_min_delta), np.floor(theta_min_delta) + 1], p=[1-p_theta_min,p_theta_min]), self.N),theta_round)
            gamma_exp_step = opt_delta[2]
            gamma_exp_step = round(gamma_exp_step,gamma_round)
            if gamma_exp_step < 0:
                gamma_exp_step = max(opt_delta[2],self.gamma_min) #max(opt_delta[2], 0.01) np.log(0.01)
                #penalty_gamma +=1
            #print(f"OOPPPT_array {sim_type}",opt_delta)
            opt_step = np.array([theta_step,theta_min_step ,gamma_exp_step])
            #print("OOPPPT_array {sim_type}",opt_step)
            simulator.expiration_process.rate = round(gamma_exp_step/self.K_exp, digit_round)
            self.opt_step = opt_step
            self.costs_avg_plus += cost
            self.k += 1

            if self.k == self.K * tau_n:
                running_function_count = 0
                idle_function_count = 0
                init_free_function_count = 0
                init_reserved_function_count = 0
                
                #simulator.initialiaze_system(simulator.t, running_function_count, idle_function_count, init_free_function_count, init_reserved_function_count )
                # np.random.seed(self.n+2*self.K)
                #simulator.rang_delta_plus = np.random.default_rng(simulator.seed)
                simulator.rang_delta_minus = np.random.default_rng(simulator.seed)
                simulator.rang_delta_min_minus = np.random.default_rng(simulator.seed)
                
            
        # update theta to theta_step = theta - delta_n
        elif self.k < 2 * self.K * tau_n:
            #np.random.seed(self.n + 2 * self.K)
            opt_delta = np.array(self.opt) - self.perturbations #sim.autoscaler.theta - delta_n
            #opt_delta[2] = self.opt[2]
            theta_delta = opt_delta[0]
            theta_min_delta = opt_delta[1]
            p_theta = theta_delta - np.floor(theta_delta)
            p_theta_min = theta_min_delta - np.floor(theta_min_delta)
            theta_step = round(max(simulator.rang_delta_minus.choice([np.floor(theta_delta), np.floor(theta_delta) + 1],p=[1-p_theta,  p_theta]), 1),theta_round)
            theta_min_step = round(max(simulator.rang_delta_min_minus.choice([np.floor(theta_min_delta), np.floor(theta_min_delta) + 1],p=[1-p_theta_min,p_theta_min]), 1),theta_round )
            gamma_exp_step = opt_delta[2]
            gamma_exp_step = round(gamma_exp_step,gamma_round)
            if gamma_exp_step < 0:
                gamma_exp_step = max(opt_delta[2],self.gamma_min)#max(opt_delta[2], 0.01)np.log(0.01)
                #penalty_gamma +=1
            opt_step = np.array([theta_step,theta_min_step ,gamma_exp_step])
            #print("OOPPPT_array {sim_type}",opt_step)
            simulator.expiration_process.rate = round(gamma_exp_step/self.K_exp, digit_round)
                    #sim.expiration_process.rate = 0.01
            self.opt_step = opt_step
            self.costs_avg_minus += cost
            self.k += 1

        # update theta with the gradient descent
            if self.k == 2 * self.K * tau_n:
                self.costs_avg_plus /= (self.K * tau_n)
                self.costs_avg_minus /= (self.K * tau_n)
                
                self.costs_avg_plus = np.full(len(self.opt_init),self.costs_avg_plus)
                self.costs_avg_minus = np.full(len(self.opt_init),self.costs_avg_minus)
                grad = (self.costs_avg_plus - self.costs_avg_minus) / (2.0 * self.perturbations) 
                
                
                if simulator.optimization == "adam":
                    self.m = self.beta1 * self.m + (1 - self.beta1) * grad
                    self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

                    # Bias correction (important early on)
                    m_hat = self.m / (1 - self.beta1 ** self.n)
                    v_hat = self.v / (1 - self.beta2 ** self.n)
                    
                    opt = self.opt - gamma_n * m_hat / (v_hat ** 0.5 + self.epsilon)
                    
                elif simulator.optimization == "RMSProp":
                    # RMSProp optimization
                    # Update the running average of squared gradients
                    self.grad_avg_sq = self.beta * self.grad_avg_sq + (1 - self.beta) * grad**2
                    opt = self.opt - gamma_n * grad / (np.sqrt(self.grad_avg_sq) + self.epsilon)
                
                else:
                    opt = self.opt - gamma_n * grad

                theta_opt = round(min(max(opt[0], 1), self.N),theta_round)
                theta_min_opt = round(min(max(opt[1], 1), self.N),theta_round)
                gamma_exp_opt = round(max(opt[2],self.gamma_min),gamma_round) #max(opt[2], 0.01)
                self.opt = np.array([theta_opt,theta_min_opt,gamma_exp_opt])
                
                simulator.expiration_process.rate = round(gamma_exp_opt/self.K_exp, digit_round)
                
                #self.opt = min(max(self.opt - gamma_n * grad, 1.0), self.N)
                self.opt_step = self.opt
                self.thetas.append(self.opt)
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
                
                print(f"n = {self.n}, OPT= {self.opt}, GRAD = {grad}, COST_P = {self.costs_avg_plus}, COST_M = {self.costs_avg_minus}, tau = {tau_n}, perturbations = {self.perturbations}, perturbation = {self.perturbation}")

                #print(f"COST_P = {self.costs_avg_plus}, COST_M = {self.costs_avg_minus}, tau = {tau_n}")
                with open(f"{self.log_dir}/theta_costs_v.txt", "a") as file:
                    file.write(f"{self.opt};{self.costs_avg_plus};{self.costs_avg_minus};{cost};{total_req};{served_req};{served};{average_resource_usage};{self.state}\n")
                self.costs_avg_plus = 0
                self.costs_avg_minus = 0
                
               
                running_function_count = 0
                idle_function_count = 0
                init_free_function_count = 0
                init_reserved_function_count = 0
                
                simulator.rang_delta_plus = np.random.default_rng(simulator.seed)
                #simulator.rang_delta_minus = np.random.default_rng(simulator.seed)
                simulator.rang_delta_min_plus = np.random.default_rng(simulator.seed)
                #simulator.rang_delta_min_minus = np.random.default_rng(simulator.seed)

                #simulator.initialiaze_system(simulator.t,running_function_count, idle_function_count, init_free_function_count, init_reserved_function_count )
                #np.random.seed(self.n + self.K)
                delta_n = self.k_delta / (self.n ** (2.0 / 3.0))
                
                all_choices = np.array(self.prtb)
                #rang_perturb = np.random.default_rng(simulator.seed+10)

                # Use random indices (0 or 1) for each row
                random_indices =  self.rang_perturb.integers(0, 2, size=all_choices.shape[0])
                # Select independently
                self.perturbation = all_choices[np.arange(all_choices.shape[0]), random_indices]

                self.perturbations =  self.perturbation * delta_n 
                
                simulator.missed_update = 0

        time_after_algo = time.time()
        algo_time = time_after_algo - current_checkpoint 
        self.has_rejected_job = False   
   
       
    def compute_cost(self, state=None):
        if state is None:
            raise Exception("State is not provided")
        self.state = state.copy()
        return np.dot(self.state, self.weights) + self.w_rej * self.has_rejected_job

