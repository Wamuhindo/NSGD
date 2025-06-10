from AutoscalerFaasParallel.utils import SystemState
import numpy as np
import time

class AutoScalingAlgorithm:
    def __init__(self, N, k_delta, k_gamma,opt_init,tau, K, T=1e9, log_dir=""):
        self.N = N
        self.k_delta = k_delta
        self.k_gamma = k_gamma
        self.Tmax = T
        self.opt = opt_init
        self.opt_step = opt_init
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
        self.costs_avg = 0
        self.all_states = []
        self.all_thetas = []
        self.all_costs = []
        self.has_rejected_job=False
        self.last_chackpoint = time.time()
        self.steps = 0
        self.log_dir = log_dir

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
        self.weights[SystemState.INIT_RESERVED.value] = 100
        self.w_rej = 200 
        
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
    
    def get_opt_step(self):
        #return 2
        return self.opt_step
    
    def running_condition(self):
        
        return self.t < self.Tmax
    
       
    def compute_cost(self, state=None):
        if state is None:
            raise Exception("State is not provided")
        self.state = state.copy()
        return np.dot(self.state, self.weights) + self.w_rej * self.has_rejected_job

