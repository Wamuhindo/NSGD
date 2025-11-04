# The main simulator for serverless computing platforms
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from AutoscalerFaas.SimProcess import ExpSimProcess, ConstSimProcess, ParetoSimProcess
from AutoscalerFaas.FunctionInstance import FunctionInstance
from AutoscalerFaas.utils import FunctionState, SystemState
from AutoscalerFaas.Algorithm import AutoScalingAlgorithm
import numpy as np
from numpy.random import default_rng, SeedSequence
import time
import pandas as pd

from tqdm import tqdm
import random



class ServerlessSimulator:
    """ServerlessSimulator is responsible for executing simulations of a sample serverless computing platform, mainly for the performance analysis and performance model evaluation purposes.

    Parameters
    ----------
    arrival_process : simfaas.SimProcess.SimProcess, optional
        The process used for generating inter-arrival samples, if absent, `arrival_rate` should be passed to signal exponential distribution, by default None
    warm_service_process : simfaas.SimProcess.SimProcess, optional
        The process which will be used to calculate service times, if absent, `warm_service_rate` should be passed to signal exponential distribution, by default None
    cold_service_process : simfaas.SimProcess.SimProcess, optional
        The process which will be used to calculate service times, if absent, `cold_service_rate` should be passed to signal exponential distribution, by default None
    expiration_threshold : float, optional
        The period of time after which the instance will be expired and the capacity release for use by others, by default 600
    max_time : float, optional
        The maximum amount of time for which the simulation should continue, by default 24*60*60 (24 hours)
    maximum_concurrency : int, optional
        The maximum number of concurrently executing function instances allowed on the system This will be used to determine when a rejection of request should happen due to lack of capacity, by default 1000

    Raises
    ------
    Exception
        Raises if neither arrival_process nor arrival_rate are present
    Exception
        Raises if neither warm_service_process nor warm_service_rate are present
    Exception
        Raises if neither cold_service_process nor cold_service_rate are present
    ValueError
        Raises if warm_service_rate is smaller than cold_service_rate
    """
    def __init__(self, arrival_process=None, warm_service_process=None, 
            cold_service_process=None, cold_start_process=None, service_process_type="Exponential", expiration_process_type="Exponential", maximum_concurrency=50,
            log_dir ="", **kwargs):
        super().__init__()
        self.seed = kwargs.get("seed",1)

        ss = SeedSequence(self.seed)
        cs_rng, svc_rng, exp_rng, arr_rng = [default_rng(s) for s in ss.spawn(4)]        
        # setup arrival process
        self.arrival_process = arrival_process
        # if the user wants a exponentially distributed arrival process
        if 'arrival_rate' in kwargs:
            self.arrival_process = ExpSimProcess(rate=kwargs.get('arrival_rate'),gen=arr_rng)
        # in the end, arrival process should be defined
        if self.arrival_process is None:
            raise Exception('Arrival process not defined!')

        # setup warm service process
        self.warm_service_process = warm_service_process
        if 'warm_service_rate' in kwargs:
            if service_process_type == "Pareto":
                shape = 1+np.sqrt(1.25) # x_min
                scale = (np.sqrt(1.25)/(1+np.sqrt(1.25))) # alpha
                self.warm_service_process = ParetoSimProcess(scale=scale,shape=shape,gen=svc_rng)    
            else:
                self.warm_service_process = ExpSimProcess(rate=kwargs.get('warm_service_rate'),gen=svc_rng)
        if self.warm_service_process is None:
            raise Exception('Warm Service process not defined!')

        # setup cold service process
        self.cold_service_process = cold_service_process
        if 'cold_service_rate' in kwargs:
            self.cold_service_process = ExpSimProcess(rate=kwargs.get('cold_service_rate'))
        if self.cold_service_process is None:
            raise Exception('Cold Service process not defined!')
        if 'cold_start_rate' in kwargs:
            self.cold_start_process = ExpSimProcess(rate=kwargs.get('cold_start_rate'),gen=cs_rng)
        if self.cold_start_process is None:
            raise Exception('Cold Start process not defined!')

        
        
        self.maximum_concurrency = maximum_concurrency

        # reset trace values
        self.reset_trace()

        k_delta = kwargs.get('k_delta',1)
        k_gamma = kwargs.get('k_gamma',[1,1,1])
        theta_init = kwargs.get('theta_init',[1,1,1])
        tau = kwargs.get('tau',1e4)
        T = kwargs.get('max_time',1e5)
        K = kwargs.get('K',1)
        K_exp = kwargs.get('K_exp',1000)
        
        if expiration_process_type == "Deterministic":
            self.expiration_process = ConstSimProcess(theta_init[2]/K_exp , gen=exp_rng)
        else: 
            self.expiration_process = ExpSimProcess(theta_init[2]/K_exp , gen=exp_rng)


        self.seed = kwargs.get('seed',1)
        self.rang_plus = np.random.default_rng(self.seed)
        self.rang_init= np.random.default_rng(self.seed)
        self.rang_minus = np.random.default_rng(self.seed)
        self.rang_delta_plus = np.random.default_rng(self.seed)
        self.rang_delta_minus = np.random.default_rng(self.seed)
        self.rang_delta_min_plus = np.random.default_rng(self.seed)
        self.rang_delta_min_minus = np.random.default_rng(self.seed)
        
        self.autoscaler = AutoScalingAlgorithm(N=maximum_concurrency, k_delta=k_delta, k_gamma=k_gamma,theta_init=theta_init,tau=tau, K=K, T=T, log_dir=log_dir)
        self.autoscaler.set_params(**kwargs)
        
        self.state = self.autoscaler.get_state()
        self.max_time = self.autoscaler.Tmax
        self.log_dir = log_dir

    def set_initial_state(self, running_function_instances, idle_function_instances, init_free_function_instances, init_reserved_function_instances):
        
        init_running_count = len(running_function_instances)
        init_idle_count = len(idle_function_instances)
        init_free_count = len(init_free_function_instances)
        init_reserved_count = len(init_reserved_function_instances)
        init_server_count = init_running_count + init_idle_count + init_free_count + init_reserved_count

        self.server_count = init_server_count
        self.running_count = init_running_count
        self.init_free_count = init_free_count
        self.init_reserved_count = init_reserved_count
        self.idle_count = init_server_count - (init_running_count + init_free_count + init_reserved_count)
        self.servers = [*running_function_instances, *idle_function_instances, *init_free_function_instances, *init_reserved_function_instances]


    def initialiaze_system(self, t, running_function_count, idle_function_count, init_free_function_count, init_reserved_function_count ):
        np.random.seed(self.seed)
        #print("TT", t)
        idle_functions = []

        warm_service_process = ExpSimProcess(rate=self.warm_service_process.rate)
        cold_service_process = ExpSimProcess(rate=self.cold_service_process.rate)
        expiration_process = ExpSimProcess(rate=self.expiration_process.rate)
        cold_start_process = ExpSimProcess(rate=self.cold_start_process.rate)

        
        np.random.seed(self.seed)
        idle_functions = []
        for _ in range(idle_function_count):
            f = FunctionInstance(t,
                                cold_service_process=cold_service_process,
                                warm_service_process=warm_service_process,
                                expiration_process=expiration_process,
                                cold_start_process=cold_start_process
                                )
            f.cold_start_process = self.cold_start_process
            f.warm_service_process = self.warm_service_process
            f.expiration_process = self.expiration_process
            f.cold_service_process = self.cold_service_process
            f.state = FunctionState.IDLE_ON
            # when will it be destroyed if no requests
            #f.next_termination = 100
            # so that they would be less likely to be chosen by scheduler
            f.creation_time = 0.01
            idle_functions.append(f)

        running_functions = []
        for _ in range(running_function_count):
            f = FunctionInstance(t,
                                cold_service_process=cold_service_process,
                                warm_service_process=warm_service_process,
                                expiration_process=expiration_process,
                                cold_start_process=cold_start_process
                                )
            f.cold_start_process = self.cold_start_process
            f.warm_service_process = self.warm_service_process
            f.expiration_process = self.expiration_process
            f.cold_service_process = self.cold_service_process

            f.state = FunctionState.IDLE_ON
            # transition it into running mode
            f.arrival_transition(t)

            running_functions.append(f)
        
        init_free_functions = []    
        for _ in range(init_free_function_count):
            f = FunctionInstance(t,
                                cold_service_process=cold_service_process,
                                warm_service_process=warm_service_process,
                                expiration_process=expiration_process,
                                cold_start_process=cold_start_process
                                )

            f.cold_start_process = self.cold_start_process
            f.warm_service_process = self.warm_service_process
            f.expiration_process = self.expiration_process
            f.cold_service_process = self.cold_service_process

            f.make_Init_Free()


            init_free_functions.append(f)
            
        init_reserved_functions = []     
        for _ in range(init_reserved_function_count):
            f = FunctionInstance(t,
                                cold_service_process=cold_service_process,
                                warm_service_process=warm_service_process,
                                expiration_process=expiration_process,
                                cold_start_process=cold_start_process
                                )
            f.cold_start_process = self.cold_start_process
            f.warm_service_process = self.warm_service_process
            f.expiration_process = self.expiration_process
            f.cold_service_process = self.cold_service_process


            f.make_Init_Reserved()

            init_reserved_functions.append(f)
        
        self.set_initial_state(running_functions, idle_functions, init_free_functions, init_reserved_functions)

    def reset_trace(self):
        """resets all the historical data to prepare the class for a new simulation
        """
        # an archive of previous servers
        self.prev_servers = []
        self.total_req_count = 0
        self.total_cold_count = 0
        self.total_init_free_count = 0
        self.total_init_reserved_count = 0
        self.total_warm_count = 0
        self.total_reject_count = 0
        # current state of instances
        self.servers:list[FunctionInstance] = []
        self.server_count = 0
        self.running_count = 0
        self.idle_count = 0
        self.init_free_count = 0
        self.init_reserved_count = 0
        self.init_free_booked_count = 0
        self.reject_count = 0
        # history results
        self.hist_times = []
        self.hist_server_count = []
        self.hist_server_running_count = []
        self.hist_server_idle_count = []
        self.hist_server_init_reserved_count = []
        self.hist_server_init_free_count = []
        self.hist_server_queued_jobs_count = []
        self.hist_req_cold_idxs = []
        self.hist_req_init_free_idxs = []
        self.hist_req_init_reserved_idxs = []
        self.hist_req_warm_idxs = []
        self.hist_req_queued_idxs = []
        self.hist_req_rej_idxs = []
        #queued jobs
        self.queued_jobs = []
        self.queued_jobs_count = 0
        self.total_queued_jobs_count = 0
        self.total_requests_log = []
        self.served_requests_log = []
        self.last_t = 0
        self.t = 0
        self.total_finished = 0
        self.last_total_finished = 0
        self.job_rejected = False
        self.missed_update = 0

    def has_server(self):
        """Returns True if there are still instances (servers) in the simulated platform, False otherwise.

        Returns
        -------
        bool
            Whether or not the platform has instances (servers)
        """
        return len(self.servers) > 0

    def __str__(self):
        return f"idle/running/total: \t {self.idle_count}/{self.running_count}/{self.server_count}"

    def req(self):
        """Generate a request inter-arrival from `self.arrival_process`

        Returns
        -------
        float
            The generated inter-arrival sample
        """
        return self.arrival_process.generate_trace()
    
    def current_concurrency(self):
        return (self.running_count + self.init_reserved_count + self.init_free_booked_count)
    def current_cold_servers(self):
        return self.maximum_concurrency - (self.running_count + self.init_reserved_count + self.init_free_booked_count+self.init_free_count+self.idle_count)
    
    def has_reached_max_concurrency(self):
        total_now = self.current_concurrency()
        assert  total_now <= self.maximum_concurrency, f"Concurrency limit exceeded {total_now}, something is wrong"
        return total_now >= self.maximum_concurrency

    def cold_start_arrival(self, t, theta=1):
        """Goes through the process necessary for a cold start arrival which includes generation of a new function instance in the `COLD` state and adding it to the cluster.

        Parameters
        ----------
        t : float
            The time at which the arrival has happened. This is used to record the creation time for the server and schedule the expiration of the instance if necessary.
        """
        self.total_req_count += 1

        # reject request if maximum concurrency reached
        current_cold_servers = self.current_cold_servers()
        if self.has_reached_max_concurrency() or current_cold_servers<=0:
            self.total_reject_count += 1
            self.job_rejected = True
            self.hist_req_rej_idxs.append(len(self.hist_times) - 1)
            return

        self.total_cold_count += 1
        self.hist_req_cold_idxs.append(len(self.hist_times) - 1)
        
        self.hist_req_init_reserved_idxs.append(len(self.hist_times) - 1)
        self.init_reserved_count += 1
        new_server = FunctionInstance(t, self.cold_service_process, self.warm_service_process, self.expiration_process,self.cold_start_process)
        new_server.make_Init_Reserved()
        self.servers.append(new_server)
        self.server_count += 1
        self.total_init_reserved_count += 1
        
        #self.running_count += 1 : CHECK THIS
        self.start_init_free_servers( t, theta)

    def start_init_free_servers(self, t, theta, type="Cold start"):
        
        if theta < 0:
            return

        current_cold_servers = self.current_cold_servers()
        pi_theta = max(0, theta - self.init_free_count)
        current_concurrency = self.current_concurrency()
        init_free = self.maximum_concurrency - current_concurrency  if current_concurrency + pi_theta > self.maximum_concurrency else pi_theta
        
        
        if current_cold_servers > 0:
            init_free = min(current_cold_servers,init_free)
            #print("COld arrival theta",theta, init_free, current_cold_servers)
            for _ in range(int(init_free)):
                self.total_init_free_count += 1
                self.hist_req_init_free_idxs.append(len(self.hist_times) - 1)
                self.init_free_count += 1
                self.server_count += 1
                new_server = FunctionInstance(t, self.cold_service_process, self.warm_service_process, self.expiration_process,self.cold_start_process)
                new_server.make_Init_Free()
                self.servers.append(new_server)
        else:
            #print("STARTING INIT FREE", current_cold_servers, pi_theta, current_concurrency, init_free,type)
            #print("Concurrency",self.running_count, self.init_reserved_count, self.init_free_booked_count)
            #print(f"No cold server. In {type}")
            return

    def schedule_warm_instance(self, t):
        """Goes through a process to determine which warm instance should process the incoming request.

        Parameters
        ----------
        t : float
            The time at which the scheduling is happening

        Returns
        -------
        simfaas.FunctionInstance.FunctionInstance
            The function instances that the scheduler has selected for the incoming request.
        """
        idle_instances = [s for s in self.servers if s.is_idle_on()]
        creation_times = [s.creation_time for s in idle_instances]
        
        # scheduling mechanism
        creation_times = np.array(creation_times)
        # find the newest instance
        idx = np.argmax(creation_times)
        return idle_instances[idx]
    
    def schedule_init_free_instance(self, t):
        """Goes through a process to determine which init_free instance should process the incoming request.

        Parameters
        ----------
        t : float
            The time at which the scheduling is happening

        Returns
        -------
        simfaas.FunctionInstance.FunctionInstance
            The function instances that the scheduler has selected for the incoming request.
        """
        init_free_instances = [s for s in self.servers if (s.is_init_free() and not s.is_reserved())]
        
        creation_times = [s.creation_time for s in init_free_instances]
        
        # scheduling mechanism
        creation_times = np.array(creation_times)
        # find the newest instance
        idx = np.argmax(creation_times)
        return init_free_instances[idx]

    def warm_start_arrival(self, t, theta):
        """Goes through the process necessary for a warm start arrival which includes selecting a warm instance for processing and recording the request information.

        Parameters
        ----------
        t : float
            The time at which the arrival has happened. This is used to record the creation time for the server and schedule the expiration of the instance if necessary.
        """
        self.total_req_count += 1

        # reject request if maximum concurrency reached
        if self.has_reached_max_concurrency():
            self.total_reject_count += 1
            self.job_rejected = True
            self.hist_req_rej_idxs.append(len(self.hist_times) - 1)
            return

        self.hist_req_warm_idxs.append(len(self.hist_times) - 1)

        # schedule the request
        instance = self.schedule_warm_instance(t)
        was_idle = instance.is_idle_on()
        instance.arrival_transition(t)
        
        # transition from idle to running
        self.total_warm_count += 1
        if was_idle:
            self.idle_count -= 1
            self.running_count += 1
            # a instance has passed from idle to running
            self.served_requests_log.append(t)
            if self.idle_count < self.autoscaler.get_theta_step()[1]:
                to_start = theta - self.idle_count
                if to_start < 0:
                    self.missed_update +=1
                self.start_init_free_servers(t,to_start,"Warm start")

    def init_free_arrival(self, t, theta):
        """Goes through the process necessary for a warm start arrival which includes selecting a warm instance for processing and recording the request information.

        Parameters
        ----------
        t : float
            The time at which the arrival has happened. This is used to record the creation time for the server and schedule the expiration of the instance if necessary.
        """
        self.total_req_count += 1
        self.total_queued_jobs_count += 1
        self.queued_jobs_count += 1

        # reject request if maximum concurrency reached
        current_cold_servers = self.current_cold_servers()
        if self.has_reached_max_concurrency():
            self.total_reject_count += 1
            self.job_rejected = True
            self.hist_req_rej_idxs.append(len(self.hist_times) - 1)
            return

        self.hist_req_queued_idxs.append(len(self.hist_times) - 1)

        # schedule the request
        instance = self.schedule_init_free_instance(t)
        was_init_free = instance.is_init_free()
        instance.arrival_transition(t)
        self.init_free_count -= 1
        self.init_free_booked_count += 1
        #if instance.get_next_transition_time(t) > instance.cold_start_process.generate_trace()/3:
        self.total_cold_count += 1
        #self.running_count += 1 : CHECK THIS
        self.start_init_free_servers(t, theta,"Init free arrival")

    def get_trace_end(self):
        """Get the time at which the trace (one iteration of the simulation) has ended. This mainly due to the fact that we keep on simulating until the trace time goes beyond max_time, but the time is incremented until the next event.

        Returns
        -------
        float
            The time at which the trace has ended
        """
        return self.hist_times[-1]

    def calculate_time_lengths(self):
        """Calculate the time length for each step between two event transitions. Records the values in `self.time_lengths`.
        """
        self.time_lengths = np.diff(self.hist_times)

    def get_average_resource_usage(self):
        resource_usage = [ s.get_resource_usages() for s in self.servers]
        average_usage = np.mean(resource_usage, axis=0)
        return average_usage
    def get_average_server_count(self):
        """Get the time-average server count.

        Returns
        -------
        float
            Average server count
        """
        avg_server_count = (self.hist_server_count * self.time_lengths).sum() / self.get_trace_end()
        return avg_server_count

    def get_average_server_running_count(self):
        """Get the time-averaged running server count.

        Returns
        -------
        float
            Average running server coutn
        """
        avg_running_count = (self.hist_server_running_count * self.time_lengths).sum() / self.get_trace_end()
        return avg_running_count

    def get_average_server_idle_count(self):
        """Get the time-averaged idle server count.

        Returns
        -------
        float
            Average idle server count
        """
        avg_idle_count = (self.hist_server_idle_count * self.time_lengths).sum() / self.get_trace_end()
        return avg_idle_count
    
    def get_average_server_init_free_count(self):
        """Get the time-averaged idle server count.

        Returns
        -------
        float
            Average idle server count
        """
        avg_init_free_count = (self.hist_server_init_free_count * self.time_lengths).sum() / self.get_trace_end()
        return avg_init_free_count
    
    def get_average_server_init_reserved_count(self):
        """Get the time-averaged idle server count.

        Returns
        -------
        float
            Average idle server count
        """
        avg_init_reserved_count = (self.hist_server_init_reserved_count * self.time_lengths).sum() / self.get_trace_end()
        return avg_init_reserved_count
    
    def get_average_server_queued_jobs_count(self):
        """Get the time-averaged idle server count.

        Returns
        -------
        float
            Average idle server count
        """
        avg_init_reserved_count = (self.hist_server_queued_jobs_count * self.time_lengths).sum() / self.get_trace_end()
        return avg_init_reserved_count


    def get_index_after_time(self, t):
        """Get the first historical array index (for all arrays storing hisotrical events) that is after the time t.

        Parameters
        ----------
        t : float
            The time in the beginning we want to skip

        Returns
        -------
        int
            The calculated index in `self.hist_times`
        """
        return np.min(np.where(np.array(self.hist_times) > t))

    def get_skip_init(self, skip_init_time=None, skip_init_index=None):
        """Get the minimum index which satisfies both the time and index count we want to skip in the beginning of the simulation, which is used to reduce the transient effect for calculating the steady-state values.

        Parameters
        ----------
        skip_init_time : float, optional
            The amount of time skipped in the beginning, by default None
        skip_init_index : [type], optional
            The number of indices we want to skip in the historical events, by default None

        Returns
        -------
        int
            The number of indices after which both index and time requirements are satisfied
        """
        # how many initial values should be skipped
        skip_init = 0
        if skip_init_time is not None:
            skip_init = self.get_index_after_time(skip_init_time)
        if skip_init_index is not None:
            skip_init = max(skip_init, skip_init_index)
        return skip_init

    def get_request_custom_states(self, hist_states, skip_init_time=None, skip_init_index=None):
        """Get request statistics for an array of custom states.

        Parameters
        ----------
        hist_states : list[object]
            An array of custom states calculated by the user for which the statistics should be calculated, should be the same size as `hist_*` objects, these values will be used as the keys for the returned dataframe.
        skip_init_time : float, optional
            The amount of time skipped in the beginning, by default None
        skip_init_index : int, optional
            The number of indices that should be skipped in the beginning to calculate steady-state results, by default None

        Returns
        -------
        pandas.DataFrame
            A pandas dataframe including different statistics like `p_cold` (probability of cold start)
        """
        req_skip_init = self.get_skip_init(skip_init_time=skip_init_time, 
                                        skip_init_index=skip_init_index)

        state_req_colds = {}
        state_req_warm = {}
        state_req_rejs = {}
        state_req_init_free = {}
        state_req_init_reserved = {}
        state_req_queued = {}
        for s in hist_states[req_skip_init:]:
            if s not in state_req_colds:
                state_req_colds[s] = 0
                state_req_warm[s] = 0
                state_req_rejs[s] = 0
                state_req_init_free[s] = 0
                state_req_init_reserved[s] = 0
                state_req_queued[s] = 0

        hist_req_cold = [i for i in self.hist_req_cold_idxs if i > req_skip_init]
        hist_req_warm = [i for i in self.hist_req_warm_idxs if i > req_skip_init]
        hist_req_rej = [i for i in self.hist_req_rej_idxs if i > req_skip_init]
        hist_req_init_free = [i for i in self.hist_req_init_free_idxs if i > req_skip_init]
        hist_req_init_reserved = [i for i in self.hist_req_init_reserved_idxs if i > req_skip_init]
        hist_req_queued = [i for i in self.hist_req_queued_idxs if i > req_skip_init]


        for idx in hist_req_cold:
            state_req_colds[hist_states[idx]] += 1
        for idx in hist_req_warm:
            state_req_warm[hist_states[idx]] += 1
        for idx in hist_req_rej:
            state_req_warm[hist_states[idx]] += 1
        for idx in hist_req_init_free:
            state_req_init_free[hist_states[idx]] += 1
        for idx in hist_req_init_reserved:
            state_req_init_reserved[hist_states[idx]] += 1
        for idx in hist_req_queued:
            state_req_queued[hist_states[idx]] += 1

        states = list(state_req_colds.keys())
        state_req_colds = list(state_req_colds.values())
        state_req_warm = list(state_req_warm.values())
        state_req_rejs = list(state_req_rejs.values())
        state_req_init_free = list(state_req_init_free.values())
        state_req_init_reserved = list(state_req_init_reserved.values())
        state_req_queued = list(state_req_queued.values())

        reqdf = pd.DataFrame(data = {'state': states, 'cold': state_req_colds, 'warm': state_req_warm, 'rej': state_req_rejs, 'init_free': state_req_init_free, 'init_reserved': state_req_init_reserved, 'queued': state_req_queued})
        reqdf['total'] = reqdf['cold'] + reqdf['warm'] + reqdf['rej']
        reqdf['p_cold'] = reqdf['cold'] / reqdf['total']
        return reqdf

    def analyze_custom_states(self, hist_states, skip_init_time=None, skip_init_index=None):
        """Analyses a custom states list and calculates the amount of time spent in each state each time we enterred that state, and the times at which transitions have happened.

        Parameters
        ----------
        hist_states : list[object]
            The states calculated, should have the same dimensions as the `hist_*` arrays.
        skip_init_time : float, optional
            The amount of time skipped in the beginning, by default None
        skip_init_index : int, optional
            The number of indices skipped in the beginning, by default None

        Returns
        -------
        list[float], list[float]
            (residence_times, transition_times) where residence_times is an array of the amount of times we spent in each state, and transition_times are the moments of time at which each transition has occured
        """
        skip_init = self.get_skip_init(skip_init_time=skip_init_time, 
                                        skip_init_index=skip_init_index)

        values = hist_states[skip_init:]
        time_lengths = self.time_lengths[skip_init:]

        residence_times = {}
        transition_times = {}
        curr_time_sum = time_lengths[0]
        # encode states
        for idx in range(1,len(values)):
            if values[idx] == values[idx-1]:
                curr_time_sum += time_lengths[idx]
            else:
                if values[idx-1] in residence_times:
                    residence_times[values[idx-1]].append(curr_time_sum)
                else:
                    residence_times[values[idx-1]] = [curr_time_sum]

                transition_pair = (values[idx-1], values[idx])
                if transition_pair in transition_times:
                    transition_times[transition_pair].append(curr_time_sum)
                else:
                    transition_times[transition_pair] = [curr_time_sum]
                
                curr_time_sum = time_lengths[idx]

        return residence_times, transition_times

    def get_average_residence_times(self, hist_states, skip_init_time=None, skip_init_index=None):
        """Get the average residence time for each state in custom state encoding.

        Parameters
        ----------
        hist_states : list[object]
            The states calculated, should have the same dimensions as the `hist_*` arrays.
        skip_init_time : float, optional
            The amount of time skipped in the beginning, by default None
        skip_init_index : int, optional
            The number of indices skipped in the beginning, by default None

        Returns
        -------
        float
            The average residence time for each state, averaged over the times we transitioned into that state
        """
        residence_times, _ = self.analyze_custom_states(hist_states, skip_init_time, skip_init_index)

        residence_time_avgs = {}
        for s in residence_times:
            residence_time_avgs[s] = np.mean(residence_times[s])

        return residence_time_avgs

    def get_cold_start_prob(self):
        """Get the probability of cold start for the simulated trace.

        Returns
        -------
        float
            The probability of cold start calculated by dividing the number of cold start requests, over all requests
        """
        return self.total_cold_count / self.total_req_count


    def get_average_lifespan(self):
        """Get the average lifespan of each instance, calculated by the amount of time from creation of instance, until its expiration.

        Returns
        -------
        float
            The average lifespan
        """
        life_spans = np.array([s.get_life_span() for s in self.prev_servers])
        if len(life_spans):
            return life_spans.mean()
        else:
            return np.inf

    
    def get_result_dict(self):
        """Get the results of the simulation as a dict, which can easily be integrated into web services.

        Returns
        -------
        dict
            A dictionary of different characteristics.
        """
        return {
            "reqs_cold": self.total_cold_count,
            "reqs_total": self.total_req_count,
            "reqs_init_free": self.total_init_free_count,
            "reqs_init_reserved": self.total_init_reserved_count,
            "reqs_warm": self.total_warm_count,
            "reqs_queued": self.total_queued_jobs_count,
            "reqs_cold": self.total_cold_count,
            "prob_cold": self.get_cold_start_prob(),
            "reqs_reject": self.total_reject_count,
            "prob_reject": self.total_reject_count / self.total_req_count,
            "lifespan_avg": self.get_average_lifespan(),
            "inst_count_avg": self.get_average_server_count(),
            "inst_running_count_avg": self.get_average_server_running_count(),
            "inst_idle_count_avg": self.get_average_server_idle_count(),
            "inst_init_free_count_avg": self.get_average_server_init_free_count(),
            "inst_init_reserved_count_avg": self.get_average_server_init_reserved_count(),
            "inst_queued_jobs_count_avg": self.get_average_server_queued_jobs_count()
        }

    def print_trace_results(self):
        """Print a brief summary of the results of the trace.
        """
        self.calculate_time_lengths()

        print(f"Cold Starts / total requests: \t {self.total_cold_count} / {self.total_req_count}")
        print(f"Cold Start Probability: \t {self.total_cold_count / self.total_req_count:.4f}")

        print(f"Rejection / total requests: \t {self.total_reject_count} / {self.total_req_count}")
        print(f"Rejection Probability: \t\t {self.total_reject_count / self.total_req_count:.4f}")

        # average instance life span
        life_spans = np.array([s.get_life_span() for s in self.prev_servers])
        if len(life_spans) > 0:
            print(f"Average Instance Life Span: \t {life_spans.mean():.4f}")

        # average instance count
        print(f"Average Server Count:  \t\t {self.get_average_server_count():.4f}")
        # average running count
        print(f"Average Running Count:  \t {self.get_average_server_running_count():.4f}")
        # average idle count
        print(f"Average Idle Count:  \t\t {self.get_average_server_idle_count():.4f}")
        # average init free count
        print(f"Average Init Free Count:  \t {self.get_average_server_init_free_count():.4f}")
        # average init reserved count
        print(f"Average Init Reserved Count:  \t {self.get_average_server_init_reserved_count():.4f}")
        # average queued jobs count
        print(f"Average Queued Jobs Count:  \t {self.get_average_server_queued_jobs_count():.4f}")
        # total init_free count
        print(f"Total Init Free Count:  \t {self.total_init_free_count}")
        # total init_reserved count
        print(f"Total Init Reserved Count:  \t {self.total_init_reserved_count}")
        # total queued jobs count
        print(f"Total Queued Jobs Count:  \t {self.total_queued_jobs_count}")

    def trace_condition(self, t):
        """The condition for resulting the trace, we continue the simulation until this function returns false.

        Parameters
        ----------
        t : float
            current time in the simulation since the start of simulation

        Returns
        -------
        bool
            True if we should continue the simulation, false otherwise
        """
        #return t < self.max_time
        return self.autoscaler.running_condition()

    @staticmethod
    def print_time_average(vals, probs, column_width=15):
        """Print the time average of states.

        Parameters
        ----------
        vals : list[object]
            The values for which the time average is to be printed
        probs : list[float]
            The probability of each of the members of the values array
        column_width : int, optional
            The width of the printed result for `vals`, by default 15
        """
        print(f"{'Value'.ljust(column_width)} Prob")
        print("".join(["="]*int(column_width*1.5)))
        for val, prob in zip(vals, probs):
            print(f"{str(val).ljust(column_width)} {prob:.4f}")

    def calculate_time_average(self, values, skip_init_time=None, skip_init_index=None):
        """calculate_time_average calculates the time-averaged of the values passed in with optional skipping a specific number of time steps (skip_init_index) and a specific amount of time (skip_init_time).

        Parameters
        ----------
        values : list
            A list of values with the same dimensions as history array (number of transitions)
        skip_init_time : Float, optional
            Amount of time skipped in the beginning to let the transient part of the solution pass, by default None
        skip_init_index : [type], optional
            Number of steps skipped in the beginning to let the transient behaviour of system pass, by default None

        Returns
        -------
        (list, list)
            returns (unq_vals, val_times) where unq_vals is the unique values inside the values list
            and val_times is the portion of the time that is spent in that value.
        """
        assert len(values) == len(self.time_lengths), "Values shoud be same length as history array (number of transitions)"

        skip_init = self.get_skip_init(skip_init_time=skip_init_time, 
                                        skip_init_index=skip_init_index)

        values = values[skip_init:]
        time_lengths = self.time_lengths[skip_init:]

        # get unique values
        unq_vals = list(set(values))
        val_times = []
        for val in unq_vals:
            t = time_lengths[[v == val for v in values]].sum()
            val_times.append(t)

        # convert to percent
        val_times = np.array(val_times)
        val_times = val_times / val_times.sum()
        return unq_vals, val_times

    
    def is_warm_available(self, t):
        """Whether we have at least one available instance in the warm pool that can process requests

        Parameters
        ----------
        t : float
            Current time

        Returns
        -------
        bool
            True if at least one server is able to accept a request
        """
        return self.idle_count > 0
    
    def is_init_free_available(self, t):
        """Whether we have at least one available instance in the init-free pool that can process requests

        Parameters
        ----------
        t : float
            Current time

        Returns
        -------
        bool
            True if at least one server is able to accept a request
        """
        #print("INIT FREE COUNT/ BOOKED",self.init_free_count,self.init_free_booked_count)
        #init_free_instances = [s for s in self.servers if (s.is_init_free() and not s.is_reserved())]
        #init_free_count = len(init_free_instances)
        #assert len(init_free_instances) == self.init_free_count, f"Incoherent init free count {len(init_free_instances)} != {self.init_free_count}"
        #print("INfree available", self.init_free_count,  self.init_free_booked_count, self.init_free_count > 0 and self.init_free_booked_count < self.init_free_count)
        return self.init_free_count > 0


    def update_hist_arrays(self, t):
        """Update history arrays

        Parameters
        ----------
        t : float
            Current time
        """
        self.hist_server_count.append(self.server_count)
        self.hist_server_running_count.append(self.running_count)
        self.hist_server_idle_count.append(self.idle_count)
        self.hist_server_init_reserved_count.append(self.init_reserved_count)
        self.hist_server_init_free_count.append(self.init_free_count)
        self.hist_server_queued_jobs_count.append(self.queued_jobs_count)
        

    def update_state(self):
        """Update the state
        """
        
        
        self.state[SystemState.INITIALIZING.value] = self.init_free_count + self.init_free_booked_count + self.init_reserved_count
        self.state[SystemState.INIT_RESERVED.value] = self.init_reserved_count + self.init_free_booked_count
        self.state[SystemState.BUSY.value] = self.running_count
        self.state[SystemState.IDLE_ON.value] = self.idle_count
        self.state[SystemState.COLD.value] = self.maximum_concurrency - (self.init_free_count + self.init_free_booked_count+ self.init_reserved_count + self.running_count + self.idle_count)
        self.autoscaler.set_has_rejected_job(self.job_rejected)
        # running = len([s for s in self.servers if s.is_busy()])
        # assert running == self.running_count, f"running not the same {running}, {self.running_count}"

        # init_reserved = len([s for s in self.servers if s.is_init_reserved()])
        # assert init_reserved == self.init_reserved_count, f"init reserved not the same {init_reserved}, {self.init_reserved_count}"
        # init_free = len([s for s in self.servers if s.is_init_free()])
        # assert init_free == self.init_free_count+self.init_free_booked_count, f"init free not the same {init_free}, {self.init_free_count+self.init_free_booked_count}"
        # idle = len([s for s in self.servers if s.is_idle_on()])
        # assert idle == self.idle_count, f"init reserved not the same {idle}, {self.idle_count}"
        if False:
            print("+++++++++++++++++++++++++")
            print("RUNNING",running, self.running_count)
            print("INIT RESERVED",init_reserved,self.init_reserved_count)
            print("INIT FREE",init_free,self.init_free_count)
            print("IDLE",idle,self.idle_count)
            print("+++++++++++++++++++++++++")

    def get_request_stats_between(self, start_t, end_t):
        total = sum(start_t <= t <= end_t for t in self.total_requests_log)
        served = sum(start_t <= t <= end_t for t in self.served_requests_log)
        return total, served

    def calculate_cost(self):
        """
        Calculate cost based on current system state
        """
        # Forward the cost calculation to the autoscaler
        #print(self.state)
        #self.all_states.append(self.state.copy())
        cost = self.autoscaler.compute_cost(self.state)
        self.job_rejected = False
        return cost
            
    def generate_trace(self, debug_print=False, progress=False):
        """Generate a sample trace.

        Parameters
        ----------
        debug_print : bool, optional
            If True, will print each transition occuring during the simulation, by default False
        progress : bool, optional
            Whether or not the progress should be outputted using the `tqdm` library, by default False

        Raises
        ------
        Exception
            Raises if FunctionInstance enters an unknown state (other than `IDLE` for idle or `TERM` for terminated) after making an internal transition
        """
        pbar = None
        if progress:
            pbar = tqdm(total=int(self.max_time))

        t = 0
        pbar_t_update = 0
        pbar_interval = int(self.max_time / 100)
        next_arrival = t + self.req()

        while self.trace_condition(t):
    
            if progress:
                # if int(t - pbar_t_update) > pbar_interval:
                #     pbar.update(int(t) - pbar_t_update)
                #     pbar_t_update = int(t)
                if int(self.autoscaler.t - pbar_t_update) > pbar_interval:
                    pbar.update(int(self.autoscaler.t) - pbar_t_update)
                    pbar_t_update = int(self.autoscaler.t)
                    
            self.hist_times.append(t)
            self.update_hist_arrays(t)
            if debug_print:
                print()
                print(f"Time: {t:.2f} \t NextArrival: {next_arrival:.2f}")
                print(self)
                # print state of all servers
                [print(s) for s in self.servers]

            # if there are no servers, next transition is arrival
            if self.has_server() == False:
                theta = self.autoscaler.get_theta_step()[0]
                #print("THETA",theta)
                t = next_arrival
                self.t = t
                self.total_requests_log.append(t)
                next_arrival = t + self.req()
                # no servers, so cold start
                self.cold_start_arrival(t,theta=theta)
                self.update_state()
                self.autoscaler.simulate_step(self.state,self)
                continue
            #print("T",t)
            # if there are servers, next transition is the soonest one
            server_next_transitions = np.array([s.get_next_transition_time(t) for s in self.servers])

            # if next transition is arrival
            if (next_arrival - t) < server_next_transitions.min():
                t = next_arrival
                self.t = t
                next_arrival = t + self.req()
                self.total_requests_log.append(t)
                
                
                # if warm start
                if self.is_warm_available(t):
                    theta = self.autoscaler.get_theta_step()[0]
                    self.warm_start_arrival(t,theta)
                
                elif self.is_init_free_available(t):
                     theta = self.autoscaler.get_theta_step()[0]
                     self.init_free_arrival(t,theta)
                # if cold start
                else:
                    theta = self.autoscaler.get_theta_step()[0]
                    self.cold_start_arrival(t,theta=theta)

                self.update_state()
                self.autoscaler.simulate_step(self.state,self)
                continue

            # if next transition is a state change in one of servers
            else:
                # find the server that needs transition
                
                idx = server_next_transitions.argmin()
                t = t + server_next_transitions[idx]
                self.t = t
                old_state = self.servers[idx].get_state()
                
                new_state = self.servers[idx].make_transition()
                
                # delete instance if it was just terminated
                if new_state == FunctionState.COLD:
                    self.prev_servers.append(self.servers[idx])
                    self.idle_count -= 1
                    self.server_count -= 1
                    del self.servers[idx]
                    if debug_print:
                        print(f"Termination for: # {idx}")
                
                # if request has done processing (exit event)
                elif new_state == FunctionState.IDLE_ON:
                    #self.served_requests_log.append(t)
                    self.total_finished += 1
                    if old_state == FunctionState.BUSY:
                        # transition from running to idle
                        self.running_count -= 1
                    elif old_state == FunctionState.INIT_FREE and not self.servers[idx].is_reserved():
                        # Transition from init_free to idle
                        self.init_free_count -= 1
                        self.servers[idx].unreserve()
                    else:
                        # Force this only if we are running current class, not child classes
                        if self.__class__ == ServerlessSimulator:
                            raise Exception(f"Unknown transition in states: {new_state}")
                    self.idle_count += 1
                elif new_state == FunctionState.BUSY:
                    self.served_requests_log.append(t)
                    
                    if old_state == FunctionState.INIT_RESERVED:
                        # transition from init_reserved to running
                        self.init_reserved_count -= 1
                    elif old_state == FunctionState.INIT_FREE and self.servers[idx].is_reserved():
                        
                        # transition from init_free to running
                        self.servers[idx].update_next_transition(t)
                        self.servers[idx].unreserve()
                        #self.init_free_count -= 1
                        self.init_free_booked_count -= 1
                        self.queued_jobs_count -= 1
                        self.total_warm_count += 1
                        self.hist_req_warm_idxs.append(len(self.hist_times) - 1)
                    else:
                    # Force this only if we are running current class, not child classes
                        if self.__class__ == ServerlessSimulator:
                            raise Exception(f"Unknown transition in states: {new_state}")
                    self.running_count += 1 
                    
                else:
                    # force this only if we are running current class, not child classes
                    if self.__class__ == ServerlessSimulator:
                        raise Exception(f"Unknown transition in states: {new_state}")
                self.update_state()
                self.autoscaler.simulate_step(self.state,self)

        # after the trace loop, append the last time recorded
        self.hist_times.append(t)
        self.calculate_time_lengths()
        if progress:
            pbar.update(int(self.max_time) - pbar_t_update)
            pbar.close()
        
        
        np.savetxt(f"{self.log_dir}/theta.csv", self.autoscaler.thetas, delimiter=",", fmt="%2f")
        np.savetxt(f"{self.log_dir}/states.csv", self.autoscaler.states, delimiter=",", fmt="%2f")
        np.savetxt(f"{self.log_dir}/all_costs.csv", self.autoscaler.all_costs, delimiter=",", fmt="%2f")
        #np.savetxt(f"{self.log_dir}/costs.csv", self.autoscaler.costs, delimiter="," )  # Use "%.2f" for floats
        
        #np.savetxt("all_states.csv", self.autoscaler.all_states, delimiter=",", fmt="%2f")




def load_config(config_path):
    """Load configuration from JSON file.

    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file

    Returns
    -------
    dict
        Configuration dictionary
    """
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def parse_distribution_params(dist_config):
    """Parse distribution configuration into rate parameter.

    Parameters
    ----------
    dist_config : dict
        Dictionary with 'rate' and 'type' keys

    Returns
    -------
    float
        Rate parameter for the distribution
    """
    return dist_config.get('rate')


def run_single_experiment(config, seed, run_idx, total_runs, base_log_dir):
    """Run a single simulation experiment with given parameters.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    seed : int
        Random seed for this run
    run_idx : int
        Index of current run (0-based)
    total_runs : int
        Total number of runs
    base_log_dir : str
        Base directory for logging

    Returns
    -------
    dict
        Results dictionary including timing information
    """
    import json

    # Setup initial system state
    running_function_count = 0
    idle_function_count = 0
    init_free_function_count = 0
    init_reserved_function_count = 0
    t = 0

    # Extract parameters from config
    arrival_rate = config['arrival_rate']
    warm_service_rate = parse_distribution_params(config['warm_service'])
    cold_service_rate = parse_distribution_params(config['cold_service'])
    cold_start_rate = parse_distribution_params(config['cold_start'])
    expiration_rate = parse_distribution_params(config['expiration'])

    service_process_type = config['warm_service'].get('type', 'Exponential')
    expiration_process_type = config['expiration'].get('type', 'Exponential')

    optimization = config['optimization'].get('type', 'adam')
    learning_rate = config['optimization'].get('learning_rate', 0.01)

    # Algorithm parameters
    theta_init = config['theta'][0]  # First theta configuration
    tau = config['tau']
    max_concurrency = config['max_currency']
    max_time = config['max_time']
    K = config['K']

    # Additional parameters with defaults
    K_exp = config.get('K_exp', 1000)
    gamma_min = config.get('gamma_min', 1)
    exp_lr = np.array(config.get('exp_lr', [1, 1, 1]))
    prtb = config.get('prtb', [[-0.5, 0.5], [-0.5, 0.5], [-1, 1]])

    # Calculate derived parameters
    k_delta = config.get('k_delta', 1)
    k_gamma = np.array(config.get('k_gamma', [1, 1, 1]))

    # Create run-specific log directory
    current_time = time.strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(base_log_dir, f"run_{run_idx+1}_seed_{seed}")

    if not os.path.exists(run_log_dir):
        os.makedirs(run_log_dir)

    # Setup algorithm parameters
    algo_params = {
        "k_gamma": k_gamma,
        "k_delta": k_delta,
        "K": K,
        "theta_init": theta_init,
        "tau": tau,
        "max_time": max_time,
        "seed": seed,
        'K_exp': K_exp,
        'gamma_min': gamma_min,
        'exp_lr': exp_lr,
        'prtb': prtb,
    }

    # Save run configuration (convert numpy arrays to lists for JSON serialization)
    def convert_to_serializable(obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    run_config = {
        'run_index': run_idx + 1,
        'seed': seed,
        'arrival_rate': arrival_rate,
        'warm_service_rate': warm_service_rate,
        'cold_service_rate': cold_service_rate,
        'cold_start_rate': cold_start_rate,
        'expiration_rate': expiration_rate,
        'service_process_type': service_process_type,
        'expiration_process_type': expiration_process_type,
        'optimization': optimization,
        'learning_rate': learning_rate,
        'max_concurrency': max_concurrency,
        'max_time': max_time,
        'tau': tau,
        'K': K,
        'theta_init': theta_init,
    }

    # Add algo_params with conversion
    run_config.update(convert_to_serializable(algo_params))

    with open(os.path.join(run_log_dir, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Starting Run {run_idx+1}/{total_runs} with seed={seed}")
    print(f"Log directory: {run_log_dir}")
    print(f"{'='*80}\n")

    # Record start time
    start_time = time.time()
    start_perf = time.perf_counter()

    # Initialize random seeds
    random.seed(seed)
    np.random.seed(seed)

    # Create simulator
    sim = ServerlessSimulator(
        arrival_rate=arrival_rate,
        warm_service_rate=warm_service_rate,
        cold_service_rate=cold_service_rate,
        cold_start_rate=cold_start_rate,
        maximum_concurrency=max_concurrency,
        log_dir=run_log_dir,
        service_process_type=service_process_type,
        expiration_process_type=expiration_process_type,
        **algo_params
    )

    sim.optimization = optimization
    sim.initialiaze_system(t, running_function_count, idle_function_count,
                          init_free_function_count, init_reserved_function_count)

    # Run simulation
    sim.generate_trace(debug_print=False, progress=True)

    # Record end time
    end_time = time.time()
    end_perf = time.perf_counter()

    wall_clock_time = end_time - start_time
    cpu_time = end_perf - start_perf

    # Get results
    results = sim.get_result_dict()
    results['seed'] = seed
    results['run_index'] = run_idx + 1
    results['wall_clock_time_seconds'] = wall_clock_time
    results['cpu_time_seconds'] = cpu_time
    results['simulated_time'] = sim.get_trace_end()

    # Print results
    print(f"\nResults for Run {run_idx+1}:")
    sim.print_trace_results()
    print(f"\nExecution Time: {wall_clock_time:.2f} seconds ({wall_clock_time/60:.2f} minutes)")
    print(f"CPU Time: {cpu_time:.2f} seconds")
    print(f"Simulated Time: {results['simulated_time']:.2f}")

    # Save results
    with open(os.path.join(run_log_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save summary
    with open(os.path.join(run_log_dir, 'summary.txt'), 'w') as f:
        f.write(f"Run {run_idx+1} - Seed {seed}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Execution Time: {wall_clock_time:.2f} seconds\n")
        f.write(f"CPU Time: {cpu_time:.2f} seconds\n")
        f.write(f"Simulated Time: {results['simulated_time']:.2f}\n\n")
        f.write(json.dumps(results, indent=2))

    return results


def run_experiments_from_config(config_path):
    """Run multiple experiments from configuration file.

    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file
    """
    import json

    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # Get experiment parameters
    seeds = config.get('seeds', [1])  # Default to single seed
    exp_per_run = config.get('exp_per_run', 1)

    # Create base log directory
    current_time = time.strftime("%Y%m%d_%H%M%S")
    arrival_rate = config['arrival_rate']
    service_type = config['warm_service'].get('type', 'Exponential')
    expiration_type = config['expiration'].get('type', 'Exponential')
    theta_str = '_'.join(str(x) for x in config['theta'][0])

    base_log_dir = config.get('log_dir', 'logs/')
    base_log_dir = os.path.join(
        base_log_dir,
        f"experiment_arr{arrival_rate}_{service_type}_{expiration_type}",
        f"theta_{theta_str}_{current_time}"
    )

    if not os.path.exists(base_log_dir):
        os.makedirs(base_log_dir)

    print(f"Base log directory: {base_log_dir}")

    # Save master configuration
    with open(os.path.join(base_log_dir, 'experiment_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Run experiments for each seed
    all_results = []
    total_runs = len(seeds) * exp_per_run

    experiment_start = time.time()

    for exp_idx in range(exp_per_run):
        for seed_idx, seed in enumerate(seeds):
            run_idx = exp_idx * len(seeds) + seed_idx

            try:
                results = run_single_experiment(
                    config, seed, run_idx, total_runs, base_log_dir
                )
                all_results.append(results)
            except Exception as e:
                print(f"\nError in run {run_idx+1} with seed {seed}: {e}")
                import traceback
                traceback.print_exc()
                continue

    experiment_end = time.time()
    total_experiment_time = experiment_end - experiment_start

    # Aggregate results
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")
    print(f"Total runs completed: {len(all_results)}/{total_runs}")
    print(f"Total experiment time: {total_experiment_time:.2f} seconds ({total_experiment_time/60:.2f} minutes)")

    # Save aggregated results
    aggregated_results = {
        'total_runs': len(all_results),
        'total_experiment_time_seconds': total_experiment_time,
        'config': config,
        'runs': all_results
    }

    with open(os.path.join(base_log_dir, 'aggregated_results.json'), 'w') as f:
        json.dump(aggregated_results, f, indent=2)

    # Create summary CSV
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(base_log_dir, 'all_runs_summary.csv'), index=False)
        print(f"\nSummary CSV saved to: {os.path.join(base_log_dir, 'all_runs_summary.csv')}")

    # Print statistics
    if all_results:
        print(f"\nStatistics across {len(all_results)} runs:")
        print(f"  Mean cold start probability: {np.mean([r['prob_cold'] for r in all_results]):.4f} ± {np.std([r['prob_cold'] for r in all_results]):.4f}")
        print(f"  Mean rejection probability: {np.mean([r['prob_reject'] for r in all_results]):.4f} ± {np.std([r['prob_reject'] for r in all_results]):.4f}")
        print(f"  Mean execution time: {np.mean([r['wall_clock_time_seconds'] for r in all_results]):.2f} ± {np.std([r['wall_clock_time_seconds'] for r in all_results]):.2f} seconds")

    print(f"\nAll results saved to: {base_log_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description='Serverless Computing Platform Simulator with NSGD Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simulation with default input.json
  python ServerlessSimulator.py --input input.json

  # Run simulation with custom configuration
  python ServerlessSimulator.py --input my_config.json
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input JSON configuration file'
    )

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        sys.exit(1)

    # Run experiments
    try:
        run_experiments_from_config(args.input)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    