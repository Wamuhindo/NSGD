# The main simulator for serverless computing platforms
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))


from AutoscalerFaasParallel.SimProcess import ExpSimProcess, ConstSimProcess, ParetoSimProcess
from AutoscalerFaasParallel.FunctionInstance import FunctionInstance
from AutoscalerFaasParallel.utils import FunctionState, SystemState
from AutoscalerFaasParallel.Algorithm import AutoScalingAlgorithm

from multiprocessing import Process, Event, Manager, Lock,  Barrier
import multiprocessing

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
    def __init__(self,id, arrival_process=None, warm_service_process=None, 
            cold_service_process=None,cold_start_process=None, service_process_type = "Exponential",expiration_process_type = "Exponential", max_time=1e6,
            maximum_concurrency=50, log_dir = "", **kwargs):
        super().__init__()
        
        self.seed = kwargs.get("seed",1234)
        self.opt_init = kwargs.get("opt_init",[0,0,0.01])

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

        # force this only if we are running the current class, not a child class (cause they might work differently)
        if self.__class__ == ServerlessSimulator:
            # if both warm and cold service rate is provided (exponential distribution)
            # then, warm service rate should be larger than cold service rate
            if 'warm_service_rate' in kwargs and 'cold_service_rate' in kwargs:
                if kwargs.get('warm_service_rate') < kwargs.get('cold_service_rate'):
                    raise ValueError("Warm service rate cannot be smaller than cold service rate!")

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

        if expiration_process_type == "Deterministic":
            self.expiration_process = ConstSimProcess(self.opt_init[2]/1e3 , gen=exp_rng)
        else: 
            self.expiration_process = ExpSimProcess(self.opt_init[2]/1e3 , gen=exp_rng)
        
        self.maximum_concurrency = maximum_concurrency
        self.id = id
        self.set_debug(False)
        self.set_progress(True)
        self.max_time = max_time
        self.log_dir = log_dir
        
        # reset trace values
        self.reset_trace()
        
        #self.autoscaler = AutoScalingAlgorithm(N=maximum_concurrency, k_delta=1, k_gamma=10/100,theta_init=2,tau=1e5, K=1, T=1e8)
        #self.state = self.autoscaler.get_state()
        
        
    def set_progress(self, progress):
        self.progress = progress
        
    def set_debug(self, debug):
        self.debug=debug
        
    def set_algo_params(self,k_delta, k_gamma, opt_init, tau, K):
        
        self.autoscaler = AutoScalingAlgorithm(N=self.maximum_concurrency, k_delta=k_delta, k_gamma=k_gamma,opt_init=opt_init,tau=tau, K=K, T=self.max_time, log_dir=self.log_dir)
        self.state = self.autoscaler.get_state()
        self.opt = opt_init
        self.opt_step = opt_init

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
        self.total_finished = 0
        self.last_total_finished = 0
        self.job_rejected = False
        self.missed_update = 0
        
        self.pbar = None
        if self.progress and  False:
            self.pbar = tqdm(total=int(self.max_time))

        self.t = 0
        self.pbar_t_update = 0
        self.pbar_interval = int(self.max_time / 100)
        self.next_arrival = self.t + self.req()

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
        #print("STATWE",t,self.state)
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
            if self.idle_count < self.autoscaler.get_opt_step()[1]:
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
        self.start_init_free_servers( t, theta, "Int free arrival")

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

    def get_request_stats_between(self, start_t, end_t):
        total = sum(start_t <= t <= end_t for t in self.total_requests_log)
        served = sum(start_t <= t <= end_t for t in self.served_requests_log)
        return total, served
        
          
    def calculate_cost(self):
        """
        Calculate cost based on current system state
        """
        # Forward the cost calculation to the autoscaler
        
        cost = self.autoscaler.compute_cost(self.state)
        self.job_rejected = False
        return cost

    def process_next_event(self):
        """
        Process the next event in the simulation
        """
        # Record current state in history
        self.hist_times.append(self.t)
        self.update_hist_arrays(self.t)
        
        # If there are no servers, next transition is arrival
        if not self.has_server():
            theta = self.autoscaler.get_opt_step()[0]
            self.t = self.next_arrival
            self.total_requests_log.append(self.t)
            self.next_arrival = self.t + self.req()
            # No servers, so cold start
            self.cold_start_arrival(self.t, theta=theta)
            return
        
        # If there are servers, next transition is the soonest one
        server_next_transitions = np.array([s.get_next_transition_time(self.t) for s in self.servers])
        
        # If next transition is arrival
        if (self.next_arrival - self.t) < server_next_transitions.min():
            self.t = self.next_arrival
            self.next_arrival = self.t + self.req()
            self.total_requests_log.append(self.t)
            
            # If warm start
            if self.is_warm_available(self.t):
                theta = self.autoscaler.get_opt_step()[0]
                self.warm_start_arrival(self.t,theta)
            # If init free available
            elif self.is_init_free_available(self.t):
                theta = self.autoscaler.get_opt_step()[0]
                self.init_free_arrival(self.t, theta)
            # If cold start
            else:
                theta = self.autoscaler.get_opt_step()[0]
                self.cold_start_arrival(self.t, theta=theta)
        # If next transition is a state change in one of servers
        else:
            # Find the server that needs transition
            idx = server_next_transitions.argmin()
            self.t = self.t + server_next_transitions[idx]
            old_state = self.servers[idx].get_state()
            
            new_state = self.servers[idx].make_transition()
            
            # Delete instance if it was just terminated
            if new_state.value == FunctionState.COLD.value:
                self.prev_servers.append(self.servers[idx])
                self.idle_count -= 1
                self.server_count -= 1
                del self.servers[idx]
            
            # If request has done processing (exit event)
            elif new_state.value == FunctionState.IDLE_ON.value:
                self.total_finished += 1
                if old_state.value == FunctionState.BUSY.value:
                    # Transition from running to idle
                    self.running_count -= 1
                elif old_state.value == FunctionState.INIT_FREE.value and not self.servers[idx].is_reserved():
                    # Transition from init_free to idle
                    self.init_free_count -= 1
                    self.servers[idx].unreserve()
                else:
                    # Force this only if we are running current class, not child classes
                    if self.__class__ == ServerlessSimulator:
                        raise Exception(f"Unknown transition in states: {new_state}")
                self.idle_count += 1
            elif new_state.value == FunctionState.BUSY.value:
                self.served_requests_log.append(self.t)
                
                if old_state.value == FunctionState.INIT_RESERVED.value:
                    # Transition from init_reserved to running
                    self.init_reserved_count -= 1
                elif old_state.value == FunctionState.INIT_FREE.value and self.servers[idx].is_reserved():
                    # Transition from init_free to running
                    self.servers[idx].update_next_transition(self.t)
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
                # Force this only if we are running current class, not child classes
                if self.__class__ == ServerlessSimulator:
                    raise Exception(f"Unknown transition in states: {new_state}")


def run_simulator_process(sim_id, shared_params, sim_type, shared_data, exp, theta_lock,barrier):
    """
    Function to run a single simulator process with specified parameters
    """
    seed =shared_params['seed']
    log_dir = shared_params['log_dir']

    random.seed(seed)
    np.random.seed(seed)
    rang_delta_plus = np.random.default_rng(seed)
    rang_delta_minus = np.random.default_rng(seed)
    # Create a fresh simulator with the shared parameters
    if sim_type == f"plus_{sim_id}":
        seed = seed
    else:
        seed = seed

    opt_init=shared_params['opt_init']
    arrival_rate=shared_params['arrival_rate']

    sim:ServerlessSimulator = ServerlessSimulator(
        id=sim_id,
        arrival_rate=shared_params['arrival_rate'], 
        warm_service_rate=shared_params['warm_service_rate'], 
        cold_service_rate=shared_params['cold_service_rate'], 
        cold_start_rate=shared_params['cold_start_rate'],
        max_time=shared_params['max_time'],
        maximum_concurrency=shared_params['maximum_concurrency'],
        log_dir= log_dir,
        seed = seed,
        opt_init= opt_init,
        service_process_type = shared_params['service_process_type'],
        expiration_process_type = shared_params['expiration_process_type'],
        
    )
    max_time=shared_params['max_time']
    maximum_concurrency=shared_params['maximum_concurrency']
    k_delta=shared_params['k_delta']
    k_gamma=shared_params['k_gamma']
    gamma_min = shared_params['gamma_min']
    
    tau=shared_params['tau']
    K=shared_params['K']
    K_exp = shared_params['K_exp']
    
    # Set algorithm parameters
    sim.set_algo_params(
            k_delta=k_delta ,
            k_gamma=k_gamma,
            opt_init=opt_init,
            tau=tau,
            K=K
    )
    
    should_write_csv = False

    
    n = 1
    
    #delta_n = k_delta / (n ** (2.0 / 3.0))
    tau_n = int(tau * (1 + np.log10(n)))
    
    # Disable progress bar for subprocess
    sim.set_progress(False)
    sim.autoscaler.opt_step = shared_data["opt"]

    theta_rows = []
    theta_costs_rows = []
    penalty_gamma = 0
    digit_round = 6
    gamma_round = 6
    theta_round = 6

    all_costs =[]
    all_states = []
    
    # Execute the simulation
    steps = 0
    costs_sum = 0
    t = 0
    simulation_start_time = time.time()

    if sim_type == f"plus_{sim_id}":
            rang_perturb_plus = np.random.default_rng(seed)
    else:
            rang_perturb_minus = np.random.default_rng(seed) 


    while t < max_time:
    # Run the simulation for tau_n steps or until the termination condition
        #n = shared_data[f'n']
        #tau_n = shared_data["tau_n"]
        opt = shared_data["opt"]
        print("OOPT new n",opt)
        
        rang_delta_plus = np.random.default_rng(seed+sim_id*10)
        rang_delta_min_plus = np.random.default_rng(seed+sim_id*10)
        rang_delta_minus = np.random.default_rng(seed+sim_id*10)
        rang_delta_min_minus = np.random.default_rng(seed+sim_id*10)

        running_function_count = 0
        idle_function_count = 0
        init_free_function_count = 0
        init_reserved_function_count = 0
        #sim.initialiaze_system(sim.t, running_function_count, idle_function_count, init_free_function_count, init_reserved_function_count )
        
        if sim_type == "plus":

            rang_delta_plus = np.random.default_rng(seed)

        else:
            rang_delta_minus = np.random.default_rng(seed) 


        if sim_type == f"plus_{sim_id}":
            rng = rang_perturb_plus
        else:
            rng = rang_perturb_minus

        # Build the array of choices
        all_choices = np.array([
            [-1, 1],   # for element 0
            [-1, 1],   # for element 1
            [-1, 1]    # for element 2
        ])

        # Use random indices (0 or 1) for each row
        random_indices = rng.integers(0, 2, size=all_choices.shape[0])

        # Select independently
        perturbation = all_choices[np.arange(all_choices.shape[0]), random_indices]

        delta_n = (k_delta / n ** (2.0 / 3.0))

        perturbations =  perturbation * delta_n
        #probs = distribution * delta_n#* (k_delta / n ** (2.0 / 3.0))

        sim.expiration_process.rate = round(shared_data["opt"][2]/K_exp, digit_round)
        print("Expiration process int", sim.expiration_process.rate)

        while steps < tau_n: #and sim.trace_condition(sim.t):
            # Process the next event

            sim.process_next_event()
            
            # Update state and calculate cost
            sim.update_state()
            cost = sim.calculate_cost()
            costs_sum += cost
            #all_costs.append(cost)
            #all_states.append(sim.state)
            steps += 1
            t += 1
            
            if sim_type == f"plus_{sim_id}":

                theta_delta = opt_init[0]
                theta_min_delta = opt_init[1]
                p_theta = theta_delta - np.floor(theta_delta)
                theta_step = round(min(rang_delta_plus.choice([np.floor(theta_delta), np.floor(theta_delta) + 1], p=[1-p_theta,  p_theta]), maximum_concurrency),theta_round)#,p=[1-probs[0],probs[0]]
                p_theta_min = theta_min_delta - np.floor(theta_min_delta)
                theta_min_step = round(min(rang_delta_min_plus.choice([np.floor(theta_min_delta), np.floor(theta_min_delta) + 1], p=[1-p_theta_min,p_theta_min]), maximum_concurrency),theta_round)#,p=[1-probs[1],probs[1]]
                gamma_exp_step = opt_init[2]
                opt_step = np.array([theta_step,theta_min_step ,gamma_exp_step])
                sim.expiration_process.rate = round(gamma_exp_step/K_exp, digit_round)

                
            else:
                theta_delta = opt_init[0]
                theta_min_delta = opt_init[1]
                p_theta = theta_delta - np.floor(theta_delta)
                p_theta_min = theta_min_delta - np.floor(theta_min_delta)
                theta_step = round(max(rang_delta_minus.choice([np.floor(theta_delta), np.floor(theta_delta) + 1],p=[1-p_theta,  p_theta]), 1),theta_round)#,p=[1-probs[1],probs[1]]
                theta_min_step = round(max(rang_delta_min_minus.choice([np.floor(theta_min_delta), np.floor(theta_min_delta) + 1],p=[1-p_theta_min,p_theta_min]), 0),theta_round )#,p=[1-probs[1],probs[1]]
                gamma_exp_step = opt_init[2]
                opt_step = np.array([theta_step,theta_min_step ,gamma_exp_step])
                sim.expiration_process.rate = round(gamma_exp_step/K_exp, digit_round)

            # set theta for the next_step
            sim.autoscaler.opt_step = opt_step

            if steps == tau_n :
                cost_avg = costs_sum / steps
                sim.missed_update = 0
                with theta_lock:
                    shared_data["ready_count"] += 1
                    shared_data[f'{sim_type}_avg_cost_tau'] = round(cost_avg,3)
                    

                barrier.wait() # Wait for all processes to finish the tau_n steps
                    
                if sim_type == f"plus_0" :
                    with theta_lock:
                        should_write_csv = True
                            
                        data = shared_data.copy()
                        avg_plus = data[f'plus_0_avg_cost_tau'] 
                        avg_minus = data[f'minus_0_avg_cost_tau'] 

                        avg_plus = np.full(len(opt_init),avg_plus)
                        avg_minus = np.full(len(opt_init),avg_minus)

                        opt = np.array([opt_init[0],opt_init[1],opt_init[2]])
                        
                        print(f"Iteration {n} from {sim_type}: Current opt = {opt}, tau_n = {tau_n}, cost_p = {avg_plus}, cost_m = {avg_minus}, delat_n = {delta_n}, perturbations = {perturbation}") # , probs= {} delta_n = {delta_n}, gamma_n = {gamma_n}
                        with open(f"{sim.log_dir}/costs_function_min_{str(arrival_rate).replace('.','')}.txt", "a") as file:
                            file.write(f"{opt_init};{avg_plus[0]};{avg_minus[0]};{(avg_plus[0]+avg_minus[0])/2}\n")
                        n += 1
                        tau_n = int(tau * (1 + np.log10(n)))

                        #Set thetha and thetha for next step
                        sim.autoscaler.opt = opt
                        sim.autoscaler.opt_step = opt
                        shared_data[f'opt'] = opt
                        shared_data["ready_count"] = 0
                        steps = 0
                        costs_sum = 0
                        break
                
                else:
                    while shared_data["ready_count"] != 0:
                        time.sleep(0.01)
                        #print("waiting",sim_type )
                    costs_sum = 0
                    steps=0
                    
                opt = shared_data["opt"]
                
                sim.autoscaler.opt = opt
                sim.autoscaler.opt_step = opt
                n += 1
                steps = 0
                costs_sum = 0
                tau_n = int(tau * (1 + np.log10(n)))

                gamma_n = k_gamma / (n ** 1.0)
                break           
        
        if should_write_csv:
        
            time_before_write = time.time()
            data = shared_data.copy()
            avg_plus = data[f'plus_0_avg_cost_tau']
            avg_minus = data[f'minus_0_avg_cost_tau']
            opt_ = data.get("opt")
            
            
            theta_rows.append({"opt": opt_ })
            theta_costs_rows.append({
                                    "time": time.time(),
                                    "mode": f"{sim_type}",
                                    "theta": opt_[0],
                                    "theta_min":opt_[1],
                                    "gamma":opt_[2],
                                    "avg_plus": avg_plus,
                                    "avg_minus": avg_minus,
                                })

            time_after_write = time.time()
            print(f"Time to write: {time_after_write - time_before_write}")
            should_write_csv = False  
        shared_data[f'{sim_type}_avg_cost_tau'] = 0  
    
    df_theta = pd.DataFrame(theta_rows)
    df_theta_costs = pd.DataFrame(theta_costs_rows)

    df_theta.to_csv(f"{sim.log_dir}/theta_{sim_type}.csv", index=False, header=True)
    df_theta_costs.to_csv(f"{sim.log_dir}/theta_costs_{sim_type}.csv", index=False, header=True)
    
        
    simulation_end_time = time.time()
    total_simulation_time = simulation_end_time - simulation_start_time
    
    max_time=shared_params['max_time']
    maximum_concurrency=shared_params['maximum_concurrency']
    k_delta=shared_params['k_delta']
    k_gamma=shared_params['k_gamma']
    opt_init=shared_params['opt_init']
    tau=shared_params['tau']
    K=shared_params['K']
    
    print(f"[{sim_type}] Process DONE before exiting run_simulator_process, writing in file")
    
    with open(f"{sim.log_dir}/simulation_times.txt", "a") as file:
        file.write(f"{exp};{total_simulation_time};{max_time};{maximum_concurrency};{k_delta};{k_gamma};{opt_init};{tau};{K}\n")
    print("\n=========== SIMULATION SUMMARY ===========")
    print(f"Total simulation wall-clock time: {total_simulation_time:.2f} seconds ({total_simulation_time/60:.2f} minutes)")
    print(f"Final theta value: {shared_data['opt']}")
    
    # Process final results
    print(f"\nResults for simulator {sim_type}:")
    sim.hist_times.append(sim.t)
    sim.calculate_time_lengths()

    # Close progress bar if open
    if sim.progress and sim.pbar:
        sim.pbar.update(int(sim.max_time) - sim.pbar_t_update)
        sim.pbar.close()
    import json
    with open(f"{sim.log_dir}/summary_{sim_type}.txt", "a") as file:
        file.write(f"{json.dumps(sim.get_result_dict())}\n")

    # Print summary results
    sim.print_trace_results()
    #np.savetxt(f"{sim.log_dir}/all_costs_{sim_type}_{'_'.join(str(x) for x in opt_init)}.csv", all_costs, delimiter=",", fmt="%2f")
    #np.savetxt(f"{sim.log_dir}/all_states_{sim_type}_{'_'.join(str(x) for x in opt_init)}.csv", all_states, delimiter=",", fmt="%2f")
    print("======================================================")
        


def parallel_simulation(shared_params, shared_data,exp):
    """
    Runs two simulations in parallel and updates theta based on the results
    """
    
    # Extract parameters into a simple dictionary that can be pickled
    theta_lock = Lock()
    multiprocessing.set_start_method("spawn",force=True)
    
    K = shared_params['K']
    barrier = Barrier(K * 2)
    
    processes = []
    i=0
    
    # Create and start processes
    for k in range(K):
        p_plus = Process(
            target=run_simulator_process, 
            args=(k, shared_params, f"plus_{k}", shared_data, exp[i+k], theta_lock,barrier)
        )
        
        p_minus = Process(
            target=run_simulator_process, 
            args=(k, shared_params, f"minus_{k}", shared_data,exp[i+k+1], theta_lock,barrier)
        )
        i+=1
        processes.append(p_plus)
        processes.append(p_minus)
    
    for p in processes:
        p.start()
    
    for p in processes:
        p.join()
        print ("simulation joined successfully")
    

if __name__ == "__main__":
    
    arrival_rate =5
    warm_service_rate = 1
    cold_service_rate = 1 #not used in our case
    cold_start_rate = 0.1
    service_process_type = "Exponential"  # "Exponential" or "Pareto"
    expiration_process_type = "Exponential"  # "Exponential" or "Deterministic"
    K_exp = 1000
    
    opt_inits =[[3.2,2.7,2.7]]*1
    weights = [100]*10
    seeds = [10,20,30,40,50,60,70,80,90,100]
    
    power_taus = [6]
    gamma_min = 0.1
    scenario = f"arr{arrival_rate}"
    
    i=0
    for opt_init in opt_inits:
        print(opt_init, seeds[i])
        for power_tau in power_taus:    
            K = 1 
            k_delta = 0
            #power_tau = 6
            tau = 4 * 10 ** power_tau
            k_gamma = np.array([3,3, 1.1])#(1* tau ) / 1e6 max(1,opt_init[2]*6/10)]
            max_time = 0.5 * 8 * (10 ** (power_tau))
            current_time = time.strftime("%d_%m_%Y_%H_%M_%S")
            log_dir = f"simulation_cost_{scenario}_{service_process_type}_{expiration_process_type}/experiment_{arrival_rate}"
            maximum_concurrency = 50
            seed = seeds[i]

            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            multiprocessing.set_start_method("spawn",force=True)
            

            
            # Create a manager for shared data
            manager = Manager()
            shared_data = manager.dict()
            shared_data["opt"]=opt_init
            shared_data["ready_count"]=0
            for k in range(K):
                shared_data[f"plus_{k}_avg_cost_tau"] = 0.0
                shared_data[f"minus_{k}_avg_cost_tau"] = 0.0
            plus_done = Event()
            minus_done = Event()
            tau_plus_done = Event()
            tau_minus_done = Event()

            
            shared_params = {
                'arrival_rate': arrival_rate,
                'warm_service_rate': warm_service_rate,
                'cold_service_rate': cold_service_rate,
                'cold_start_rate': cold_start_rate,
                'max_time': max_time,
                'k_delta': k_delta,
                'k_gamma': k_gamma,
                'opt_init': opt_init,
                'tau': tau,
                'K': K,
                'maximum_concurrency': maximum_concurrency,
                'seed' : seed,
                'log_dir':log_dir,
                'K_exp':K_exp,
                'gamma_min':gamma_min,
                'service_process_type': service_process_type,
                'expiration_process_type': expiration_process_type
            }
            
            max_iterations = 1  # Set a maximum number of iterations as a safety
            iteration = 0
            # Run simulations until termination condition
            with open(f"{log_dir}/simulation_times.txt", "a") as file:
                file.write(f"==============================================================================\n")
                file.write(f"==============================================================================\n")
                file.write(f"{max_time};{maximum_concurrency};{k_delta};{k_gamma};{opt_init};{tau};{K}\n")
            
            while iteration < max_iterations:
                iteration += 1
                exps = [iteration]*2*K
                # Run parallel simulation
                parallel_simulation(shared_params, shared_data, exps)
                
            with open(f"{log_dir}/simulation_times.txt", "a") as file:
                file.write(f"==============================================================================\n")
        i+=1