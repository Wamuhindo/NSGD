# The main simulator for serverless computing platforms
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))


from AutoscalerFaasPipelineParallel.SimProcess import ExpSimProcess, ConstSimProcess
from AutoscalerFaasPipelineParallel.FunctionInstance import FunctionInstance
from AutoscalerFaasPipelineParallel.utils import FunctionState, SystemState
from AutoscalerFaasPipelineParallel.Algorithm import AutoScalingAlgorithm

from multiprocessing import Process, Event, Manager, Lock
import multiprocessing

import numpy as np
import time
import pandas as pd
from tqdm import tqdm
import random
import json

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
    def __init__(self,id, config_file,max_time=24*60*60, log_dir ="", **kwargs):
        super().__init__()
        

        config = self.load_config_file(config_file)
        keys = ["maximum_concurrency", "function_config", "theta_init"]
        self.check_and_init_keys(keys, config)

        keys = ["resources","functions"]
        self.check_and_init_keys(keys, self.function_config)

        self.layer_types = list(self.functions.keys())

        
        self.id = id
        self.log_dir = log_dir
        self.max_time = max_time
        self.set_debug(False)
        self.set_progress(True)

        self.reset_trace()
        self._init_processes_and_transition(self.functions)
        

        if None in self.cold_start_processes:
            raise Exception('Cold Start process of one ore more functions not defined!')
        
        if None in self.warm_service_processes:
            raise Exception('Warm Service process of one ore more functions not defined!')
        
        if None in self.expiration_processes:
            raise Exception('Expiration process of one ore more functions not defined!')
        
        if len(self.arrival_processes)==0 :
            raise Exception('The arrival process for the first function not defined!')
        elif self.arrival_processes[0] is None:
            raise Exception('The arrival process of the first function should be defined')
        
        self.t = 0
        self.pbar_t_update = 0
        self.pbar_interval = int(self.max_time / 100)
        self.next_arrival = self.t + self.req()

        self.gamma_exp_min = 0
        self.gamma_exp_max = np.inf
        self.gamma_exp = kwargs.get("gamma_exp",[1]*len(self.layer_types))
    
        
        
    def set_progress(self, progress):
        self.progress = progress
        
    def set_debug(self, debug):
        self.debug=debug
        
    def set_algo_params(self,k_delta, k_gamma, theta_init, tau, K):
        
        self.autoscaler = AutoScalingAlgorithm(N=self.maximum_concurrency, k_delta=k_delta, k_gamma=k_gamma,theta_init=theta_init,tau=tau, K=K, T=self.max_time, log_dir=self.log_dir)
        self.state = self.autoscaler.get_state()
        self.theta = theta_init
        self.theta_step = theta_init

    def reset_trace(self):
        """resets all the historical data to prepare the class for a new simulation
        """
        # an archive of previous servers
        self.prev_servers = []
        self.total_req_count = [0] * len(self.layer_types)
        self.total_cold_count = [0] * len(self.layer_types)
        self.total_init_free_count = [0] * len(self.layer_types)
        self.total_init_reserved_count = [0] * len(self.layer_types)
        self.total_warm_count = [0] * len(self.layer_types)
        self.total_reject_count = [0] * len(self.layer_types)
        # current state of instances
        self.servers:list[FunctionInstance] = []
        self.server_count = 0
        self.running_count = [0] * len(self.layer_types)
        self.idle_count = [0] * len(self.layer_types)
        self.init_free_count = [0] * len(self.layer_types)
        self.init_reserved_count = [0] * len(self.layer_types)
        self.init_free_booked_count = [0] * len(self.layer_types)
        self.reject_count = [0] * len(self.layer_types)
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
        self.queued_jobs_count = [0] * len(self.layer_types)
        self.total_queued_jobs_count = [0] * len(self.layer_types)
        self.total_requests_log = []
        self.served_requests_log = []
        self.last_t = 0
        self.t = 0
        self.total_finished = [0]*len(self.layer_types)
        self.last_total_finished = [0]*len(self.layer_types)
        self.job_rejected = [False] * len(self.layer_types)

        self.cold_start_processes = []
        self.warm_service_processes = []
        self.expiration_processes = []
        self.arrival_processes = []
        self.cold_service_process = None

        self.layers_transitions = []

        
        self.pbar = None
        if self.progress and  False:
            self.pbar = tqdm(total=int(self.max_time))


    
    def check_and_init_keys(self, keys, dictionary):
        for key in keys:
            if key not in dictionary.keys():
                self._error.log(f"The key {key} is missing in the configuration file")
                sys.exit(1)
            setattr(self, key, dictionary[key])

    def _create_process(self, config, gen):
        """Creates a simulation process based on type and parameters."""
        if config is None:
            return None
        process_type = config.get('type')
        if process_type == 'exp':
            rate = 1 / config.get('mean') if 'mean' in config else config.get('rate')
            return ExpSimProcess(rate=rate, rng=gen)
        elif process_type == 'const':
            rate = 1 / config.get('mean') if 'mean' in config else config.get('rate')
            return ConstSimProcess(rate)
        else:
            raise Exception(f"Process type '{process_type}' not defined!")
        
    def _init_processes_and_transition(self, functions_config,seed=None):
        
        if functions_config:
            for function_type in functions_config:
                processes = functions_config[function_type]["processes"]
                transitions = functions_config[function_type]["transitions"]

                self.warm_service_processes.append(self._create_process(processes.get("warm_service_process",None),svc_rng))
                self.cold_start_processes.append(self._create_process(processes.get("cold_start_process",None),cs_rng))
                self.expiration_processes.append(self._create_process(processes.get("expiration_process",None),exp_rng))
                self.arrival_processes.append(self._create_process(processes.get("arrival_process",None),arr_rng))

                self.layers_transitions.append(transitions)
                
    def set_initial_state(self, running_function_instances, idle_function_instances, init_free_function_instances, init_reserved_function_instances):
        
        self.servers = [*running_function_instances, *idle_function_instances, *init_free_function_instances, *init_reserved_function_instances]


    def initialiaze_system(self, t, running_function_count, idle_function_count, init_free_function_count, init_reserved_function_count ):
        np.random.seed(self.seed)
        #print("TT", t)
        
        self.running_count = running_function_count
        self.idle_count = idle_function_count
        self.init_free_count = init_free_function_count
        self.init_reserved_count = init_reserved_function_count
        self.server_count = sum(running_function_count) + sum(idle_function_count) + sum(init_free_function_count) + sum(init_reserved_function_count)

        idle_functions = []
        for func in len(idle_function_count):
            warm_service_process = ExpSimProcess(rate=self.warm_service_processes[func].rate)
            cold_service_process = ExpSimProcess(rate=self.cold_service_process.rate)
            expiration_process = ExpSimProcess(rate=self.expiration_processes[func].rate)
            cold_start_process = ExpSimProcess(rate=self.cold_start_processes[func].rate)
            for _ in range(idle_function_count[func]):
                f =  FunctionInstance(t, 
                                      self.layer_types[func],
                                      cold_service_process=cold_service_process,
                                      warm_service_process=warm_service_process,
                                      expiration_process=expiration_process,
                                      cold_start_process=cold_start_process)

                f.cold_start_process = self.cold_start_processes[func]
                f.warm_service_process = self.warm_service_processes[func]
                f.expiration_process = self.expiration_processes[func]
                f.cold_service_process = self.cold_service_process

                f.state = FunctionState.IDLE_ON
                # when will it be destroyed if no requests
                #f.next_termination = 100
                # so that they would be less likely to be chosen by scheduler
                f.creation_time = 0.01
                idle_functions.append(f)

        running_functions = []
        for func in len(running_function_count):
            warm_service_process = ExpSimProcess(rate=self.warm_service_processes[func].rate)
            cold_service_process = ExpSimProcess(rate=self.cold_service_process.rate)
            expiration_process = ExpSimProcess(rate=self.expiration_processes[func].rate)
            cold_start_process = ExpSimProcess(rate=self.cold_start_processes[func].rate)
            for _ in range(running_function_count[func]):
                f = FunctionInstance(t,
                                    cold_service_process=cold_service_process,
                                    warm_service_process=warm_service_process,
                                    expiration_process=expiration_process,
                                    cold_start_process=cold_start_process
                                    )
                f.cold_start_process = self.cold_start_processes[func]
                f.warm_service_process = self.warm_service_processes[func]
                f.expiration_process = self.expiration_processes[func]
                f.cold_service_process = self.cold_service_process

                f.state = FunctionState.IDLE_ON
                # transition it into running mode
                f.arrival_transition(t)

                running_functions.append(f)
            
        init_free_functions = []

        for func in len(init_free_function_count):   
            warm_service_process = ExpSimProcess(rate=self.warm_service_processes[func].rate)
            cold_service_process = ExpSimProcess(rate=self.cold_service_process.rate)
            expiration_process = ExpSimProcess(rate=self.expiration_processes[func].rate)
            cold_start_process = ExpSimProcess(rate=self.cold_start_processes[func].rate) 
            for _ in range(init_free_function_count[func]):
                
                f = FunctionInstance(t,
                                    cold_service_process=cold_service_process,
                                    warm_service_process=warm_service_process,
                                    expiration_process=expiration_process,
                                    cold_start_process=cold_start_process
                                    )

                f.cold_start_process = self.cold_start_processes[func]
                f.warm_service_process = self.warm_service_processes[func]
                f.expiration_process = self.expiration_processes[func]
                f.cold_service_process = self.cold_service_process

                f.make_Init_Free()


                init_free_functions.append(f)
            
        init_reserved_functions = []   

        for func in len(init_reserved_function_count):
            warm_service_process = ExpSimProcess(rate=self.warm_service_processes[func].rate)
            cold_service_process = ExpSimProcess(rate=self.cold_service_process.rate)
            expiration_process = ExpSimProcess(rate=self.expiration_processes[func].rate)
            cold_start_process = ExpSimProcess(rate=self.cold_start_processes[func].rate)
            for _ in range(init_reserved_function_count):
                f = FunctionInstance(t,
                                    cold_service_process=cold_service_process,
                                    warm_service_process=warm_service_process,
                                    expiration_process=expiration_process,
                                    cold_start_process=cold_start_process
                                    )
                f.cold_start_process = self.cold_start_processes[func]
                f.warm_service_process = self.warm_service_processes[func]
                f.expiration_process = self.expiration_processes[func]
                f.cold_service_process = self.cold_service_process

                f.make_Init_Reserved()

                init_reserved_functions.append(f)
        
        self.set_initial_state(running_functions, idle_functions, init_free_functions, init_reserved_functions)

    def load_config_file(self,filename: str) -> dict:
        """
        Load the configuration file whose name is provided as parameter 
        (if available)
        """
        config = None
        if filename is not None and os.path.exists(filename):
            with open(filename, "r") as istream:
                config = json.load(istream)
        return config

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
        return self.arrival_processes[0].generate_trace()
    
    def current_concurrency(self):
        return (sum(self.running_count) + sum(self.init_reserved_count) + sum(self.init_free_booked_count))
    def current_cold_servers(self):
        return self.maximum_concurrency - (sum(self.running_count) + sum(self.init_reserved_count) + sum(self.init_free_booked_count) + sum(self.init_free_count) + sum(self.idle_count))
    
    def has_reached_max_concurrency(self):
        total_now = self.current_concurrency()
        assert  total_now <= self.maximum_concurrency, f"Concurrency limit exceeded {total_now}, something is wrong"
        return total_now >= self.maximum_concurrency

    def cold_start_arrival(self, t, theta=None, func=0):
        """Goes through the process necessary for a cold start arrival which includes generation of a new function instance in the `COLD` state and adding it to the cluster.

        Parameters
        ----------
        t : float
            The time at which the arrival has happened. This is used to record the creation time for the server and schedule the expiration of the instance if necessary.
        """
        self.total_req_count[func] += 1

        # reject request if maximum concurrency reached
        current_cold_servers = self.current_cold_servers()
        if self.has_reached_max_concurrency() or current_cold_servers<=0:
            self.total_reject_count[func] += 1
            self.job_rejected[func] = True
            self.hist_req_rej_idxs.append(len(self.hist_times) - 1)
            return

        self.total_cold_count[func] += 1
        self.hist_req_cold_idxs.append(len(self.hist_times) - 1)
        
        self.hist_req_init_reserved_idxs.append(len(self.hist_times) - 1)
        self.init_reserved_count[func] += 1
        new_server = FunctionInstance(t, self.layer_types[func], self.cold_service_process, self.warm_service_processes[func], self.expiration_processes[func],self.cold_start_processes[func])
        new_server.make_Init_Reserved()
        self.servers.append(new_server)
        self.server_count += 1
        self.total_init_reserved_count[func] += 1
        
        #self.running_count += 1 : CHECK THIS
        if theta is None:
            return
        self.start_init_free_servers(t,theta,func,"Cold start")

    def schedule_warm_instance(self, t, func=0):
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
        idle_instances = [s for s in self.servers if s.is_idle_on() and s.type==self.layer_types[func]]
        creation_times = [s.creation_time for s in idle_instances]
        
        # scheduling mechanism
        creation_times = np.array(creation_times)
        # find the newest instance
        idx = np.argmax(creation_times)
        return idle_instances[idx]
    
    def schedule_init_free_instance(self, t, func=0):
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
        init_free_instances = [s for s in self.servers if (s.is_init_free() and not s.is_reserved() and s.type==self.layer_types[func])]
        
        creation_times = [s.creation_time for s in init_free_instances]
        
        # scheduling mechanism
        creation_times = np.array(creation_times)
        # find the newest instance
        idx = np.argmax(creation_times)
        return init_free_instances[idx]

    def warm_start_arrival(self, t,theta, func=0):
        """Goes through the process necessary for a warm start arrival which includes selecting a warm instance for processing and recording the request information.

        Parameters
        ----------
        t : float
            The time at which the arrival has happened. This is used to record the creation time for the server and schedule the expiration of the instance if necessary.
        """
        self.total_req_count[func] += 1

        # reject request if maximum concurrency reached
        if self.has_reached_max_concurrency():
            self.total_reject_count[func] += 1
            self.job_rejected[func] = True
            self.hist_req_rej_idxs.append(len(self.hist_times) - 1)
            return

        self.hist_req_warm_idxs.append(len(self.hist_times) - 1)

        # schedule the request
        instance = self.schedule_warm_instance(t,func)
        was_idle = instance.is_idle_on()
        instance.arrival_transition(t)
        
        # transition from idle to running
        self.total_warm_count[func] += 1
        if was_idle:
            #print("Warm arrival theta",func)
            self.idle_count[func] -= 1
            self.running_count[func] += 1
            # a instance has passed from idle to running
            self.served_requests_log.append(t)

            if self.idle_count[func] < self.theta_min[func]:
                self.start_init_free_servers(t, theta, func, "Warm start")
            
    def init_free_arrival(self, t, theta=None, func=0):
        """Goes through the process necessary for a warm start arrival which includes selecting a warm instance for processing and recording the request information.

        Parameters
        ----------
        t : float
            The time at which the arrival has happened. This is used to record the creation time for the server and schedule the expiration of the instance if necessary.
        """
        self.total_req_count[func] += 1
        self.total_queued_jobs_count[func] += 1
        self.queued_jobs_count[func] += 1

        # reject request if maximum concurrency reached
        current_cold_servers = self.current_cold_servers()
        if self.has_reached_max_concurrency():
            self.total_reject_count[func] += 1
            self.job_rejected[func] = True
            self.hist_req_rej_idxs.append(len(self.hist_times) - 1)
            return

        self.hist_req_queued_idxs.append(len(self.hist_times) - 1)

        # schedule the request
        instance = self.schedule_init_free_instance(t,func)
        was_init_free = instance.is_init_free()
        instance.arrival_transition(t)
        self.init_free_count[func] -= 1
        self.init_free_booked_count[func] += 1
        #if instance.get_next_transition_time(t) > instance.cold_start_process.generate_trace()/3:
        self.total_cold_count[func] += 1
        #self.running_count += 1 : CHECK THIS
        if theta is None:
            return
        self.start_init_free_servers(t, theta, func, "Init free start")

    def start_init_free_servers(self, t, theta, func=0, type="Cold start"):

        current_cold_servers = self.current_cold_servers()
        pi_theta = max(0, theta[func] - self.init_free_count[func])
        current_concurrency = self.current_concurrency()
        init_free = self.maximum_concurrency - current_concurrency  if current_concurrency + pi_theta > self.maximum_concurrency else pi_theta
        
        if current_cold_servers > 0:
            init_free = min(current_cold_servers,init_free)
            #print("Init free arrival theta",func) #theta, theta[func],init_free, current_cold_servers
            for _ in range(int(init_free)):
                self.total_init_free_count[func] += 1
                self.hist_req_init_free_idxs.append(len(self.hist_times) - 1)
                self.init_free_count[func] += 1
                self.server_count += 1
                new_server = FunctionInstance(t, self.layer_types[func], self.cold_service_process, self.warm_service_processes[func], self.expiration_processes[func],self.cold_start_processes[func])
                new_server.make_Init_Free()
                self.servers.append(new_server)
        else:
            print(f"No cold servers: {type}")

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
        avg_running_count = (self.hist_server_running_count *  self.time_lengths[:, np.newaxis]).sum(axis=0) / self.get_trace_end()

        return avg_running_count

    def get_average_server_idle_count(self):
        """Get the time-averaged idle server count.

        Returns
        -------
        float
            Average idle server count
        """
        avg_idle_count = (self.hist_server_idle_count * self.time_lengths[:, np.newaxis]).sum(axis=0) / self.get_trace_end()
        return avg_idle_count
    
    def get_average_server_init_free_count(self):
        """Get the time-averaged idle server count.

        Returns
        -------
        float
            Average idle server count
        """
        avg_init_free_count = (self.hist_server_init_free_count * self.time_lengths[:, np.newaxis]).sum(axis=0) / self.get_trace_end()
        return avg_init_free_count
    
    def get_average_server_init_reserved_count(self):
        """Get the time-averaged idle server count.

        Returns
        -------
        float
            Average idle server count
        """
        avg_init_reserved_count = (self.hist_server_init_reserved_count * self.time_lengths[:, np.newaxis]).sum(axis=0) / self.get_trace_end()
        return avg_init_reserved_count
    
    def get_average_server_queued_jobs_count(self):
        """Get the time-averaged idle server count.

        Returns
        -------
        float
            Average idle server count
        """
        avg_init_reserved_count = (self.hist_server_queued_jobs_count * self.time_lengths[:, np.newaxis]).sum(axis=0) / self.get_trace_end()
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
        return life_spans.mean()

    
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

        cold_prob = np.array(self.total_cold_count) / np.array(self.total_req_count)
        formatted = ", ".join([f"{p:.4f}" for p in cold_prob])
    
        print(f"Cold Start Probability: \t [{formatted}]")

        print(f"Rejection / total requests: \t {self.total_reject_count} / {self.total_req_count}")

        reject_prob = np.array(self.total_reject_count) / np.array(self.total_req_count)
        formatted = ", ".join([f"{p:.4f}" for p in reject_prob])
        print(f"Rejection Probability: \t\t [{formatted}]")

        # average instance life span
        life_spans = np.array([s.get_life_span() for s in self.prev_servers])
        if len(life_spans) > 0:
            print(f"Average Instance Life Span: \t {life_spans.mean():.4f}")

        # average instance count
        print(f"Average Server Count:  \t\t {self.get_average_server_count():.4f}")
        # average running count
        formatted = ", ".join([f"{p:.4f}" for p in self.get_average_server_running_count()])
        print(f"Average Running Count:  \t [{formatted}]")
        # average idle count
        print(f"Average Idle Count:  \t\t [{', '.join([f'{p:.4f}' for p in self.get_average_server_idle_count()])}]")
        # average init free count
        print(f"Average Init Free Count:  \t [{', '.join([f'{p:.4f}' for p in self.get_average_server_init_free_count()])}]")
        # average init reserved count
        print(f"Average Init Reserved Count:  \t [{', '.join([f'{p:.4f}' for p in self.get_average_server_init_reserved_count()])}]")
        # average queued jobs count
        print(f"Average Queued Jobs Count:  \t [{', '.join([f'{p:.4f}' for p in self.get_average_server_queued_jobs_count()])}]")
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


    def is_warm_available(self, t, func=0):
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
        return self.idle_count[func] > 0
    
    def is_init_free_available(self, t, func=0):
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
        #return self.init_free_count > 0
        return self.init_free_count[func] > 0


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
        
        
        self.state[SystemState.INITIALIZING.value] = np.array(self.init_free_count) + np.array(self.init_free_booked_count) + np.array(self.init_reserved_count)
        self.state[SystemState.INIT_RESERVED.value] = np.array(self.init_reserved_count) + np.array(self.init_free_booked_count)
        self.state[SystemState.BUSY.value] = np.array(self.running_count)
        self.state[SystemState.IDLE_ON.value] = np.array(self.idle_count)
        self.state[SystemState.COLD.value] = self.maximum_concurrency - (sum(self.init_free_count) + sum(self.init_free_booked_count)+ sum(self.init_reserved_count) + sum(self.running_count) + sum(self.idle_count))
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
        cost = self.autoscaler.compute_cost(self.state)
        self.job_rejected = [False]*len(self.layer_types)
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
            theta = self.autoscaler.get_theta_step()
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
                theta = self.autoscaler.get_theta_step()
                self.warm_start_arrival(self.t,theta)
            # If init free available
            # elif self.is_init_free_available(self.t):
            #     theta = self.autoscaler.get_theta_step()
            #     self.init_free_arrival(self.t, theta)
            # If cold start
            else:
                theta = self.autoscaler.get_theta_step()
                self.cold_start_arrival(self.t, theta=theta)
        # If next transition is a state change in one of servers
        else:
            # Find the server that needs transition
            idx = server_next_transitions.argmin()
            self.t = self.t + server_next_transitions[idx]
            old_state = self.servers[idx].get_state()
            function_type = self.servers[idx].type
            func = self.layer_types.index(function_type)
            
            new_state = self.servers[idx].make_transition()
            
            # Delete instance if it was just terminated
            if new_state.value == FunctionState.COLD.value:
                self.prev_servers.append(self.servers[idx])
                self.idle_count[func] -= 1
                self.server_count -= 1
                del self.servers[idx]
            
            # If request has done processing (exit event)
            elif new_state.value == FunctionState.IDLE_ON.value:
                self.total_finished[func] += 1
                if old_state.value == FunctionState.BUSY.value:
                    # Transition from running to idle
                    self.running_count[func] -= 1

                    # Get the probability distribution
                    probabilities = self.layers_transitions[func]

                    # Generate function choices based on probabilities
                    choices = list(range(len(probabilities)))  # Possible indices: [0, 1, 2, 3, ...]
                    new_func = random.choices(choices, weights=probabilities)[0]

                    #print("NEW FUNC",new_func)
                    
                    if new_func != len(probabilities) - 1:
                        # if warm start
                        if self.is_warm_available(self.t,new_func):
                            theta = self.autoscaler.get_theta_step()
                            self.warm_start_arrival(self.t,theta,new_func)
                        elif self.is_init_free_available(self.t,new_func):
                            theta = self.autoscaler.get_theta_step()
                            self.init_free_arrival(self.t,theta,new_func)
                        # if cold start
                        else:
                            theta = self.autoscaler.get_theta_step()
                            self.cold_start_arrival(self.t,theta,new_func)                    


                elif old_state.value == FunctionState.INIT_FREE.value and not self.servers[idx].is_reserved():
                    # Transition from init_free to idle
                    self.init_free_count[func] -= 1
                    self.servers[idx].unreserve()
                else:
                    # Force this only if we are running current class, not child classes
                    if self.__class__ == ServerlessSimulator:
                        raise Exception(f"Unknown transition in states: {new_state}")
                self.idle_count[func] += 1
            elif new_state.value == FunctionState.BUSY.value:
                self.served_requests_log.append(self.t)
                
                if old_state.value == FunctionState.INIT_RESERVED.value:
                    # Transition from init_reserved to running
                    self.init_reserved_count[func] -= 1
                elif old_state.value == FunctionState.INIT_FREE.value and self.servers[idx].is_reserved():
                    # Transition from init_free to running
                    self.servers[idx].update_next_transition(self.t)
                    self.servers[idx].unreserve()
                    #self.init_free_count -= 1
                    self.init_free_booked_count[func] -= 1
                    self.queued_jobs_count[func] -= 1
                    self.total_warm_count[func] += 1
                    self.hist_req_warm_idxs.append(len(self.hist_times) - 1)
                else:
                    # Force this only if we are running current class, not child classes
                    if self.__class__ == ServerlessSimulator:
                        raise Exception(f"Unknown transition in states: {new_state}")
                self.running_count[func] += 1
            else:
                # Force this only if we are running current class, not child classes
                if self.__class__ == ServerlessSimulator:
                    raise Exception(f"Unknown transition in states: {new_state}")


def run_simulator_process(sim_id, shared_params, sim_type, shared_data, done_event, other_done_event,tau_event, other_tau_event,exp, theta_lock):
    """
    Function to run a single simulator process with specified parameters
    """
    seed =shared_params['seed']
    log_dir = shared_params['log_dir']

    random.seed(seed)
    np.random.seed(seed)
    rang_plus = np.random.RandomState(seed)
    rang_minus = np.random.RandomState(2*seed)
    rang_delta_plus = np.random.RandomState(seed)
    rang_delta_minus = np.random.RandomState(2*seed)
    # Create a fresh simulator with the shared parameters
    sim = ServerlessSimulator(
        id=sim_id,
        config_file=shared_params['config_file'], 
        max_time=shared_params['max_time'],
        log_dir= log_dir,  
        gamma_exp =  shared_params['gamma_exp']
    )
    max_time=shared_params['max_time']
    maximum_concurrency=shared_params['maximum_concurrency']
    k_delta=shared_params['k_delta']
    k_gamma=shared_params['k_gamma']
    theta_init=shared_params['theta_init']
    tau=shared_params['tau']
    K=shared_params['K']
    
    # Set algorithm parameters
    sim.set_algo_params(
            k_delta=k_delta ,
            k_gamma=k_gamma,
            theta_init=theta_init,
            tau=tau,
            K=K
    )
    
    #n = 1
    
    #shared_data[f'{sim_type}_steps'] = 0
    #shared_data[f'{sim_type}_costs'] = 0
    should_write_csv = False
    #shared_data["enter"] = True
    #shared_data[f'n'] = 1
    #shared_data["tau_n"] = tau
    
    n = 1
    
    # Adjust the theta based on simulation type
    #delta_n = k_delta / (n ** (2.0 / 3.0))
    #delta_n = np.random.dirichlet(np.ones(len(theta_init))) * (k_delta / n ** (2.0 / 3.0))
    tau_n = int(tau * (1 + np.log10(n)))
    
    # Disable progress bar for subprocess
    sim.set_progress(False)
    sim.autoscaler.theta_step = shared_data["theta"]
    sim.gamma_exp = shared_data["gamma_exp"]
    
    
    theta_rows = []
    theta_costs_rows = []
    
    # Execute the simulation
    steps = 0
    costs_sum = np.zeros(len(theta_init))
    t = 0
    simulation_start_time = time.time()
    while t < max_time:
    # Run the simulation for tau_n steps or until the termination condition
        #n = shared_data[f'n']
        #tau_n = shared_data["tau_n"]
        theta = shared_data["theta"]
        #print(f"[{sim_type}]starting loop n = {n}")

        # initialize the system
        running_function_count = [1]*len(theta_init)
        idle_function_count = [0]*len(theta_init)
        init_free_function_count = [1]*len(theta_init)
        init_reserved_function_count = [0]*len(theta_init)

        _seed = n+K if sim_type=="plus" else n*100+2*K
        sim._init_processes_and_transition(sim.functions,_seed)
        sim.initialiaze_system(sim.t, running_function_count, idle_function_count, init_free_function_count, init_reserved_function_count)

        if sim_type == "plus":
            #np.random.seed(n+K)
            rang_delta_plus = np.random.default_rng(n+K)
            rang =  np.random.default_rng(n+K)
        else:
            rang_delta_minus = np.random.default_rng(n*100+2*K)
            rang =  np.random.default_rng(n*100+2*K)

        while steps < tau_n or (not (tau_event.is_set() and other_tau_event.is_set())): #and sim.trace_condition(sim.t):
            # Process the next event

            sim.process_next_event()
            
            # Update state and calculate cost
            sim.update_state()
            cost = sim.calculate_cost()
            costs_sum += cost
            steps += 1
            t += 1

            if sim_type == "plus": 
                distribution = rang_delta_plus.dirichlet(np.ones(len(theta_init)))
            else:
                distribution = rang_delta_minus.dirichlet(np.ones(len(theta_init)))
            delta_n = distribution * (k_delta / n ** (2.0 / 3.0))

            with open(f"{sim.log_dir}/delta_{sim_type}.txt", "a") as file:
                file.write(f"{distribution}\n")

            
            if sim_type == "plus":
                #np.random.seed(n + K)
                theta_step = np.where(
                rang.rand(len(theta)) < 0.5,
                np.floor(theta + delta_n),
                np.floor(theta + delta_n) + 1
                )
                theta_step = np.minimum(theta_step, maximum_concurrency)
        
                #theta_delta = theta + delta_n #sim.autoscaler.theta + delta_n
                #theta_step = min(np.random.choice([np.floor(theta_delta), np.floor(theta_delta) + 1]), maximum_concurrency)
            else:
                #np.random.seed(n + 2 * K)
                theta_step = np.where(
                rang.rand(len(theta)) < 0.5,
                np.floor(theta - delta_n),
                np.floor(theta - delta_n) + 1
                )
                theta_step = np.maximum(theta_step, 1)
                #theta_delta = theta - delta_n #sim.autoscaler.theta - delta_n
                #theta_step = max(np.random.choice([np.floor(theta_delta), np.floor(theta_delta) + 1]), 1)

            # set theta for the next_step
            sim.autoscaler.theta_step = theta_step
            
            #print(f"{sim_id}: {sim_type} simulation at step {steps}, tau = {tau_n}")
            
            # Store state at tau_n steps for comparison
            if steps == tau_n-1 :
                cost_avg = costs_sum / steps
                with theta_lock:
                    shared_data[f'{sim_type}_avg_cost_tau'] = cost_avg

                if not other_tau_event.is_set():
                    tau_event.set()
 
                else :
                    if not tau_event.is_set():
                        #print("TAU DONE SET")
                        # tau_done.set()
                        # other_tau_done.wait()
                        # tau_done.clear()
                        # other_tau_done.clear()
                        with theta_lock:
                            should_write_csv = True
                                
                            #shared_data[f'{sim_type}_gap_step'] = steps - steps
                            #shared_data[f'{other_type}_gap_step'] = 0
                            data = shared_data.copy()
                            avg_plus = data[f'plus_avg_cost_tau'] 
                            avg_minus = data[f'minus_avg_cost_tau'] 

                            
                            #delta_n = np.random.dirichlet(np.ones(len(theta_init))) * (k_delta / n ** (2.0 / 3.0))
                            gamma_n = k_gamma / (n ** 1.0)
                            
                            current_theta = data[f'theta']
                            current_gamma_exp = data[f'gamma_exp']

                            gradient = (avg_plus - avg_minus) / (2.0 * delta_n)
                            theta = np.clip(current_theta - gamma_n * gradient, 1.0, maximum_concurrency)

                            gamma_exp = np.clip(current_gamma_exp - gamma_n * gradient, sim.gamma_exp_min, sim.gamma_exp_max)
                            
                            print(f"Iteration {n} from {sim_type}: Current theta = {theta}, current gm = {gamma_exp}, tau_n = {tau_n}, cost_p = {avg_plus}, cost_m = {avg_minus}") #delta_n = {delta_n}, gamma_n = {gamma_n}

                            n += 1
                            tau_n = int(tau * (1 + np.log10(n)))

                            #Set thetha and thetha for next step
                            sim.autoscaler.theta = theta
                            sim.gamma_exp = gamma_exp
                            sim.autoscaler.theta_step = theta
                            shared_data[f'theta'] = theta
                            shared_data[f'gamma_exp'] = gamma_exp
                            
                            steps = 0
                            costs_sum = np.zeros(len(theta_init))
                            
                        tau_event.set()

                        break
                
                other_tau_event.wait()

                th = shared_data["theta"]
                gx = shared_data["gamma_exp"]
                #print(f"{sim_type}, OTHER TAU EVENT SET thetha = {th}, n = {n}, tau_n = {tau_n}")
                #shared_data[f'{sim_type}_gap_step'] = steps - tau_n
                
                sim.autoscaler.theta = th
                sim.autoscaler.theta_step = th
                sim.gamma_exp = gx
                n += 1
                steps = 0
                costs_sum = np.zeros(len(theta_init))
                tau_n = int(tau * (1 + np.log10(n)))
                #delta_n = np.random.dirichlet(np.ones(len(theta_init))) * (k_delta / n ** (2.0 / 3.0))
                gamma_n = k_gamma / (n ** 1.0)
                break                   

        #print(f"LEAVING THE WHILE LOOP {sim_type} {sim_id}", tau_event.is_set(),other_tau_event.is_set(),t)
        tau_event.clear()
        other_tau_event.clear()
        
        if should_write_csv:
        
            time_before_write = time.time()
            data = shared_data.copy()
            avg_plus = data[f'plus_avg_cost_tau']
            avg_minus = data[f'minus_avg_cost_tau']
            gamma_exp = data[f'gamma_exp']
            theta_ = data.get("theta")
            #gap_p = data.get("plus_gap_step")
            #gap_m = data.get("minus_gap_step")
            #shared_data["enter"] = True
            
            
            theta_rows.append({"theta": theta_ })
            theta_costs_rows.append({
                                    "time": time.time(),
                                    "mode": f"{sim_type}",
                                    "theta": theta_,
                                    #"gap_step_p": gap_p,
                                    #"gap_step_m": gap_m,
                                    "avg_plus": avg_plus,
                                    "avg_minus": avg_minus,
                                    "gamma":gamma_exp
                                    # "total_req": shared_data.get("plus_total_req", 0),
                                    # "served_req": shared_data.get("plus_served_req", 0),
                                    # "finished": shared_data.get("plus_finished", 0),
                                    # "resource_usage": shared_data.get("plus_resource_usage", 0),
                                    # "state": shared_data.get("plus_state", 0),
                                })
            
            # with open(f"theta_v.txt", "a") as file:
            #     file.write(f"{theta}\n")
            # with open(f"theta_costs_v.txt", "a") as file:
            #     file.write(f"{'plus'};{theta};{shared_data.get(f'plus_gap_step')};{avg_plus};{avg_minus};{(avg_plus+avg_minus)/2};{shared_data.get('plus_total_req', 0)};{shared_data.get('plus_served_req', 0)};{shared_data.get('plus_finished', 0)};{shared_data.get('plus_resource_usage', 0)};{shared_data.get('plus_state', 0)}\n")
            #     file.write(f"{'minus'};{theta};{shared_data.get(f'minus_gap_step')};{avg_plus};{avg_minus};{(avg_plus+avg_minus)/2};{shared_data.get('minus_total_req', 0)};{shared_data.get('minus_served_req', 0)};{shared_data.get('minus_finished', 0)};{shared_data.get('minus_resource_usage', 0)};{shared_data.get('minus_state', 0)}\n")
            time_after_write = time.time()
            print(f"Time to write: {time_after_write - time_before_write}")
            should_write_csv = False  
        shared_data[f'{sim_type}_avg_cost_tau'] = 0  
    
    done_event.set()
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
    theta_init=shared_params['theta_init']
    tau=shared_params['tau']
    K=shared_params['K']
    
    print(f"[{sim_type}] Process DONE before exiting run_simulator_process, writing in file")
    
    with open(f"{sim.log_dir}/simulation_times.txt", "a") as file:
        file.write(f"{exp};{total_simulation_time};{max_time};{maximum_concurrency};{k_delta};{k_gamma};{theta_init};{tau};{K}\n")
    print("\n=========== SIMULATION SUMMARY ===========")
    print(f"Total simulation wall-clock time: {total_simulation_time:.2f} seconds ({total_simulation_time/60:.2f} minutes)")
    print(f"Final theta value: {shared_data['theta']}")
    
    # Process final results
    print(f"\nResults for simulator {sim_type}:")
    sim.hist_times.append(sim.t)
    sim.calculate_time_lengths()

    # Close progress bar if open
    if sim.progress and sim.pbar:
        sim.pbar.update(int(sim.max_time) - sim.pbar_t_update)
        sim.pbar.close()

    # Print summary results
    sim.print_trace_results()
    print("======================================================")
        


def parallel_simulation(shared_params, shared_data, plus_done, minus_done, tau_plus_done, tau_minus_done,exp):
    """
    Runs two simulations in parallel and updates theta based on the results
    """
    
    # Extract parameters into a simple dictionary that can be pickled
    theta_lock = Lock()
    multiprocessing.set_start_method("spawn",force=True)
    
    # Create and start processes
    p_plus = Process(
        target=run_simulator_process, 
        args=(0, shared_params, "plus", shared_data, plus_done, minus_done, tau_plus_done, tau_minus_done,exp[0], theta_lock)
    )
    
    p_minus = Process(
        target=run_simulator_process, 
        args=(1, shared_params, "minus", shared_data, minus_done, plus_done,tau_minus_done, tau_plus_done,exp[1], theta_lock)
    )
    
    p_plus.start()
    p_minus.start()
    p_plus.join()
    print("plus simulation joined successfully")
    p_minus.join()
    print("Both simulations joined successfully")
    

if __name__ == "__main__":
    

    config_file = "AutoscalerFaasVectorielPar/config.json"

    def load_config_file(filename: str) -> dict:
        """
        Load the configuration file whose name is provided as parameter 
        (if available)
        """
        config = None
        if filename is not None and os.path.exists(filename):
            with open(filename, "r") as istream:
                config = json.load(istream)
        return config
    

    configs = load_config_file(config_file)
    maximum_concurrency = configs.get("maximum_concurrency",50)
    theta_init = configs.get("theta_init",[2,2,2])


    K = 1 
    k_delta = 1
    power_tau = 4
    tau = 10 ** power_tau
    k_gamma = (1 * tau ) / 1e6
    max_time = 0.5 * 1 * (10 ** (power_tau + 2))
    current_time = time.strftime("%d_%m_%Y_%H_%M_%S")
    log_dir = f"zzvectoriel_Par/zexperiment_PAR_{theta_init}_{power_tau}_{current_time}"
    
    seed = 1234
    np.random.seed(seed)
    random.seed(seed)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    multiprocessing.set_start_method("spawn",force=True)
    

    
    # Create a manager for shared data
    manager = Manager()
    shared_data = manager.dict()
    shared_data["theta"]=theta_init
    shared_data["plus_avg_cost_tau"] = np.zeros(len(theta_init))
    shared_data["minus_avg_cost_tau"] = np.zeros(len(theta_init))
    plus_done = Event()
    minus_done = Event()
    tau_plus_done = Event()
    tau_minus_done = Event()

    
    shared_params = {
        'max_time': max_time,
        'k_delta': k_delta,
        'k_gamma': k_gamma,
        'theta_init': theta_init,
        'tau': tau,
        'K': K,
        'maximum_concurrency': maximum_concurrency,
        'seed' : seed,
        'log_dir':log_dir,
        'config_file':config_file
    }
    
    max_iterations = 1  # Set a maximum number of iterations as a safety
    iteration = 0
    # Run simulations until termination condition
    with open(f"{log_dir}/simulation_times.txt", "a") as file:
        file.write(f"==============================================================================\n")
        file.write(f"==============================================================================\n")
        file.write(f"{max_time};{maximum_concurrency};{k_delta};{k_gamma};{theta_init};{tau};{K}\n")
    while iteration < max_iterations:
        iteration += 1
        exps = [iteration]*2
        # Run parallel simulation
        parallel_simulation(shared_params, shared_data, plus_done, minus_done, tau_plus_done, tau_minus_done,exps)
        
    with open(f"{log_dir}/simulation_times.txt", "a") as file:
        file.write(f"==============================================================================\n")
    