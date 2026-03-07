"""
Parallel scalar NSKW serverless simulator.

Runs 2*K simulator processes in parallel:
  - K "plus" processes: simulate with theta + delta_n
  - K "minus" processes: simulate with theta - delta_n
Synchronize at tau_n boundaries using a Barrier.
The "plus_0" process computes the gradient update.

Supports SGD, Adam, RMSProp optimizers.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from AutoscalerFaasScalarPar.SimProcess import ExpSimProcess, ConstSimProcess, ParetoSimProcess
from AutoscalerFaasScalarPar.FunctionInstance import FunctionInstance
from AutoscalerFaasScalarPar.utils import FunctionState, SystemState
from AutoscalerFaasScalarPar.Algorithm import ScalarAutoScalingAlgorithm

from multiprocessing import Process, Manager, Lock, Barrier
import multiprocessing

import numpy as np
from numpy.random import default_rng, SeedSequence
import time
import pandas as pd
from tqdm import tqdm
import random


class ServerlessSimulator:
    def __init__(self, id, arrival_process=None, warm_service_process=None,
                 cold_service_process=None, cold_start_process=None,
                 service_process_type="Exponential", expiration_process_type="Exponential",
                 maximum_concurrency=50, log_dir="", **kwargs):
        super().__init__()
        self.seed = kwargs.get("seed", 1)
        self.id = id

        ss = SeedSequence(self.seed)
        cs_rng, svc_rng, exp_rng, arr_rng = [default_rng(s) for s in ss.spawn(4)]

        # Setup arrival process
        self.arrival_process = arrival_process
        if 'arrival_rate' in kwargs:
            self.arrival_process = ExpSimProcess(rate=kwargs.get('arrival_rate'), gen=arr_rng)
        if self.arrival_process is None:
            raise Exception('Arrival process not defined!')

        # Setup warm service process
        self.warm_service_process = warm_service_process
        if 'warm_service_rate' in kwargs:
            if service_process_type == "Pareto":
                shape = 1 + np.sqrt(1.25)
                scale = (np.sqrt(1.25) / (1 + np.sqrt(1.25)))
                self.warm_service_process = ParetoSimProcess(scale=scale, shape=shape, gen=svc_rng)
            else:
                self.warm_service_process = ExpSimProcess(rate=kwargs.get('warm_service_rate'), gen=svc_rng)
        if self.warm_service_process is None:
            raise Exception('Warm Service process not defined!')

        # Setup cold service process
        self.cold_service_process = cold_service_process
        if 'cold_service_rate' in kwargs:
            self.cold_service_process = ExpSimProcess(rate=kwargs.get('cold_service_rate'))
        if self.cold_service_process is None:
            raise Exception('Cold Service process not defined!')

        # Setup cold start process
        if 'cold_start_rate' in kwargs:
            self.cold_start_process = ExpSimProcess(rate=kwargs.get('cold_start_rate'), gen=cs_rng)
        if self.cold_start_process is None:
            raise Exception('Cold Start process not defined!')

        # Setup expiration process (fixed rate, not optimized in scalar version)
        expiration_rate = kwargs.get('expiration_rate', 0.01)
        if expiration_process_type == "Deterministic":
            self.expiration_process = ConstSimProcess(expiration_rate)
        else:
            self.expiration_process = ExpSimProcess(expiration_rate, gen=exp_rng)

        self.maximum_concurrency = maximum_concurrency
        self.set_debug(False)
        self.set_progress(True)
        self.max_time = kwargs.get('max_time', 1e6)
        self.log_dir = log_dir
        self.all_states = []
        self.reset_trace()

    def set_progress(self, progress):
        self.progress = progress

    def set_debug(self, debug):
        self.debug = debug

    def set_algo_params(self, k_delta, k_gamma, theta_init, tau, K):
        self.autoscaler = ScalarAutoScalingAlgorithm(
            N=self.maximum_concurrency, k_delta=k_delta, k_gamma=k_gamma,
            theta_init=theta_init, tau=tau, K=K, T=self.max_time, log_dir=self.log_dir
        )
        self.state = self.autoscaler.get_state()
        self.theta_init = theta_init
        self.theta_step = theta_init

    def reset_trace(self):
        self.prev_servers = []
        self.total_req_count = 0
        self.total_cold_count = 0
        self.total_init_free_count = 0
        self.total_init_reserved_count = 0
        self.total_warm_count = 0
        self.total_reject_count = 0
        self.servers = []
        self.server_count = 0
        self.running_count = 0
        self.idle_count = 0
        self.init_free_count = 0
        self.init_reserved_count = 0
        self.init_free_booked_count = 0
        self.reject_count = 0
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
        self.t = 0
        self.pbar_t_update = 0
        self.pbar_interval = int(self.max_time / 100)
        self.next_arrival = self.t + self.req()

    def set_initial_state(self, running_function_instances, idle_function_instances,
                          init_free_function_instances, init_reserved_function_instances):
        init_running_count = len(running_function_instances)
        init_idle_count = len(idle_function_instances)
        init_free_count = len(init_free_function_instances)
        init_reserved_count = len(init_reserved_function_instances)

        self.server_count = init_running_count + init_idle_count + init_free_count + init_reserved_count
        self.running_count = init_running_count
        self.init_free_count = init_free_count
        self.init_reserved_count = init_reserved_count
        self.idle_count = self.server_count - (init_running_count + init_free_count + init_reserved_count)
        self.servers = [*running_function_instances, *idle_function_instances,
                        *init_free_function_instances, *init_reserved_function_instances]

        ss = SeedSequence(self.seed)
        cs_rng, svc_rng, exp_rng, arr_rng = [default_rng(s) for s in ss.spawn(4)]
        self.arrival_process.rangen = arr_rng
        self.warm_service_process.rangen = svc_rng
        self.cold_start_process.rangen = cs_rng
        self.expiration_process.rangen = exp_rng

    def initialiaze_system(self, t, running_function_count, idle_function_count,
                           init_free_function_count, init_reserved_function_count):
        np.random.seed(self.seed)
        idle_functions = []
        for _ in range(idle_function_count):
            f = FunctionInstance(t, self.cold_service_process, self.warm_service_process,
                                self.expiration_process, self.cold_start_process)
            f.cold_start_process = self.cold_start_process
            f.warm_service_process = self.warm_service_process
            f.expiration_process = self.expiration_process
            f.cold_service_process = self.cold_service_process
            f.state = FunctionState.IDLE_ON
            f.creation_time = 0.01
            idle_functions.append(f)

        running_functions = []
        for _ in range(running_function_count):
            f = FunctionInstance(t, self.cold_service_process, self.warm_service_process,
                                self.expiration_process, self.cold_start_process)
            f.cold_start_process = self.cold_start_process
            f.warm_service_process = self.warm_service_process
            f.expiration_process = self.expiration_process
            f.cold_service_process = self.cold_service_process
            f.state = FunctionState.IDLE_ON
            f.arrival_transition(t)
            running_functions.append(f)

        init_free_functions = []
        for _ in range(init_free_function_count):
            f = FunctionInstance(t, self.cold_service_process, self.warm_service_process,
                                self.expiration_process, self.cold_start_process)
            f.cold_start_process = self.cold_start_process
            f.warm_service_process = self.warm_service_process
            f.expiration_process = self.expiration_process
            f.cold_service_process = self.cold_service_process
            f.make_Init_Free()
            init_free_functions.append(f)

        init_reserved_functions = []
        for _ in range(init_reserved_function_count):
            f = FunctionInstance(t, self.cold_service_process, self.warm_service_process,
                                self.expiration_process, self.cold_start_process)
            f.cold_start_process = self.cold_start_process
            f.warm_service_process = self.warm_service_process
            f.expiration_process = self.expiration_process
            f.cold_service_process = self.cold_service_process
            f.make_Init_Reserved()
            init_reserved_functions.append(f)

        self.set_initial_state(running_functions, idle_functions,
                               init_free_functions, init_reserved_functions)

    def has_server(self):
        return len(self.servers) > 0

    def __str__(self):
        return f"idle/running/total: \t {self.idle_count}/{self.running_count}/{self.server_count}"

    def req(self):
        return self.arrival_process.generate_trace()

    def current_concurrency(self):
        return self.running_count + self.init_reserved_count + self.init_free_booked_count

    def current_cold_servers(self):
        return self.maximum_concurrency - (self.running_count + self.init_reserved_count +
                                           self.init_free_booked_count + self.init_free_count + self.idle_count)

    def has_reached_max_concurrency(self):
        total_now = self.current_concurrency()
        assert total_now <= self.maximum_concurrency, f"Concurrency limit exceeded {total_now}"
        return total_now >= self.maximum_concurrency

    def cold_start_arrival(self, t, theta=1):
        self.total_req_count += 1
        current_cold_servers = self.current_cold_servers()
        if self.has_reached_max_concurrency() or current_cold_servers <= 0:
            self.total_reject_count += 1
            self.job_rejected = True
            self.hist_req_rej_idxs.append(len(self.hist_times) - 1)
            return

        self.total_cold_count += 1
        self.hist_req_cold_idxs.append(len(self.hist_times) - 1)
        self.hist_req_init_reserved_idxs.append(len(self.hist_times) - 1)
        self.init_reserved_count += 1
        new_server = FunctionInstance(t, self.cold_service_process, self.warm_service_process,
                                     self.expiration_process, self.cold_start_process)
        new_server.make_Init_Reserved()
        self.servers.append(new_server)
        self.server_count += 1
        self.total_init_reserved_count += 1
        self.start_init_free_servers(t, theta)

    def start_init_free_servers(self, t, theta):
        if theta < 0:
            return
        current_cold_servers = self.current_cold_servers()
        pi_theta = max(0, theta - self.init_free_count)
        current_concurrency = self.current_concurrency()
        init_free = self.maximum_concurrency - current_concurrency if current_concurrency + pi_theta > self.maximum_concurrency else pi_theta

        if current_cold_servers > 0:
            init_free = min(current_cold_servers, init_free)
            for _ in range(int(init_free)):
                self.total_init_free_count += 1
                self.hist_req_init_free_idxs.append(len(self.hist_times) - 1)
                self.init_free_count += 1
                self.server_count += 1
                new_server = FunctionInstance(t, self.cold_service_process, self.warm_service_process,
                                             self.expiration_process, self.cold_start_process)
                new_server.make_Init_Free()
                self.servers.append(new_server)

    def schedule_warm_instance(self, t):
        idle_instances = [s for s in self.servers if s.is_idle_on()]
        creation_times = np.array([s.creation_time for s in idle_instances])
        idx = np.argmax(creation_times)
        return idle_instances[idx]

    def schedule_init_free_instance(self, t):
        init_free_instances = [s for s in self.servers if (s.is_init_free() and not s.is_reserved())]
        creation_times = np.array([s.creation_time for s in init_free_instances])
        idx = np.argmax(creation_times)
        return init_free_instances[idx]

    def warm_start_arrival(self, t, theta):
        self.total_req_count += 1
        if self.has_reached_max_concurrency():
            self.total_reject_count += 1
            self.job_rejected = True
            self.hist_req_rej_idxs.append(len(self.hist_times) - 1)
            return

        self.hist_req_warm_idxs.append(len(self.hist_times) - 1)
        instance = self.schedule_warm_instance(t)
        was_idle = instance.is_idle_on()
        instance.arrival_transition(t)
        self.total_warm_count += 1
        if was_idle:
            self.idle_count -= 1
            self.running_count += 1
            self.served_requests_log.append(t)
            # Scalar version: scale up when idle_count == 0
            if self.idle_count == 0:
                self.start_init_free_servers(t, theta)

    def init_free_arrival(self, t, theta):
        self.total_req_count += 1
        self.total_queued_jobs_count += 1
        self.queued_jobs_count += 1
        if self.has_reached_max_concurrency():
            self.total_reject_count += 1
            self.job_rejected = True
            self.hist_req_rej_idxs.append(len(self.hist_times) - 1)
            return

        self.hist_req_queued_idxs.append(len(self.hist_times) - 1)
        instance = self.schedule_init_free_instance(t)
        instance.arrival_transition(t)
        self.init_free_count -= 1
        self.init_free_booked_count += 1
        self.total_cold_count += 1
        self.start_init_free_servers(t, theta)

    def is_warm_available(self, t):
        return self.idle_count > 0

    def is_init_free_available(self, t):
        return self.init_free_count > 0

    def update_hist_arrays(self, t):
        self.hist_server_count.append(self.server_count)
        self.hist_server_running_count.append(self.running_count)
        self.hist_server_idle_count.append(self.idle_count)
        self.hist_server_init_reserved_count.append(self.init_reserved_count)
        self.hist_server_init_free_count.append(self.init_free_count)
        self.hist_server_queued_jobs_count.append(self.queued_jobs_count)

    def update_state(self):
        self.state[SystemState.INITIALIZING.value] = self.init_free_count + self.init_free_booked_count + self.init_reserved_count
        self.state[SystemState.INIT_RESERVED.value] = self.init_reserved_count + self.init_free_booked_count
        self.state[SystemState.BUSY.value] = self.running_count
        self.state[SystemState.IDLE_ON.value] = self.idle_count
        self.state[SystemState.COLD.value] = self.maximum_concurrency - (
            self.init_free_count + self.init_free_booked_count +
            self.init_reserved_count + self.running_count + self.idle_count)
        self.autoscaler.set_has_rejected_job(self.job_rejected)

    def get_request_stats_between(self, start_t, end_t):
        total = sum(start_t <= t <= end_t for t in self.total_requests_log)
        served = sum(start_t <= t <= end_t for t in self.served_requests_log)
        return total, served

    def calculate_time_lengths(self):
        self.time_lengths = np.diff(self.hist_times)

    def get_trace_end(self):
        return self.hist_times[-1]

    def get_average_server_count(self):
        return (self.hist_server_count * self.time_lengths).sum() / self.get_trace_end()

    def get_average_server_running_count(self):
        return (self.hist_server_running_count * self.time_lengths).sum() / self.get_trace_end()

    def get_average_server_idle_count(self):
        return (self.hist_server_idle_count * self.time_lengths).sum() / self.get_trace_end()

    def get_average_server_init_free_count(self):
        return (self.hist_server_init_free_count * self.time_lengths).sum() / self.get_trace_end()

    def get_average_server_init_reserved_count(self):
        return (self.hist_server_init_reserved_count * self.time_lengths).sum() / self.get_trace_end()

    def get_average_server_queued_jobs_count(self):
        return (self.hist_server_queued_jobs_count * self.time_lengths).sum() / self.get_trace_end()

    def get_average_resource_usage(self):
        resource_usage = [s.get_resource_usages() for s in self.servers]
        if resource_usage:
            return np.mean(resource_usage, axis=0)
        return np.zeros(4)

    def get_index_after_time(self, t):
        return np.min(np.where(np.array(self.hist_times) > t))

    def get_skip_init(self, skip_init_time=None, skip_init_index=None):
        skip_init = 0
        if skip_init_time is not None:
            skip_init = self.get_index_after_time(skip_init_time)
        if skip_init_index is not None:
            skip_init = max(skip_init, skip_init_index)
        return skip_init

    def get_request_custom_states(self, hist_states, skip_init_time=None, skip_init_index=None):
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
        reqdf = pd.DataFrame(data={
            'state': states, 'cold': list(state_req_colds.values()),
            'warm': list(state_req_warm.values()), 'rej': list(state_req_rejs.values()),
            'init_free': list(state_req_init_free.values()),
            'init_reserved': list(state_req_init_reserved.values()),
            'queued': list(state_req_queued.values())
        })
        reqdf['total'] = reqdf['cold'] + reqdf['warm'] + reqdf['rej']
        reqdf['p_cold'] = reqdf['cold'] / reqdf['total']
        return reqdf

    def analyze_custom_states(self, hist_states, skip_init_time=None, skip_init_index=None):
        skip_init = self.get_skip_init(skip_init_time=skip_init_time,
                                        skip_init_index=skip_init_index)
        values = hist_states[skip_init:]
        time_lengths = self.time_lengths[skip_init:]
        residence_times = {}
        transition_times = {}
        curr_time_sum = time_lengths[0]
        for idx in range(1, len(values)):
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
        residence_times, _ = self.analyze_custom_states(hist_states, skip_init_time, skip_init_index)
        return {s: np.mean(residence_times[s]) for s in residence_times}

    def get_cold_start_prob(self):
        return self.total_cold_count / self.total_req_count

    def get_average_lifespan(self):
        life_spans = np.array([s.get_life_span() for s in self.prev_servers])
        return life_spans.mean() if len(life_spans) else np.inf

    def get_result_dict(self):
        return {
            "reqs_cold": self.total_cold_count,
            "reqs_total": self.total_req_count,
            "reqs_init_free": self.total_init_free_count,
            "reqs_init_reserved": self.total_init_reserved_count,
            "reqs_warm": self.total_warm_count,
            "reqs_queued": self.total_queued_jobs_count,
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
        self.calculate_time_lengths()
        print(f"Cold Starts / total requests: \t {self.total_cold_count} / {self.total_req_count}")
        print(f"Cold Start Probability: \t {self.total_cold_count / self.total_req_count:.4f}")
        print(f"Rejection / total requests: \t {self.total_reject_count} / {self.total_req_count}")
        print(f"Rejection Probability: \t\t {self.total_reject_count / self.total_req_count:.4f}")
        life_spans = np.array([s.get_life_span() for s in self.prev_servers])
        if len(life_spans) > 0:
            print(f"Average Instance Life Span: \t {life_spans.mean():.4f}")
        print(f"Average Server Count:  \t\t {self.get_average_server_count():.4f}")
        print(f"Average Running Count:  \t {self.get_average_server_running_count():.4f}")
        print(f"Average Idle Count:  \t\t {self.get_average_server_idle_count():.4f}")
        print(f"Average Init Free Count:  \t {self.get_average_server_init_free_count():.4f}")
        print(f"Average Init Reserved Count:  \t {self.get_average_server_init_reserved_count():.4f}")
        print(f"Average Queued Jobs Count:  \t {self.get_average_server_queued_jobs_count():.4f}")

    @staticmethod
    def print_time_average(vals, probs, column_width=15):
        print(f"{'Value'.ljust(column_width)} Prob")
        print("".join(["="] * int(column_width * 1.5)))
        for val, prob in zip(vals, probs):
            print(f"{str(val).ljust(column_width)} {prob:.4f}")

    def calculate_time_average(self, values, skip_init_time=None, skip_init_index=None):
        assert len(values) == len(self.time_lengths), "Values should be same length as history array"
        skip_init = self.get_skip_init(skip_init_time=skip_init_time,
                                        skip_init_index=skip_init_index)
        values = values[skip_init:]
        time_lengths = self.time_lengths[skip_init:]
        unq_vals = list(set(values))
        val_times = []
        for val in unq_vals:
            t = time_lengths[[v == val for v in values]].sum()
            val_times.append(t)
        val_times = np.array(val_times)
        val_times = val_times / val_times.sum()
        return unq_vals, val_times

    def calculate_cost(self):
        cost = self.autoscaler.compute_cost(self.state)
        self.job_rejected = False
        return cost

    def process_next_event(self):
        """Process the next event in the simulation (called externally by the parallel loop)."""
        self.hist_times.append(self.t)
        self.update_hist_arrays(self.t)

        if not self.has_server():
            theta = self.autoscaler.get_theta_step()
            self.t = self.next_arrival
            self.total_requests_log.append(self.t)
            self.next_arrival = self.t + self.req()
            self.cold_start_arrival(self.t, theta=theta)
            return

        server_next_transitions = np.array([s.get_next_transition_time(self.t) for s in self.servers])

        if (self.next_arrival - self.t) < server_next_transitions.min():
            self.t = self.next_arrival
            self.next_arrival = self.t + self.req()
            self.total_requests_log.append(self.t)

            theta = self.autoscaler.get_theta_step()
            if self.is_warm_available(self.t):
                self.warm_start_arrival(self.t, theta)
            elif self.is_init_free_available(self.t):
                self.init_free_arrival(self.t, theta)
            else:
                self.cold_start_arrival(self.t, theta=theta)
        else:
            idx = server_next_transitions.argmin()
            self.t = self.t + server_next_transitions[idx]
            old_state = self.servers[idx].get_state()
            new_state = self.servers[idx].make_transition()

            if new_state.value == FunctionState.COLD.value:
                self.prev_servers.append(self.servers[idx])
                self.idle_count -= 1
                self.server_count -= 1
                del self.servers[idx]

            elif new_state.value == FunctionState.IDLE_ON.value:
                self.total_finished += 1
                if old_state.value == FunctionState.BUSY.value:
                    self.running_count -= 1
                elif old_state.value == FunctionState.INIT_FREE.value and not self.servers[idx].is_reserved():
                    self.init_free_count -= 1
                    self.servers[idx].unreserve()
                else:
                    if self.__class__ == ServerlessSimulator:
                        raise Exception(f"Unknown transition in states: {new_state}")
                self.idle_count += 1

            elif new_state.value == FunctionState.BUSY.value:
                self.served_requests_log.append(self.t)
                if old_state.value == FunctionState.INIT_RESERVED.value:
                    self.init_reserved_count -= 1
                elif old_state.value == FunctionState.INIT_FREE.value and self.servers[idx].is_reserved():
                    self.servers[idx].update_next_transition(self.t)
                    self.servers[idx].unreserve()
                    self.init_free_booked_count -= 1
                    self.queued_jobs_count -= 1
                    self.total_warm_count += 1
                    self.hist_req_warm_idxs.append(len(self.hist_times) - 1)
                else:
                    if self.__class__ == ServerlessSimulator:
                        raise Exception(f"Unknown transition in states: {new_state}")
                self.running_count += 1
            else:
                if self.__class__ == ServerlessSimulator:
                    raise Exception(f"Unknown transition in states: {new_state}")


def run_simulator_process(sim_id, shared_params, sim_type, shared_data, exp, theta_lock, barrier):
    """Run a single simulator process with scalar SPSA perturbation.

    Each process runs its own simulator with theta +/- delta_n.
    Uses ScalarAutoScalingAlgorithm methods for:
      - stochastic rounding (algo.stochastic_round)
      - SPSA sequences (algo.get_delta_n, algo.get_tau_n)
      - gradient estimation (algo.compute_gradient)
      - optimizer updates (algo.apply_optimizer)
    """
    seed = shared_params['seed']
    log_dir = shared_params['log_dir']

    random.seed(seed)
    np.random.seed(seed)

    theta_init = shared_params['theta_init']
    maximum_concurrency = shared_params['maximum_concurrency']
    max_time = shared_params['max_time']
    k_delta = shared_params['k_delta']
    k_gamma = shared_params['k_gamma']
    tau = shared_params['tau']
    K = shared_params['K']
    optimization = shared_params['optimization']

    sim = ServerlessSimulator(
        id=sim_id,
        arrival_rate=shared_params['arrival_rate'],
        warm_service_rate=shared_params['warm_service_rate'],
        cold_service_rate=shared_params['cold_service_rate'],
        cold_start_rate=shared_params['cold_start_rate'],
        max_time=max_time,
        maximum_concurrency=maximum_concurrency,
        log_dir=log_dir,
        seed=seed,
        expiration_rate=shared_params['expiration_rate'],
        service_process_type=shared_params['service_process_type'],
        expiration_process_type=shared_params['expiration_process_type'],
    )

    sim.set_algo_params(
        k_delta=k_delta, k_gamma=k_gamma,
        theta_init=theta_init, tau=tau, K=K
    )

    algo = sim.autoscaler
    sim.set_progress(False)
    algo.theta_step = shared_data["theta"]

    # Initialize stochastic rounding RNG
    rang_delta = np.random.default_rng(seed + sim_id * 10)

    # Leader process (plus_0) creates its own algo instance for gradient/optimizer state
    leader_algo = None
    if sim_type == "plus_0":
        leader_algo = ScalarAutoScalingAlgorithm(
            N=maximum_concurrency, k_delta=k_delta, k_gamma=k_gamma,
            theta_init=theta_init, tau=tau, K=K, T=max_time, log_dir=log_dir
        )

    theta_rows = []
    theta_costs_rows = []

    n = 1
    tau_n = algo.get_tau_n(n)
    steps = 0
    costs_sum = 0
    t = 0
    simulation_start_time = time.time()

    while t < max_time:
        rang_delta = np.random.default_rng(seed + sim_id * 10)

        delta_n = algo.get_delta_n(n)

        with theta_lock:
            theta = shared_data["theta"]

        # Apply perturbation: plus processes add, minus processes subtract
        if sim_type == f"plus_{sim_id}":
            theta_perturbed = theta + delta_n
        else:
            theta_perturbed = theta - delta_n

        while steps < tau_n:
            sim.process_next_event()
            sim.update_state()
            cost = sim.calculate_cost()
            costs_sum += cost
            steps += 1
            t += 1

            # Stochastic rounding via Algorithm method
            theta_step = algo.get_perturbed_theta_step(theta_perturbed, rang_delta)
            algo.theta_step = theta_step

            if steps == tau_n:
                cost_avg = costs_sum / steps

                with theta_lock:
                    shared_data["ready_count"] += 1
                    shared_data[f'{sim_type}_avg_cost_tau'] = round(cost_avg, 3)

                barrier.wait()

                if sim_type == "plus_0":
                    # Leader process: compute gradient and update theta via Algorithm
                    data = shared_data.copy()

                    avg_costs_plus = [data[f'plus_{k}_avg_cost_tau'] for k in range(K)]
                    avg_costs_minus = [data[f'minus_{k}_avg_cost_tau'] for k in range(K)]

                    # Use Algorithm's gradient computation
                    grad = leader_algo.compute_gradient(avg_costs_plus, avg_costs_minus, delta_n)

                    # Set current theta and iteration in leader algo
                    leader_algo.theta = data['theta']
                    leader_algo.n = n

                    # Use Algorithm's optimizer update
                    new_theta = leader_algo.apply_optimizer(grad, optimization)

                    print(f"n = {n}, theta = {new_theta:.4f}, grad = {grad:.6f}, "
                          f"cost+ = {data['plus_0_avg_cost_tau']:.4f}, "
                          f"cost- = {data['minus_0_avg_cost_tau']:.4f}, "
                          f"tau_n = {tau_n}, delta_n = {delta_n:.6f}")

                    shared_data['theta'] = new_theta
                    theta_rows.append({"theta": new_theta})
                    theta_costs_rows.append({
                        "time": time.time(), "mode": sim_type,
                        "theta": new_theta, "n": n,
                        **{f'plus_{k}_avg_cost_tau': data[f'plus_{k}_avg_cost_tau'] for k in range(K)},
                        **{f'minus_{k}_avg_cost_tau': data[f'minus_{k}_avg_cost_tau'] for k in range(K)},
                    })

                    n += 1
                    tau_n = algo.get_tau_n(n)
                    steps = 0
                    costs_sum = 0
                    shared_data["ready_count"] = 0
                    break

                else:
                    # Non-leader: wait for leader to finish update
                    while shared_data["ready_count"] != 0:
                        time.sleep(0.01)
                    costs_sum = 0
                    steps = 0

                # All processes: sync theta from shared state
                theta = shared_data["theta"]
                algo.theta = theta
                algo.theta_step = theta
                n += 1
                steps = 0
                costs_sum = 0
                tau_n = algo.get_tau_n(n)
                break

        shared_data[f'{sim_type}_avg_cost_tau'] = 0

    # Save results
    df_theta = pd.DataFrame(theta_rows)
    df_theta_costs = pd.DataFrame(theta_costs_rows)
    df_theta.to_csv(f"{log_dir}/theta_{sim_type}.csv", index=False, header=True)
    df_theta_costs.to_csv(f"{log_dir}/theta_costs_{sim_type}.csv", index=False, header=True)

    total_simulation_time = time.time() - simulation_start_time

    print(f"\n[{sim_type}] Process DONE. Time: {total_simulation_time:.2f}s")

    sim.hist_times.append(sim.t)
    sim.calculate_time_lengths()

    import json
    with open(f"{log_dir}/summary_{sim_type}.txt", "a") as file:
        file.write(f"{json.dumps(sim.get_result_dict())}\n")

    sim.print_trace_results()


def parallel_simulation(shared_params, shared_data, exp):
    """Launch 2*K parallel simulator processes."""
    theta_lock = Lock()
    multiprocessing.set_start_method("spawn", force=True)

    K = shared_params['K']
    barrier = Barrier(K * 2)

    processes = []
    i = 0
    for k in range(K):
        p_plus = Process(
            target=run_simulator_process,
            args=(k, shared_params, f"plus_{k}", shared_data, exp[i + k], theta_lock, barrier)
        )
        p_minus = Process(
            target=run_simulator_process,
            args=(k, shared_params, f"minus_{k}", shared_data, exp[i + k + 1], theta_lock, barrier)
        )
        i += 1
        processes.append(p_plus)
        processes.append(p_minus)

    for p in processes:
        p.start()
    for p in processes:
        p.join()
        print("Simulation joined successfully")


def load_config(config_path):
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def run_experiments_from_config(config_path):
    """Run parallel scalar NSKW experiments from a JSON configuration file."""
    import json

    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    seeds = config.get('seeds', [1])
    theta_inits = config.get('theta_inits', [config.get('theta_init', 1)])

    arrival_rate = config['arrival_rate']
    warm_service_rate = config['warm_service']['rate']
    cold_service_rate = config['cold_service']['rate']
    cold_start_rate = config['cold_start']['rate']
    expiration_rate = config['expiration']['rate']

    service_process_type = config['warm_service'].get('type', 'Exponential')
    expiration_process_type = config['expiration'].get('type', 'Exponential')
    optimization = config['optimization'].get('type', 'SGD')

    tau = config['tau']
    max_concurrency = config['max_concurrency']
    max_time = config['max_time']
    K = config['K']
    k_delta = config.get('k_delta', 1)
    k_gamma = config.get('k_gamma', 1)

    # Per-theta k_gamma override
    k_gamma_per_theta = config.get('k_gamma_per_theta', None)

    current_time = time.strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get('experiment_name', 'scalar_par')
    base_log_dir = config.get('log_dir', 'logs/')
    base_log_dir = os.path.join(base_log_dir, f"{experiment_name}_arr{arrival_rate}_{current_time}")

    if not os.path.exists(base_log_dir):
        os.makedirs(base_log_dir)

    with open(os.path.join(base_log_dir, 'experiment_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    experiment_start = time.time()

    for ti, theta_init in enumerate(theta_inits):
        current_k_gamma = k_gamma
        if k_gamma_per_theta is not None and ti < len(k_gamma_per_theta):
            current_k_gamma = k_gamma_per_theta[ti]

        for seed in seeds:
            theta_log_dir = os.path.join(base_log_dir, f"theta_{theta_init}")
            if not os.path.exists(theta_log_dir):
                os.makedirs(theta_log_dir)

            log_dir = os.path.join(theta_log_dir, f"seed_{seed}")
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            print(f"\n{'=' * 80}")
            print(f"Starting parallel run: theta_init={theta_init}, seed={seed}, K={K}")
            print(f"Log directory: {log_dir}")
            print(f"{'=' * 80}\n")

            multiprocessing.set_start_method("spawn", force=True)
            manager = Manager()
            shared_data = manager.dict()
            shared_data["theta"] = float(theta_init)
            shared_data["ready_count"] = 0
            for k in range(K):
                shared_data[f"plus_{k}_avg_cost_tau"] = 0.0
                shared_data[f"minus_{k}_avg_cost_tau"] = 0.0

            shared_params = {
                'arrival_rate': arrival_rate,
                'warm_service_rate': warm_service_rate,
                'cold_start_rate': cold_start_rate,
                'cold_service_rate': cold_service_rate,
                'max_time': max_time,
                'k_delta': k_delta,
                'k_gamma': current_k_gamma,
                'theta_init': float(theta_init),
                'tau': tau,
                'K': K,
                'maximum_concurrency': max_concurrency,
                'seed': seed,
                'log_dir': log_dir,
                'expiration_rate': expiration_rate,
                'service_process_type': service_process_type,
                'expiration_process_type': expiration_process_type,
                'optimization': optimization,
            }

            exps = [1] * 2 * K
            parallel_simulation(shared_params, shared_data, exps)

            # Save final config
            with open(os.path.join(log_dir, 'config.json'), 'w') as f:
                json.dump(convert_to_serializable(shared_params), f, indent=2)

    experiment_end = time.time()
    total_time = experiment_end - experiment_start

    print(f"\n{'=' * 80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total experiment time: {total_time:.2f}s ({total_time / 60:.2f}min)")
    print(f"All results saved to: {base_log_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Parallel Scalar NSKW Serverless Simulator')
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON configuration file')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        sys.exit(1)

    try:
        run_experiments_from_config(args.input)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
