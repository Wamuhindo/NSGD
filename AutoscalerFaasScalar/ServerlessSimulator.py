"""
Serverless simulator adapted for the scalar NSKW algorithm from paper 2502.01284v2.

Key difference from the 3-parameter version:
  - theta is a single scalar controlling scale-up (number of extra servers)
  - No theta_idle or theta_exp parameters
  - Expiration rate is fixed (not optimized)
  - Scale-up rule pi_theta from eq. 14
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from AutoscalerFaasScalar.SimProcess import ExpSimProcess, ConstSimProcess, ParetoSimProcess
from AutoscalerFaasScalar.FunctionInstance import FunctionInstance
from AutoscalerFaasScalar.utils import FunctionState, SystemState
from AutoscalerFaasScalar.Algorithm import ScalarAutoScalingAlgorithm
import numpy as np
from numpy.random import default_rng, SeedSequence
import time
import pandas as pd
from tqdm import tqdm
import random


class ServerlessSimulator:
    def __init__(self, arrival_process=None, warm_service_process=None,
                 cold_service_process=None, cold_start_process=None,
                 service_process_type="Exponential", expiration_process_type="Exponential",
                 maximum_concurrency=50, log_dir="", **kwargs):
        super().__init__()
        self.seed = kwargs.get("seed", 1)

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

        # Setup expiration process (fixed rate, not optimized)
        expiration_rate = kwargs.get('expiration_rate', 0.01)
        if expiration_process_type == "Deterministic":
            self.expiration_process = ConstSimProcess(expiration_rate)
        else:
            self.expiration_process = ExpSimProcess(expiration_rate, gen=exp_rng)

        self.maximum_concurrency = maximum_concurrency
        self.reset_trace()

        # Algorithm parameters (scalar)
        k_delta = kwargs.get('k_delta', 1)
        k_gamma = kwargs.get('k_gamma', 10)
        theta_init = kwargs.get('theta_init', 1)
        tau = kwargs.get('tau', 1e6)
        T = kwargs.get('max_time', 1e8)
        K = kwargs.get('K', 2)

        self.seed = kwargs.get('seed', 1)
        self.rang_delta_plus = np.random.default_rng(self.seed)
        self.rang_delta_minus = np.random.default_rng(self.seed)

        self.autoscaler = ScalarAutoScalingAlgorithm(
            N=maximum_concurrency, k_delta=k_delta, k_gamma=k_gamma,
            theta_init=theta_init, tau=tau, K=K, T=T, log_dir=log_dir
        )

        self.state = self.autoscaler.get_state()
        self.max_time = self.autoscaler.Tmax
        self.log_dir = log_dir

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

    def initialiaze_system(self, t, running_function_count, idle_function_count,
                           init_free_function_count, init_reserved_function_count):
        np.random.seed(self.seed)
        idle_functions = []
        for _ in range(idle_function_count):
            f = FunctionInstance(t, self.cold_service_process, self.warm_service_process,
                                self.expiration_process, self.cold_start_process)
            f.state = FunctionState.IDLE_ON
            f.creation_time = 0.01
            idle_functions.append(f)

        running_functions = []
        for _ in range(running_function_count):
            f = FunctionInstance(t, self.cold_service_process, self.warm_service_process,
                                self.expiration_process, self.cold_start_process)
            f.state = FunctionState.IDLE_ON
            f.arrival_transition(t)
            running_functions.append(f)

        init_free_functions = []
        for _ in range(init_free_function_count):
            f = FunctionInstance(t, self.cold_service_process, self.warm_service_process,
                                self.expiration_process, self.cold_start_process)
            f.make_Init_Free()
            init_free_functions.append(f)

        init_reserved_functions = []
        for _ in range(init_reserved_function_count):
            f = FunctionInstance(t, self.cold_service_process, self.warm_service_process,
                                self.expiration_process, self.cold_start_process)
            f.make_Init_Reserved()
            init_reserved_functions.append(f)

        self.set_initial_state(running_functions, idle_functions,
                               init_free_functions, init_reserved_functions)

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
        self.t = 0
        self.total_finished = 0
        self.last_total_finished = 0
        self.job_rejected = False
        self.missed_update = 0

    def save_rng_states(self):
        """Save RNG states of all simulation processes for CRN (Common Random Numbers)."""
        import copy
        self._saved_rng_states = {
            'arrival': copy.deepcopy(self.arrival_process.rangen.bit_generator.state),
            'warm_service': copy.deepcopy(self.warm_service_process.rangen.bit_generator.state),
            'cold_start': copy.deepcopy(self.cold_start_process.rangen.bit_generator.state),
            'expiration': copy.deepcopy(self.expiration_process.rangen.bit_generator.state),
        }

    def restore_rng_states(self):
        """Restore RNG states to ensure - phase sees same random numbers as + phase."""
        self.arrival_process.rangen.bit_generator.state = self._saved_rng_states['arrival']
        self.warm_service_process.rangen.bit_generator.state = self._saved_rng_states['warm_service']
        self.cold_start_process.rangen.bit_generator.state = self._saved_rng_states['cold_start']
        self.expiration_process.rangen.bit_generator.state = self._saved_rng_states['expiration']

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
            # In scalar version, no theta_idle — scale up only on cold starts
            # Paper: scale-up happens when idle_count == 0 (Algorithm 1)
            if self.idle_count == 0:
                self.start_init_free_servers(t, theta)

    def init_free_arrival(self, t, theta):
        self.total_req_count += 1
        self.total_queued_jobs_count += 1
        self.queued_jobs_count += 1
        current_cold_servers = self.current_cold_servers()
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
        """Get the first historical array index that is after the time t."""
        return np.min(np.where(np.array(self.hist_times) > t))

    def get_skip_init(self, skip_init_time=None, skip_init_index=None):
        """Get the minimum index which satisfies both the time and index count we want to skip
        in the beginning of the simulation, used to reduce the transient effect for steady-state values."""
        skip_init = 0
        if skip_init_time is not None:
            skip_init = self.get_index_after_time(skip_init_time)
        if skip_init_index is not None:
            skip_init = max(skip_init, skip_init_index)
        return skip_init

    def get_request_custom_states(self, hist_states, skip_init_time=None, skip_init_index=None):
        """Get request statistics for an array of custom states."""
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
        """Analyse custom states: amount of time spent in each state, and transition times."""
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
        """Get the average residence time for each state in custom state encoding."""
        residence_times, _ = self.analyze_custom_states(hist_states, skip_init_time, skip_init_index)

        residence_time_avgs = {}
        for s in residence_times:
            residence_time_avgs[s] = np.mean(residence_times[s])

        return residence_time_avgs

    def get_cold_start_prob(self):
        return self.total_cold_count / self.total_req_count

    def get_average_lifespan(self):
        life_spans = np.array([s.get_life_span() for s in self.prev_servers])
        if len(life_spans):
            return life_spans.mean()
        else:
            return np.inf

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

    def trace_condition(self, t):
        return self.autoscaler.running_condition()

    @staticmethod
    def print_time_average(vals, probs, column_width=15):
        """Print the time average of states."""
        print(f"{'Value'.ljust(column_width)} Prob")
        print("".join(["="]*int(column_width*1.5)))
        for val, prob in zip(vals, probs):
            print(f"{str(val).ljust(column_width)} {prob:.4f}")

    def calculate_time_average(self, values, skip_init_time=None, skip_init_index=None):
        """Calculate the time-averaged distribution of the values passed in."""
        assert len(values) == len(self.time_lengths), "Values should be same length as history array (number of transitions)"

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
        """Calculate cost based on current system state."""
        cost = self.autoscaler.compute_cost(self.state)
        self.job_rejected = False
        return cost

    def generate_trace(self, debug_print=False, progress=False):
        pbar = None
        if progress:
            pbar = tqdm(total=int(self.max_time))

        t = 0
        pbar_t_update = 0
        pbar_interval = int(self.max_time / 100)
        next_arrival = t + self.req()

        while self.trace_condition(t):
            if progress:
                if int(self.autoscaler.t - pbar_t_update) > pbar_interval:
                    pbar.update(int(self.autoscaler.t) - pbar_t_update)
                    pbar_t_update = int(self.autoscaler.t)

            self.hist_times.append(t)
            self.update_hist_arrays(t)

            # No servers — next event is arrival
            if not self.has_server():
                theta = self.autoscaler.get_theta_step()
                t = next_arrival
                self.t = t
                self.total_requests_log.append(t)
                next_arrival = t + self.req()
                self.cold_start_arrival(t, theta=theta)
                self.update_state()
                self.autoscaler.simulate_step(self.state, self)
                continue

            server_next_transitions = np.array([s.get_next_transition_time(t) for s in self.servers])

            # Next event is arrival
            if (next_arrival - t) < server_next_transitions.min():
                t = next_arrival
                self.t = t
                next_arrival = t + self.req()
                self.total_requests_log.append(t)

                theta = self.autoscaler.get_theta_step()
                if self.is_warm_available(t):
                    self.warm_start_arrival(t, theta)
                elif self.is_init_free_available(t):
                    self.init_free_arrival(t, theta)
                else:
                    self.cold_start_arrival(t, theta=theta)

                self.update_state()
                self.autoscaler.simulate_step(self.state, self)
                continue

            # Next event is a server state change
            else:
                idx = server_next_transitions.argmin()
                t = t + server_next_transitions[idx]
                self.t = t
                old_state = self.servers[idx].get_state()
                new_state = self.servers[idx].make_transition()

                if new_state == FunctionState.COLD:
                    self.prev_servers.append(self.servers[idx])
                    self.idle_count -= 1
                    self.server_count -= 1
                    del self.servers[idx]

                elif new_state == FunctionState.IDLE_ON:
                    self.total_finished += 1
                    if old_state == FunctionState.BUSY:
                        self.running_count -= 1
                    elif old_state == FunctionState.INIT_FREE and not self.servers[idx].is_reserved():
                        self.init_free_count -= 1
                        self.servers[idx].unreserve()
                    else:
                        raise Exception(f"Unknown transition in states: {new_state}")
                    self.idle_count += 1

                elif new_state == FunctionState.BUSY:
                    self.served_requests_log.append(t)
                    if old_state == FunctionState.INIT_RESERVED:
                        self.init_reserved_count -= 1
                    elif old_state == FunctionState.INIT_FREE and self.servers[idx].is_reserved():
                        self.servers[idx].update_next_transition(t)
                        self.servers[idx].unreserve()
                        self.init_free_booked_count -= 1
                        self.queued_jobs_count -= 1
                        self.total_warm_count += 1
                        self.hist_req_warm_idxs.append(len(self.hist_times) - 1)
                    else:
                        raise Exception(f"Unknown transition in states: {new_state}")
                    self.running_count += 1

                else:
                    raise Exception(f"Unknown transition in states: {new_state}")

                self.update_state()
                self.autoscaler.simulate_step(self.state, self)

        # End of trace
        self.hist_times.append(t)
        self.calculate_time_lengths()
        if progress:
            pbar.update(int(self.max_time) - pbar_t_update)
            pbar.close()

        np.savetxt(f"{self.log_dir}/theta.csv", self.autoscaler.thetas, delimiter=",", fmt="%2f")
        np.savetxt(f"{self.log_dir}/all_costs.csv", self.autoscaler.all_costs, delimiter=",", fmt="%2f")


def load_config(config_path):
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def parse_distribution_params(dist_config):
    """Parse distribution configuration into rate parameter."""
    return dist_config.get('rate')


def convert_to_serializable(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def run_single_experiment(config, seed, run_idx, total_runs, base_log_dir):
    import json

    # Extract parameters
    arrival_rate = config['arrival_rate']
    warm_service_rate = config['warm_service']['rate']
    cold_service_rate = config['cold_service']['rate']
    cold_start_rate = config['cold_start']['rate']
    expiration_rate = config['expiration']['rate']

    service_process_type = config['warm_service'].get('type', 'Exponential')
    expiration_process_type = config['expiration'].get('type', 'Exponential')

    optimization = config['optimization'].get('type', 'sgd')

    # Scalar algorithm parameters
    theta_init = config['theta_init']
    tau = config['tau']
    max_concurrency = config['max_concurrency']
    max_time = config['max_time']
    K = config['K']
    k_delta = config.get('k_delta', 1)
    k_gamma = config.get('k_gamma', 10)

    # Create run-specific log directory
    run_log_dir = os.path.join(base_log_dir, f"run_{run_idx + 1}_seed_{seed}")
    if not os.path.exists(run_log_dir):
        os.makedirs(run_log_dir)

    algo_params = {
        "k_gamma": k_gamma,
        "k_delta": k_delta,
        "K": K,
        "theta_init": theta_init,
        "tau": tau,
        "max_time": max_time,
        "seed": seed,
        "expiration_rate": expiration_rate,
    }

    # Save run config
    run_config = {
        'run_index': run_idx + 1, 'seed': seed,
        'arrival_rate': arrival_rate, 'warm_service_rate': warm_service_rate,
        'cold_service_rate': cold_service_rate, 'cold_start_rate': cold_start_rate,
        'expiration_rate': expiration_rate, 'optimization': optimization,
        'max_concurrency': max_concurrency, 'max_time': max_time,
        'tau': tau, 'K': K, 'theta_init': theta_init,
        'k_delta': k_delta, 'k_gamma': k_gamma,
    }
    with open(os.path.join(run_log_dir, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Starting Run {run_idx + 1}/{total_runs} with seed={seed}, theta_init={theta_init}")
    print(f"Log directory: {run_log_dir}")
    print(f"{'=' * 80}\n")

    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)

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
    sim.initialiaze_system(0, 0, 0, 0, 0)
    sim.generate_trace(debug_print=False, progress=True)

    end_time = time.time()
    wall_clock_time = end_time - start_time

    results = sim.get_result_dict()
    results['seed'] = seed
    results['run_index'] = run_idx + 1
    results['wall_clock_time_seconds'] = wall_clock_time
    results['simulated_time'] = sim.get_trace_end()
    results['theta_init'] = theta_init

    print(f"\nResults for Run {run_idx + 1}:")
    sim.print_trace_results()
    print(f"Execution Time: {wall_clock_time:.2f} seconds ({wall_clock_time / 60:.2f} minutes)")

    with open(os.path.join(run_log_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_experiments_from_config(config_path):
    import json

    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    seeds = config.get('seeds', [1])
    theta_inits = config.get('theta_inits', [config.get('theta_init', 1)])

    # Create base log directory
    current_time = time.strftime("%Y%m%d_%H%M%S")
    arrival_rate = config['arrival_rate']
    base_log_dir = config.get('log_dir', 'logs/')
    experiment_name = config.get('experiment_name', 'scalar')
    base_log_dir = os.path.join(base_log_dir, f"{experiment_name}_arr{arrival_rate}_{current_time}")

    if not os.path.exists(base_log_dir):
        os.makedirs(base_log_dir)

    with open(os.path.join(base_log_dir, 'experiment_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    all_results = []
    total_runs = len(theta_inits) * len(seeds)
    experiment_start = time.time()

    # Per-theta k_gamma override: k_gamma_per_theta[i] applies to theta_inits[i]
    k_gamma_per_theta = config.get('k_gamma_per_theta', None)

    run_idx = 0
    for ti, theta_init in enumerate(theta_inits):
        theta_config = dict(config)
        theta_config['theta_init'] = theta_init

        # Override k_gamma if per-theta values are provided
        if k_gamma_per_theta is not None and ti < len(k_gamma_per_theta):
            theta_config['k_gamma'] = k_gamma_per_theta[ti]

        theta_log_dir = os.path.join(base_log_dir, f"theta_{theta_init}")
        if not os.path.exists(theta_log_dir):
            os.makedirs(theta_log_dir)

        for seed in seeds:
            try:
                results = run_single_experiment(theta_config, seed, run_idx, total_runs, theta_log_dir)
                all_results.append(results)
            except Exception as e:
                print(f"\nError in run {run_idx + 1} with seed {seed}, theta_init {theta_init}: {e}")
                import traceback
                traceback.print_exc()
            run_idx += 1

    experiment_end = time.time()
    total_experiment_time = experiment_end - experiment_start

    print(f"\n{'=' * 80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total runs completed: {len(all_results)}/{total_runs}")
    print(f"Total experiment time: {total_experiment_time:.2f} seconds ({total_experiment_time / 60:.2f} minutes)")

    aggregated_results = {
        'total_runs': len(all_results),
        'total_experiment_time_seconds': total_experiment_time,
        'runs': all_results
    }
    with open(os.path.join(base_log_dir, 'aggregated_results.json'), 'w') as f:
        json.dump(aggregated_results, f, indent=2)

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(base_log_dir, 'all_runs_summary.csv'), index=False)

    print(f"\nAll results saved to: {base_log_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Scalar NSKW Serverless Simulator (paper 2502.01284v2)')
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
