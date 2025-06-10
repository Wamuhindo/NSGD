# ServerlessTemporalSimulator extends the functionality of ServerlessSimulator
# by providing necessary functionality for temporal scenarios where initial
# state is important and the initial process might be different from the
# following service process.

from AutoscalerFaasParallel.FunctionInstance import FunctionInstance
from AutoscalerFaasParallel.ServerlessSimulator import ServerlessSimulator
from AutoscalerFaasParallel.utils import FunctionState


class ServerlessTemporalSimulator(ServerlessSimulator):
    """ServerlessTemporalSimulator extends ServerlessSimulator to enable extraction of temporal characteristics. Also gets all of the arguments accepted by :class:`~simfaas.ServerlessSimulator.ServerlessSimulator`

    Parameters
    ----------
    running_function_instances : list[FunctionInstance]
        A list containing the running function instances
    idle_function_instances : list[FunctionInstance]
        A list containing the idle function instances
    """
    def __init__(self, running_function_instances, idle_function_instances, init_free_function_instances, init_reserved_function_instances,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

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



class ExponentialServerlessTemporalSimulator(ServerlessTemporalSimulator):
    """ExponentialServerlessTemporalSimulator is a simulator assuming exponential distribution for proceesing times which means each process is state-less and we can generate a service time and use that from now on. This class extends ServerlessTemporalSimulator which has functionality for other processes as well.

    Parameters
    ----------
    running_function_instance_count : integer
        running_function_instance_count is the number of instances currently processing a request
    idle_function_instance_next_terminations : list[float]
        idle_function_instance_next_terminations is an array of next termination scheduled for idle functions
        if they receive no new requests.
    """
    def __init__(self, running_function_instance_count, idle_function_instance_next_terminations, init_free_function_count, init_reserved_function_count,
                 *args, **kwargs):
        
        cold_service_process = ExpSimProcess(rate=cold_service_rate)
        warm_service_process = ExpSimProcess(rate=warm_service_rate)
        
        if 'cold_start_time' in args:
            cold_start_process = ConstSimProcess(rate=1/args.get('cold_start_time'))
        if 'cold_start_time' in kwargs:
            cold_start_process = ConstSimProcess(rate=1/kwargs.get('cold_start_time'))
        if cold_start_process is None:
            raise Exception('Cold Start process not defined!')
        

        idle_functions = []
        for next_term in idle_function_instance_next_terminations:
            f = FunctionInstance(0,
                                cold_service_process,
                                warm_service_process,
                                expiration_threshold,
                                cold_start_process=cold_start_process
                                )

            f.state = FunctionState.IDLE_ON
            # when will it be destroyed if no requests
            f.next_termination = next_term
            # so that they would be less likely to be chosen by scheduler
            f.creation_time = 0.01
            idle_functions.append(f)

        running_functions = []
        for _ in range(running_function_instance_count):
            f = FunctionInstance(0,
                                cold_service_process,
                                warm_service_process,
                                expiration_threshold,
                                cold_start_process=cold_start_process
                                )

            f.state = FunctionState.IDLE_ON
            # transition it into running mode
            f.arrival_transition(0)

            running_functions.append(f)
            
        init_free_functions = []    
        for _ in range(init_free_function_count):
            f = FunctionInstance(0,
                                cold_service_process,
                                warm_service_process,
                                expiration_threshold,
                                cold_start_process=cold_start_process
                                )

            f.make_Init_Free()

            init_free_functions.append(f)
            
        init_reserved_functions = []     
        for _ in range(init_reserved_function_count):
            f = FunctionInstance(0,
                                cold_service_process,
                                warm_service_process,
                                expiration_threshold,
                                cold_start_process=cold_start_process
                                )

            f.make_Init_Reserved()

            init_reserved_functions.append(f)

        super().__init__(
            running_function_instances=running_functions,
            idle_function_instances=idle_functions,
            init_free_function_instances=init_free_functions,
            init_reserved_function_instances=init_reserved_functions,
            *args, **kwargs
        )


if __name__ == "__main__":
    from simfaas.SimProcess import ExpSimProcess, ConstSimProcess

    print("Performing Temporal Simulation")
    cold_service_rate = 1/2.163
    warm_service_rate = 1/2.016
    expiration_threshold = 600

    arrival_rate = 2.9
    cold_start_time = 0.1
    max_time = 1000000

    running_function_count = 3
    idle_function_count = 10
    init_free_function_count = 0
    init_reserved_function_count = 4

    cold_service_process = ExpSimProcess(rate=cold_service_rate)
    warm_service_process = ExpSimProcess(rate=warm_service_rate)
    cold_start_process = ConstSimProcess(rate=1/cold_start_time)

    idle_functions = []
    for _ in range(idle_function_count):
        f = FunctionInstance(0,
                             cold_service_process,
                             warm_service_process,
                             expiration_threshold,
                             cold_start_process=cold_start_process
                             )

        f.state = FunctionState.IDLE_ON
        # when will it be destroyed if no requests
        f.next_termination = 300
        # so that they would be less likely to be chosen by scheduler
        f.creation_time = 0.01
        idle_functions.append(f)

    running_functions = []
    for _ in range(running_function_count):
        f = FunctionInstance(0,
                             cold_service_process,
                             warm_service_process,
                             expiration_threshold,
                             cold_start_process=cold_start_process
                             )

        f.state = FunctionState.IDLE_ON
        # transition it into running mode
        f.arrival_transition(0)

        running_functions.append(f)
    
    init_free_functions = []    
    for _ in range(init_free_function_count):
        f = FunctionInstance(0,
                             cold_service_process,
                             warm_service_process,
                             expiration_threshold,
                             cold_start_process=cold_start_process
                             )

        f.make_Init_Free()

        init_free_functions.append(f)
        
    init_reserved_functions = []     
    for _ in range(init_reserved_function_count):
        f = FunctionInstance(0,
                             cold_service_process,
                             warm_service_process,
                             expiration_threshold,
                             cold_start_process=cold_start_process
                             )

        f.make_Init_Reserved()

        init_reserved_functions.append(f)

    sim = ServerlessTemporalSimulator(running_functions, idle_functions, init_free_functions, init_reserved_functions, cold_start_time=cold_start_time, \
                                      arrival_rate=arrival_rate, warm_service_rate=warm_service_rate, cold_service_rate=cold_service_rate,
                                      expiration_threshold=expiration_threshold, max_time=max_time)
    sim.generate_trace(debug_print=False, progress=True)
    sim.print_trace_results()


    print("Testing out the new functionality added.")
    idle_function_instance_next_terminations = [300] * idle_function_count
    sim = ExponentialServerlessTemporalSimulator(running_function_count, idle_function_instance_next_terminations, init_free_function_count=init_free_function_count,\
                                        init_reserved_function_count=init_reserved_function_count, cold_start_time=cold_start_time, arrival_rate=arrival_rate, 
                                        warm_service_rate=warm_service_rate, cold_service_rate=cold_service_rate,
                                        expiration_threshold=expiration_threshold, max_time=max_time)
    sim.generate_trace(debug_print=False, progress=True)
    sim.print_trace_results()
