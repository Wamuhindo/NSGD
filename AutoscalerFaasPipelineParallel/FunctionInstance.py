from AutoscalerFaasVectoriel.utils import FunctionState
import numpy as np
class FunctionInstance:
  id_counter = 0
  def __init__(self, t, type, cold_service_process, warm_service_process, expiration_process, cold_start_process, execution_time=0, ram=0, cpu=0, gpu=0, tpu=0, priority=1, deadline=None):
    
    self.cold_service_process = cold_service_process
    self.cold_start_process = cold_start_process
    self.warm_service_process = warm_service_process
    self.expiration_process = expiration_process
    self.creation_time = t
    self.id = FunctionInstance.id_counter + 1
    self.execution_time = execution_time
    if deadline == None:
      self.deadline = 100 * execution_time
    else:
      self.deadline = deadline
    self.ram = ram
    self.cpu = cpu
    self.gpu = gpu
    self.tpu = tpu
    self.priority = priority
    self.finished_at = None
    self.state = FunctionState.COLD
    self.reserved = False
    FunctionInstance.id_counter += 1
    self.type = type
    
    #self.generate_cold_departure(t)
    self.generate_init_departure(t)
    self.update_next_termination()
    self.init_resource_usages()

  def __str__(self):
    return f'Function(id={self.id},RAM={self.ram},CPU={self.cpu},GPU={self.gpu},TPU={self.tpu},Entered at {self.creation_time},Finished at {self.finished_at},Execution time={self.execution_time},Deadline={self.deadline},Departure={self.next_departure})'

  def __repr__(self):
    return self.__str__()
  
  def is_cold(self):
    return self.state == FunctionState.COLD
  def is_initializing(self):
    return self.state == FunctionState.INIT_FREE or self.state == FunctionState.INIT_RESERVED
  
  def is_busy(self):
    return self.state == FunctionState.BUSY
  
  def is_idle_on(self):
    return self.state == FunctionState.IDLE_ON
  
  def is_init_free(self):
    return self.state == FunctionState.INIT_FREE
  
  def is_init_reserved(self):
    return self.state == FunctionState.INIT_RESERVED
  
  def reserve(self):
    assert self.state == FunctionState.INIT_FREE, "Function must be in INIT_FREE state to be reserved."
    self.reserved = True
    
  def is_reserved(self):
    return self.reserved
  
  def unreserve(self):
    self.reserved = False
    
  def init_resource_usages(self,cpu_warm=(50,150,),cpu_cold=(200,500,),cpu_idle=(5,10,),mem_warm=(150,256,),mem_cold=(200,256,),mem_idle=(100,200,),gpu_warm=(0,0,),gpu_cold=(0,0,),gpu_idle=(0,0,), gpu_warm_mem=(0,0,),gpu_cold_mem=(0,0,),gpu_idle_mem=(0,0,)):
    self.cpu_warm = cpu_warm
    self.cpu_cold = cpu_cold    
    self.cpu_idle = cpu_idle
    self.mem_warm = mem_warm
    self.mem_cold = mem_cold    
    self.mem_idle = mem_idle
    self.gpu_warm = gpu_warm
    self.gpu_cold = gpu_cold
    self.gpu_idle = gpu_idle
    self.gpu_warm_mem = gpu_warm_mem
    self.gpu_cold_mem = gpu_cold_mem
    self.gpu_idle_mem = gpu_idle_mem
    
  def get_resource_usages(self):
    if self.is_busy():
      return np.random.uniform(*self.cpu_warm),np.random.uniform(*self.mem_warm),np.random.uniform(*self.gpu_warm),np.random.uniform(*self.gpu_warm_mem)
    elif self.is_init_free() or self.is_init_reserved():
      return np.random.uniform(*self.cpu_cold),np.random.uniform(*self.mem_cold),np.random.uniform(*self.gpu_cold),np.random.uniform(*self.gpu_cold_mem)
    elif self.is_idle_on():
      return np.random.uniform(*self.cpu_idle),np.random.uniform(*self.mem_idle),np.random.uniform(*self.gpu_idle),np.random.uniform(*self.gpu_idle_mem)
    else:
      return 0,0,0,0
  
  def make_Init_Reserved(self):
    assert self.state == FunctionState.COLD, "Function must bein  COLD state to be init_reserved."
    self.state = FunctionState.INIT_RESERVED
  def make_Init_Free(self):
    assert self.state == FunctionState.COLD, "Function must bein  COLD state to be init_reserved."
    self.state = FunctionState.INIT_FREE
  
  def transition_state_to(self,state):
    assert state == FunctionState.INIT_FREE or state == FunctionState.INIT_RESERVED, "Cannot transition to INITIALIZING state."
    
    if state == FunctionState.IDLE_ON and self.is_initializing() and self.is_reserved():
        raise ValueError("A reserved instance cannot transition to IDLE_ON state.")
    self.state = state
      
  def generate_init_departure(self, t):
    self.next_init_departure = t + self.cold_start_process.generate_trace()
    self.next_departure = self.next_init_departure if not self.is_reserved else self.next_init_departure + self.warm_service_process.generate_trace()  #t + self.cold_service_process.generate_trace() # could also do this way(check with the others):self.next_init_departure + self.warm_service_process.generate_trace() 
    
  def generate_cold_departure(self, t):
    """generate the departure of the cold request which is the first request received by the instance.

    Parameters
    ----------
    t : float
        Current time in simulation
    """
    self.next_departure = t + self.cold_service_process.generate_trace()
    
  def update_next_termination(self):
    """Update the next scheduled termination if no other requests are made to the instance.
    """
    self.next_termination = self.next_departure + self.expiration_process.generate_trace()
  
  def get_life_span(self):
    
    """Get the life span of the server, e.g. after the server has been terminated

    Returns
    -------
    float
        life span of the instance
    """
    return self.next_termination - self.creation_time
    
  def get_state(self):
    """Get the current state

    Returns
    -------
    str
        currentstate
    """
    return self.state

  def arrival_transition(self, t):
    """Make an arrival transition, which causes the instance to go from IDLE to WARM

    Parameters
    ----------
    t : float
        The time at which the transition has occured, this also updates the next termination.

    Raises
    ------
    Exception
        Raises if currently process a request by being in `COLD` or `WARM` states
    """
    if self.is_cold() or self.is_busy():
        raise Exception('Instance is already busy!')
    elif self.is_idle_on(): 
        self.state = FunctionState.BUSY
        self.next_departure = t + self.warm_service_process.generate_trace()
        self.update_next_termination()
    elif self.is_init_free() and not self.is_reserved():
        self.reserve()
         
      
  def update_next_transition(self, t):
    self.next_departure = self.next_init_departure + self.warm_service_process.generate_trace()
    self.update_next_termination()     
        
  def make_transition(self):
    """Make the next internal transition, either transition into `IDLE` of already processing a request, or `TERM` if scheduled termination has arrived.

    Returns
    -------
    str
        The state after making the internal transition

    Raises
    ------
    Exception
        Raises if already in `TERM` state, since no other internal transitions are possible
    """
    # next transition is a departure

    if self.is_busy():
        self.state = FunctionState.IDLE_ON
    elif self.is_init_reserved() or (self.is_init_free() and self.is_reserved()):
        self.state = FunctionState.BUSY
    elif self.is_init_free() and not self.is_reserved():
        self.state = FunctionState.IDLE_ON
    elif self.is_idle_on():
        self.state = FunctionState.COLD
    else:
        raise Exception("This specific transition is not feasible!")

    return self.state
  def get_next_transition_time(self, t=0):
    """Get how long until the next transition.

    Parameters
    ----------
    t : float, optional
        The current time, by default 0

    Returns
    -------
    float
        The seconds remaining until the next transition
    """
    # next transition would be termination
    if self.is_idle_on():
        return self.get_next_termination(t)
    # next transition would be departure
    if self.is_initializing():
        return self.get_next_init_departure(t)
    return self.get_next_departure(t)

  def get_next_departure(self, t):
    """Get the time until the next departure

    Parameters
    ----------
    t : float
        Current time

    Returns
    -------
    float
        Amount of time until the next departure

    Raises
    ------
    Exception
        Raises if called after the departure
    """
    if t > self.next_departure:
        raise Exception("current time is after departure!")
    return self.next_departure - t
  
  def get_next_init_departure(self, t):
    """Get the time until the next init departure

    Parameters
    ----------
    t : float
        Current time

    Returns
    -------
    float
        Amount of time until the next init departure

    Raises
    ------
    Exception
        Raises if called after the init departure
    """
    if t > self.next_init_departure:
        raise Exception("current time is after init departure!")
    return self.next_init_departure - t

  def get_next_termination(self, t):
    """Get the time until the next termination

    Parameters
    ----------
    t : float
        Current time

    Returns
    -------
    float
        Amount of time until the next termination

    Raises
    ------
    Exception
        Raises if called after the termination
    """
    if t > self.next_termination:
        raise Exception("current time is after termination!")
    return self.next_termination - t
  
