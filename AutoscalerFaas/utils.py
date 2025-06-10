from enum import Enum

class FunctionState(Enum):
    COLD = 0
    INIT_FREE = 1
    INIT_RESERVED = 2
    IDLE_ON = 3
    BUSY = 4
    
class SystemState(Enum):
    COLD = 0
    IDLE_ON = 1
    BUSY = 2
    INITIALIZING = 3
    INIT_RESERVED = 4
    