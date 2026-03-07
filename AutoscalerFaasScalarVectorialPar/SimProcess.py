import numpy as np
from scipy.stats import expon, pareto


class SimProcess:
    def __init__(self):
        super().__init__()

    def generate_trace(self):
        raise NotImplementedError


class ExpSimProcess(SimProcess):
    def __init__(self, rate, gen=np.random):
        super().__init__()
        self.rate = rate
        self.rangen = gen

    def generate_trace(self):
        return self.rangen.exponential(1 / self.rate)


class ConstSimProcess(SimProcess):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def generate_trace(self):
        return 1 / self.rate


class ParetoSimProcess(SimProcess):
    def __init__(self, shape, scale=1.0, gen=np.random):
        super().__init__()
        self.shape = shape
        self.scale = scale
        self.rangen = gen

    def generate_trace(self):
        return self.rangen.pareto(self.shape) * self.scale
