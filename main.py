from ray.rllib.algorithms.algorithm import Algorithm
import numpy as np
from ray.rllib.policy.policy import Policy
from RL4CC.experiments.train import TrainingExperiment
import argparse
from classes.Logger import Logger
import sys
import os
import ray


if __name__ == '__main__':

    # initialize error stream
    error = Logger(stream=sys.stderr, verbose=1, is_error=True)
    # check if the system configuration file exists
    
    seed = 1234
    ray.init(address='192.168.1.6:6380', include_dashboard=False, ignore_reinit_error=True)
    np.random.seed(seed)
    exp = TrainingExperiment(f"configs/exp_config.json")
    exp.run()
        


    
