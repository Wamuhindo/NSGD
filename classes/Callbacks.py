import sys
from typing import Dict

from ray.rllib.evaluation import RolloutWorker
from ray.rllib.algorithms.algorithm import Algorithm
import json
import os
from RL4CC.callbacks import BaseCallbacks

class MyCallbacks(BaseCallbacks):

    def on_episode_created(self, *, worker, **kwargs):
        
        super().on_episode_created(worker=worker, **kwargs)

        if worker.env._current_step == 1:
            worker.env._current_step = worker.worker_index + 1
        worker_type = "training"
        if worker.env.worker_type == "evaluation":
            #if "training" in worker.env.file_name:
            #    worker.env.file_name = worker.env.file_name.replace("training", "validation")
            worker_type = "evaluation"

        self._update_environment_settings(worker, worker_type)
        
    def _update_environment_settings(self, worker,worker_type):
            if worker_type == "training":
                a = 0
                
            else:
                a = 0

    def on_episode_step(self, *, worker: RolloutWorker, **kwargs):
        
        super().on_episode_step(worker=worker, **kwargs)
        
    def on_evaluate_start(
            self,
            *,
            algorithm: Algorithm,
            **kwargs):
        super().on_evaluate_start(algorithm=algorithm, **kwargs)
        def make_update_env_fn():
            def update_env_conf(env):
                env.worker_type = "evaluation"
                env.seed = env.seed + 2
                if "/evaluation" not in env.log_dir:
                    env.log_dir = f"{env.log_dir}/evaluation_1"
                    env.simulator.log_dir = env.log_dir 
                    if not os.path.exists(env.log_dir):
                        os.makedirs(env.log_dir)
                else:
                    ev = env.log_dir.split("/")[-1]
                    log_dir = os.path.dirname(env.log_dir)
                    env.log_dir = f"{log_dir}/evaluation_{int(ev.split('_')[-1])+1}"
                    env.simulator.log_dir = env.log_dir
                    if not os.path.exists(env.log_dir):
                        os.makedirs(env.log_dir)

            def update_env_fn(worker):
                worker.foreach_env(update_env_conf)

            return update_env_fn
        algorithm.evaluation_workers.foreach_worker(
            make_update_env_fn()
        )


    def on_evaluate_end(self, *, algorithm, evaluation_metrics, **kwargs):
        super().on_evaluate_end(algorithm=algorithm, evaluation_metrics=evaluation_metrics, **kwargs)