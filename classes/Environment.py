import numpy as np
from classes.serialization import Serializable
from gymnasium.spaces import Box, Discrete, Dict
from classes.Logger import Logger
from gymnasium.envs.registration import EnvSpec
import sys
import json
import os.path
from RL4CC.environment.base_environment import BaseEnvironment
from datetime import datetime
import tensorflow as tf
from ServerlessSimulator import ServerlessSimulator
from numpy.random import default_rng, SeedSequence

class Environment(BaseEnvironment, Serializable):

    def __init__(self, env_config):

        super().__init__(env_config)
        self._error: Logger = Logger(is_error=True)
        
        #print("ENV CONFIG", env_config)
        #sys.exit()
        keys = ["env_params","model_name"]
        self.check_and_init_keys(keys, env_config)
        
        self.set_logger(env_config)
        
        self._logger.log("Initializing environment parameters...", 2)
        
        # Extract environment parameters
        keys = ["min_replicas","max_concurency","sampling_window","stats_window","observation_config","actions_config","reward_config","function_config","simulator_config"]
        self.check_and_init_keys(keys, self.env_params)
        
        
        # Extract observation parameters
        # Dynamically create spaces.Dict observation space
        self.observation_space = Dict({
            key: Box(low=np.array([value["low"]]), high=np.array([value["high"]]),dtype=np.float32)
            for key, value in self.observation_config.items()
        })


        print("Observation Space:", self.observation_space)
        
        # Dynamically create spaces.Dict action space
        self.action_space = Discrete(len(self.actions_config))
        
        self._action_to_scale = {i: v for i, v in enumerate(self.actions_config)}
        
        self.check_and_init_keys(["reward_min", "reward_max","reward_params"], self.reward_config)
        
        
        self._current_step = 0
        self.name = "env"
        self.reward_history = []
        self.score = 0
        self.timestep = 0
        self.episode = 0
        self.loop = 0
        self.seed = 2
        self._logger.level += 1

        self._last_obs = None 
        
        os.makedirs(f'{self.log_dir}/{self.model_name}', exist_ok=True)
        self.file_writer = tf.summary.create_file_writer(f'{self.log_dir}/{self.model_name}')
        #self.file_writer.set_as_default() 

        self.spec = EnvSpec(id="Environment", max_episode_steps=self.max_time)

        self._reward_file = f'{self.log_dir}/reward_history_{self.model_name}.json'
        self.current_replica = 1
        self.last_t_sim = 0
        self.simulator: ServerlessSimulator = ServerlessSimulator(maximum_concurrency=self.max_concurency, function_config=self.function_config,**self.simulator_config )
        self.simulator.log_dir = self.log_dir
        self.worker_type = "training"
        #print("REACHING THIS PLACE")


    def check_and_init_keys(self, keys, dictionary):
        for key in keys:
            if key not in dictionary.keys():
                self._error.log(f"The key {key} is missing in the configuration file (env_config.json)")
                sys.exit(1)
            setattr(self, key, dictionary[key])
            
    def set_logger(self, env_config):
                # Set the logger
        try:
            log_dir = env_config["logdir"]
        except KeyError:
            print("The key 'log_dir' is missing in the configuration file (env_config.json) using the current dir as default log dir")
            log_dir = None
        
        currentDateAndTime = datetime.now()
        currentTime = currentDateAndTime.strftime("%d_%m_%Y_%H_%M_%S")
        logfile_environment = None
        if log_dir is not None and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            logfile_environment = f"{log_dir}/env_{currentTime}.log" 
        if logfile_environment:  
            file_stream = open(logfile_environment,"w")
        else:
            file_stream = sys.stdout
        self._logger = Logger(stream=file_stream, verbose=3, is_error=False)

        # Set the log directory to current directory if not provided
        self.log_dir = os.getcwd() if log_dir is None else log_dir
    
    
    def mem_to_gb(self, mem):
        return round((mem/(1024)), 2)
    
    
    def _take_action(self, action):
        
        if action >= len(self.actions_config):
            self._error.log(f"Action {action} is not in the action space")
            sys.exit(1)
        
        action_fb = True
            
        scale_value = self.current_replica + action
        current_replicas = self.current_replica
        
        if action < 0 and scale_value < (self.min_replicas+1):
            action_fb = False
        elif action > 0 and scale_value > self.max_concurency:
            action_fb = False
        else:
            self.current_replica = scale_value
            
        action_feedback = self.simulator.set_agent_action(action)
            
        info = {'action': action, 
               'current_replicas': current_replicas,
               'scale_value': scale_value,
               'action_feedback':action_fb and action_feedback}
        return info
    
    def _get_info(self):
        return {}
    
    def _normalize_obs(self, value, low, high):
        return (value - low) / (high - low) if high > low else value
    
    def _get_obs2(self):
        
        state = self.simulator.get_state()  
          
        replicas = max(1,state["replicas"])
        observation ={
            "replica": np.array([replicas],dtype=np.float32),
            "requests": np.array([state["requests"]],dtype=np.float32),
            "throughput": np.array([state["served_requests"]],dtype=np.float32),
            "avg_cpu": np.array([state["avg_cpu"]],dtype=np.float32),
            "avg_mem": np.array([state["avg_memory"]],dtype=np.float32),
            "average_execution_time": np.array([state["average_execution_time"]],dtype=np.float32),
        }


        return observation
    
    def _get_obs(self):
        state = self.simulator.get_state() 
        replicas = max(1,state["replicas"]) 

        observation = {
            "replica": np.array([self._normalize_obs(replicas, self.observation_config["replica"]["low"], self.observation_config["replica"]["high"])], dtype=np.float32),
            "requests": np.array([self._normalize_obs(state["requests"], self.observation_config["requests"]["low"], self.observation_config["requests"]["high"])], dtype=np.float32),
            "throughput": np.array([self._normalize_obs(state["served_requests"], self.observation_config["throughput"]["low"], self.observation_config["throughput"]["high"])], dtype=np.float32),
            "avg_cpu": np.array([self._normalize_obs(state["avg_cpu"], self.observation_config["avg_cpu"]["low"], self.observation_config["avg_cpu"]["high"])], dtype=np.float32),
            "avg_mem": np.array([self._normalize_obs(state["avg_memory"], self.observation_config["avg_mem"]["low"], self.observation_config["avg_mem"]["high"])], dtype=np.float32),
            "average_execution_time": np.array([self._normalize_obs(state["average_execution_time"], self.observation_config["average_execution_time"]["low"], self.observation_config["average_execution_time"]["high"])], dtype=np.float32),
            "replicas": replicas
        }

        return observation
    
    def _write_to_board(self, obs, action, rew, info, step, episode):
        # write to tensorboard
        with self.file_writer.as_default():
            tf.summary.scalar('avg_execution_time', obs["average_execution_time"][0], step)
            tf.summary.scalar('throughput', obs["throughput"][0], step)
            tf.summary.scalar('requests', obs["requests"][0], step)
            tf.summary.scalar('replicas', obs["replica"][0], step)
            tf.summary.scalar('cpu', obs["avg_cpu"][0], step)
            tf.summary.scalar('mem', obs["avg_mem"][0], step)
            tf.summary.scalar('episode', episode, step)
            tf.summary.scalar('action', (action), step)
            if info['action_feedback']:
                tf.summary.scalar('action_feedback', 1 , step)
            else:
                tf.summary.scalar('action_feedback', 0 , step)
            tf.summary.scalar('n-step_reward', rew, step)
        
    def _calculate_reward(self, obs, metadata={}):
        reward = 0
        meta_scale_value = metadata['scale_value']
        throughput = obs["throughput"] # %
        replicas = obs["replicas"]
        avg_cpu = obs["avg_cpu"] # % 0 - 1
        avg_mem = obs["avg_mem"] # % 0 - 1


        alpha = self.reward_params["alpha"]
        beta = self.reward_params["beta"]
        gamma = self.reward_params["gamma"]
        phi = self.reward_params["phi"]

        r_th = alpha * (throughput ** 2)
        r_cpu = beta * (avg_cpu*100)
        r_mem = gamma * (avg_mem*100)
        r_rep = -phi * ((replicas - (self.min_replicas+1)) ** 2)

        reward = r_th + r_cpu + r_mem + r_rep
        #print("REWARD",reward)
        reward = round(reward[0], 5)

        # action unsuccessful
        if (meta_scale_value != replicas):
            reward += self.reward_min
        
        return reward    
        
    def reset(self, seed=None, options=None):
        """reset the environment

        Args:
            state (_type_, optional): the state to wich to reset the environement. Defaults to None.

        Returns:
            state: the state to which the enviroment is reset
        """
        super().reset(seed=seed)

        # reset other paramters based on the environment
        self.score = 0
        self.loop = 0
        observation = self._get_obs()
        info = self._get_info()

        _seed = self.seed
        ss = SeedSequence(_seed)
        cs_rng, svc_rng, exp_rng, arr_rng = [default_rng(s) for s in ss.spawn(4)]
        self.simulator.cold_start_process.rangen = cs_rng
        self.simulator.warm_service_process.rangen = svc_rng
        self.simulator.expiration_process.rangen = exp_rng
        self.simulator.arrival_process.rangen = arr_rng
        
        running_function_count = 0
        idle_function_count = 0
        init_free_function_count = 0
        init_reserved_function_count = 0
        self.simulator.initialiaze_system(self.simulator.t, running_function_count, idle_function_count, init_free_function_count, init_reserved_function_count )
        if self.time_step >= self.max_time:
            self.simulator.file.close()

        observation["requests"] = np.array([0],dtype=np.float32)
        observation["throughput"] = np.array([1],dtype=np.float32)
        observation["average_execution_time"] = np.array([0],dtype=np.float32)
        if "replicas" in observation.keys():
            del observation["replicas"] 
        self._last_obs = observation
      
        return observation, info

    def set_last_sim_time(self, t):
        self.last_t_sim = t
    def step(self, _action):
        done = False
        # Map the action (element of {0,1,2,3,4}) to scaling
        action = self._action_to_scale[_action]

        #print("ACTION",action)

        # execute the action in environment
        info = self._take_action(action=action)

        # immediate negative reward - invalid action
        if info['action_feedback'] == False:
            self._write_to_board(self._last_obs, action, -2, info, self.timestep, self.episode)
            self.timestep += 1
            self.loop += 1
            self.score += -2
            if self.loop == 10:
                done = False
            done = self.time_step >= self.max_time
                
            info['reward'] = -2
            info["current_time"] = self.timestep
            
            return self._last_obs, -2, done, False, info 
        else:
            # wait for the sampling window to get the next observation
            #time.sleep(self.sampling_window)
            rejected, requests, avg_time = self.simulator.run_simulation(self.set_last_sim_time,self.last_t_sim,self.sampling_window)
            if avg_time == np.inf:
                avg_time = self.observation_config["average_execution_time"]["high"]
                
            # get the next observation
            next_obs = self._get_obs()
            
            # next_obs["requests"] = np.array([requests],dtype=np.float32)
            throughput = 100 if requests==0 else (requests - rejected)*100/requests 
            # next_obs["throughput"] = np.array([throughput],dtype=np.float32)
            # next_obs["average_execution_time"] = np.array([avg_time],dtype=np.float32)


            next_obs["requests"] = np.array([self._normalize_obs(requests, self.observation_config["requests"]["low"], self.observation_config["requests"]["high"])], dtype=np.float32)
            next_obs["throughput"] = np.array([self._normalize_obs(throughput, self.observation_config["throughput"]["low"], self.observation_config["throughput"]["high"])], dtype=np.float32)
            next_obs["average_execution_time"] = np.array([self._normalize_obs(avg_time, self.observation_config["average_execution_time"]["low"], self.observation_config["average_execution_time"]["high"])], dtype=np.float32)


            # for key, value in next_obs.items():
            #     print(f"{key}: Value={value}, Shape={value.shape}, Type={type(value)}")
            # calculate reward
            reward = self._calculate_reward(obs=next_obs, metadata=info)
            self.score += round(reward, 2)
            self._write_to_board(next_obs, action, reward, info, self.timestep, self.episode)

            state= self.simulator.get_state()

            state["requests"] = requests
            state["throughput"] = str(throughput)
            state["average_execution_time"] = str(avg_time)
            state["action"] = str(action)
            state["reward"] = str(reward)

            with open(f"{self.log_dir}/simulation_log.txt", "a") as file:
                file.write(json.dumps(state) + "\n")

            # counter for custom metrics
            self.timestep += 1
            done = self.time_step >= self.max_time
            if (self.timestep % 10 == 0):
                #done = True
                self.episode += 1
                self.loop = 0
                self.reward_history.append(self.score)
                with self.file_writer.as_default():
                    tf.summary.scalar('episodic_reward', self.score, self.episode)
                    tf.summary.scalar('mean_reward', np.mean(self.reward_history[-self.stats_window:]), self.episode)
                self.score = 0
                history = {'reward_history': self.reward_history,
                              'last_episode': self.episode}
                # write the reward history to a file
                with open(self._reward_file, "w") as outfile:
                    json.dump(history, outfile)
                
            self._last_obs = next_obs
            
            info['reward'] = reward
            info["current_time"] = self.timestep

            if "replicas" in next_obs.keys():
                del next_obs["replicas"] 

            return next_obs, reward, done, False, info

    def render(self):
        """function to render the environment
        """

        return
    
    def close(self):
        # close any open resources
        pass

    # set the verbosity level of the environment
    def set_verbosity(self, verbose):
        """set environment verbosity level

        Args:
            verbose (int): verbosity level
        """
        self._logger.verbose = verbose

