import os
import yaml
import numpy as np
import logging
from threading import Thread
import main as dynamics
import math


class VecEnv:
    def __init__(self, config_path=None):
        if config_path is None:
            flightmare_path = os.getenv("FLIGHTMARE_PATH")
            config_path = os.path.join(flightmare_path, "flightlib/configs/vec_env.yaml")
        
        if isinstance(config_path, str):
            with open(config_path, 'r') as file:
                self.cfg = yaml.safe_load(file)
        elif isinstance(config_path, dict):
            self.cfg = config_path
        else:
            raise ValueError("Unsupported config format")

        self.init()

    def init(self):
        self.unity_render = self.cfg["env"]["render"]
        self.seed = self.cfg["env"]["seed"]
        self.num_envs = self.cfg["env"]["num_envs"]
        self.scene_id = self.cfg["env"]["scene_id"]

        self.envs = [EnvBase() for _ in range(self.num_envs)]

        self.obs_dim = self.envs[0].get_obs_dim()
        self.act_dim = self.envs[0].get_act_dim()

        self._extraInfoNames = [] #list(self.envs[0].update_extra_info())

    def getObsDim(self):
        return self.obs_dim
    
    def getActDim(self):
        return self.act_dim
    
    def getNumEnvs(self):
        return self.num_envs
    
    def getExtraInfoNames(self):
        return self._extraInfoNames

    def reset(self, seed=0):
        """if obs.shape != (self.num_envs, self.obs_dim):
            logging.error("Input matrix dimensions do not match with that of the environment.")
            return False

        self.receive_id = 0"""
        observation_ = np.zeros([self.num_envs, self.obs_dim],
                                     dtype=np.float32)

        for i in range(self.num_envs):
            observation_[i] = self.envs[i].reset()

        return observation_

    def step(self, act):
        """if (act.shape != (self.num_envs, self.act_dim) or
                obs.shape != (self.num_envs, self.obs_dim) or
                reward.shape != (self.num_envs,) or
                done.shape != (self.num_envs,) or
                extra_info.shape != (self.num_envs, len(self.extra_info_names))):
            logging.error("Input matrix dimensions do not match with that of the environment.")
            return False

        threads = []
        for i in range(self.num_envs):
            thread = Thread(target=self.per_agent_step, args=(i, act, obs, reward, done, extra_info))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

        if self.unity_render and self.unity_ready:
            self.unity_bridge.get_render(0)
            self.unity_bridge.handle_output()"""
        
        observation_ = np.zeros([self.num_envs, self.obs_dim],
                                     dtype=np.float32)
        reward_ = np.zeros(self.num_envs, dtype=np.float32)
        done_ = np.zeros((self.num_envs), dtype=np.bool)
        extraInfo_ = np.zeros([self.num_envs,
                                    len(self._extraInfoNames)], dtype=np.float32)
        
        for i in range(self.num_envs):
            observation_[i], reward_[i], done_[i], extraInfo_[i] = self.per_agent_step(i, act[i])


        return observation_, reward_, done_, extraInfo_
    

    def test_step(self, act, obs, reward, done, extra_info):
        self.per_agent_step(0, act, obs, reward, done, extra_info)
        self.envs[0].get_obs(obs[0])

    def close(self):
        for env in self.envs:
            env.close()

    def set_seed(self, seed):
        seed_inc = seed
        for env in self.envs:
            env.set_seed(seed_inc)
            seed_inc += 1

    def get_obs(self, obs):
        for i in range(self.num_envs):
            self.envs[i].get_obs(obs[i])

    def get_episode_length(self):
        if not self.envs:
            return 0
        else:
            return int(self.envs[0].t)

    def per_agent_step(self, agent_id, act):

        obs, reward = self.envs[agent_id].step(act)

        done = self.envs[agent_id].is_terminal_state()

        for i in range(len(obs)):
            if math.isnan(obs[i]):
                #print('done?')
                done = True

        extra_info = self.envs[agent_id].update_extra_info()
        for j, name in enumerate(self._extraInfoNames):
            extra_info[j] = self.envs[agent_id].extra_info[name]
        
        if done:

            obs = self.envs[agent_id].reset()
            reward -= 0.2

        return obs, reward, done, extra_info

    def curriculum_update(self):
        for env in self.envs:
            env.curriculum_update()


class EnvBase:
    def __init__(self):
        self.initial_state = np.asarray([0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.state = self.initial_state.copy()
        self.obs_dim = 12
        self.act_dim = 4


    def get_obs_dim(self):
        return self.obs_dim

    def get_act_dim(self):
        return self.act_dim

    def reset(self):
        #old_state = self.state.copy()
        self.state = self.initial_state.copy()

        return self.state

    def step(self, act):

        self.state, reward = dynamics.dynamics(self.state, act)
        #print('act:', act, 'obs:', self.state)

        return self.state, reward
	
    def is_terminal_state(self):
        x, y, z = self.state[0], self.state[1], self.state[2]

        if x > 3 or x < -3 or y > 3 or y < -3 or z > 8 or z < 0.02:
            return True
        else:
            return False

    def update_extra_info(self):
        pass

    def get_obs(self, obs):
        return self.state

    def close(self):
        pass

    def set_seed(self, seed):
        pass

    def get_max_t(self):
        return 5

    def get_sim_time_step(self):
        pass

    def add_objects_to_unity(self, unity_bridge):
        pass

    def curriculum_update(self):
        pass
