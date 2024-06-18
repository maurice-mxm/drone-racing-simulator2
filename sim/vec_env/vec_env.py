import numpy as np
from gym import spaces
from stable_baselines.common.vec_env import VecEnv


"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                        vec_env.py
                                                                        ----------
                                        Core of all of the Vec Envs. Everything is distributed by this file 
                                        from the Algorithm to 'vec_processor.py', where n_envs are created.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""


class MainVecEnv(VecEnv):
    
    """
    Custom vectorized environment wrapper for Stable Baselines.
    """

    def __init__(self, impl):

        """
        Initializes the environment.

        Args:
            impl (callable): A callable implementing the environment with `step`, `reset`, 
                             and `getObsDim`/`getActDim` methods.
        """

        np.seterr(all='raise')

        self.wrapper = impl
        self.num_obs = self.wrapper.getObsDim()
        self.num_acts = self.wrapper.getActDim()
        self._observation_space = spaces.Box(
            np.ones(self.num_obs) * -np.Inf,
            np.ones(self.num_obs) * np.Inf, dtype=np.float32)
        self._action_space = spaces.Box(
            low=np.ones(self.num_acts) * -1., #-1
            high=np.ones(self.num_acts) * 1.,
            dtype=np.float32)
        self._observation = np.zeros([self.num_envs, self.num_obs],
                                     dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros((self.num_envs), dtype=np.bool)
        self.rewards = [[] for _ in range(self.num_envs)]

        self.max_episode_steps = 300

    def seed(self, seed=0):

        """
        Sets the seed for random number generation in the environment.

        Args:
            seed (int): The seed value for random number generation.
        """

        pass

    def step(self, action):

        """
        Executes a step in the environment.

        Args:
            action (np.ndarray): Action to be executed.

        Returns:
            tuple: Observation, reward, done flags, and additional information.
        """

        self._observation, self._reward, self._done = self.wrapper.step(action)

        info = [{} for i in range(self.num_envs)]

        for i in range(self.num_envs):
            self.rewards[i].append(self._reward[i])
            if self._done[i]:
                eprew = sum(self.rewards[i])
                eplen = len(self.rewards[i])
                epinfo = {"r": eprew, "l": eplen}
                info[i]['episode'] = epinfo
                self.rewards[i].clear()

        return self._observation.copy(), self._reward.copy(), \
            self._done.copy(), info.copy()


    def reset(self):

        """
        Resets the environment and returns the initial observations.

        Returns:
            np.ndarray: The initial observations of the environment.
        """

        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._observation = self.wrapper.reset()
        return self._observation.copy()

    def reset_and_update_info(self):

        """
        Resets the environment and updates episode information.

        Returns:
            tuple: Initial observations and updated episode information.
        """

        return self.reset(), self._update_epi_info()

    def _update_epi_info(self):

        """
        Updates the episode information for each environment.

        Returns:
            list: List of dictionaries containing episode information for each environment.
        """

        info = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            eprew = sum(self.rewards[i])
            eplen = len(self.rewards[i])
            epinfo = {"r": eprew, "l": eplen}
            info[i]['episode'] = epinfo
            self.rewards[i].clear()
        return info


    @property
    def num_envs(self):

        """
        Returns the number of parallel environments.

        Returns:
            int: Number of parallel environments.
        """

        return self.wrapper.getNumEnvs()

    @property
    def observation_space(self):

        """
        Returns the observation space of the environment.

        Returns:
            gym.spaces.Box: Observation space of the environment.
        """

        return self._observation_space

    @property
    def action_space(self):

        """
        Returns the action space of the environment.

        Returns:
            gym.spaces.Box: Action space of the environment.
        """

        return self._action_space
    
    ########################################################################################################################
    # not used Methods but have to be implemented so that it can be a Subclass of VecEnv

    def close(self):
        raise RuntimeError('This method is not implemented')
    

    def step_async(self):
        raise RuntimeError('This method is not implemented')
    
    
    def step_wait(self):
        raise RuntimeError('This method is not implemented')
    

    def get_attr(self, attr_name, indices=None):
        raise RuntimeError('This method is not implemented')
    

    def set_attr(self, attr_name, value, indices=None):
        raise RuntimeError('This method is not implemented')
    

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise RuntimeError('This method is not implemented')
    ############################################################################################################################
    


