import numpy as np
import random
from sim.dynamics import drone_dynamics
from sim.dynamics import drone_dynamics2
from numba import njit,int32, float64
from numba.typed import List
from numba.types import ListType
from numba.experimental import jitclass



"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                        vec_processor.py
                                                                        ----------------
                                        This is the Core of the Vec_Envs Initialisation. All of the "sub_envs" are 
                                        created and permanently updated by inputs of the PPO Algorithm. This is the 
                                        PROCESSOR CLASS of the Vec_Envs.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

spec = [
    ('gates', float64[:,:])  # A list of gates, each represented by a fixed-size array
]

@jitclass(spec)
class GateFinder:
    def __init__(self):
        self.gates = np.empty((0, 4), dtype=np.float64)  # Initial empty array with dtype float64

    def add_gate(self, position, properties):
        new_gate = np.array(position + properties, dtype=np.float64)
        if self.gates.size == 0:
            self.gates = new_gate.reshape(1, -1)
        else:
            self.gates = np.vstack((self.gates, new_gate.reshape(1, -1)))

    def closest_gate_pair(self, object_position):
        min_distance = np.inf
        closest_pair_index = -1
        closest_point = np.zeros(3, dtype=np.float64)

        def distance_point_to_segment(p, a, b):
            """Calculate the distance from point p to the segment defined by points a and b."""
            ab = b - a
            ap = p - a
            ab_dot_ab = np.dot(ab, ab)
            if ab_dot_ab == 0:
                return np.linalg.norm(p - a), a
            
            t = np.dot(ap, ab) / ab_dot_ab
            t = self.clip(t, 0, 1)  # Manual clipping
            nearest = a + t * ab
            return np.linalg.norm(nearest - p), nearest

        object_pos = np.array(object_position, dtype=np.float64)

        for i in range(len(self.gates) - 1):
            gate1 = self.gates[i, :3]
            gate2 = self.gates[i + 1, :3]

            distance, nearest_point = distance_point_to_segment(object_pos, gate1, gate2)
            if distance < min_distance:
                min_distance = distance
                closest_pair_index = i
                closest_point = nearest_point

        return closest_pair_index, closest_point, min_distance

    def move_closest_pair_to_front(self, object_position):
        closest_pair_index, _, _ = self.closest_gate_pair(object_position)
        if closest_pair_index != -1:
            self.gates = self.roll(self.gates, -closest_pair_index)

    def next(self):
        self.gates = self.roll(self.gates, -1)

    def clip(self, value, min_value, max_value):
        """Manual clipping function."""
        if value < min_value:
            return min_value
        elif value > max_value:
            return max_value
        else:
            return value

    def roll(self, array, shift):
        """Manually roll the array."""
        n = array.shape[0]
        shift = shift % n
        return np.concatenate((array[-shift:], array[:-shift]), axis=0)




class VectorEnvironment:
    

    def __init__(self, config):

        """
        Initializes the VectorEnvironment class.
        Loads the configuration file and initializes the environment.

        Args:
            config (dict): Configuration for the given Vec_Env. Containing:
            - Number of Envs 
            - ...
        """

        # Extract number of Envs
        self.number_of_envs = config["nenvs"]    

        # Create environments
        self.envs = [BaseEnvironment() for _ in range(self.number_of_envs)]

        # Set observation and action dimensions
        self.obs_dim = self.envs[0].get_observation_dimension()
        self.act_dim = self.envs[0].get_action_dimension()

            


    def getObsDim(self):

        """
        Returns the dimension of the observation space.

        Returns:
            int: Dimension of the observation space.
        """

        return self.obs_dim
    
    
    def getActDim(self):

        """
        Returns the dimension of the action space.

        Returns:
            int: Dimension of the action space.
        """

        return self.act_dim
    
    
    def getNumEnvs(self):

        """
        Returns the number of environments.

        Returns:
            int: Number of environments.
        """

        return self.number_of_envs
    
    
    def reset(self, seed=0):

        """
        Resets all environments and returns the initial observations.

        Args:
            seed (int, optional): Random seed. Default is 0.

        Returns:
            np.ndarray: Initial observations of all environments.
        """

        observations_ = np.zeros((self.number_of_envs, self.obs_dim), dtype=np.float32)
        for i in range(self.number_of_envs):
            observations_[i] = self.envs[i].reset()

        return observations_

    def step(self, act):

        """
        Executes a step in all environments and returns the results (observations) generated by Dynamics.

        Args:
            actions (np.ndarray): Actions for all environments.

        Returns:
            tuple: Observations, rewards, done flags, and extra info.
        """

        observation_ = np.zeros((self.number_of_envs, self.obs_dim), dtype=np.float32)
        reward_ = np.zeros(self.number_of_envs, dtype=np.float32)
        done_ = np.zeros(self.number_of_envs, dtype=bool)
        
        for i in range(self.number_of_envs):
            observation_[i], reward_[i], done_[i]= self.step_env(i, act[i])

        return observation_, reward_, done_
    

    def step_env(self, env_id, action):

        """
        Executes a step for a single environment.

        Args:
            env_id (int): Environment ID.
            action (np.ndarray): Action for the environment.

        Returns:
            tuple: Observation, reward, done flag, and extra info.
        """

        observation, reward, done = self.envs[env_id].step(action)

        if done:
            observation = self.envs[env_id].reset()
            reward -= 100

        return observation, reward, done


    def get_observation(self, observations):

        """
        Retrieves current observations from all environments.

        Args:
            observations (np.ndarray): Array to store observations.
        """

        observations_ = np.zeros((self.number_of_envs, self.obs_dim), dtype=np.float32)
        for i in range(self.number_of_envs):
            observations_[i] = self.envs[i].get_observation()

        return observations_



class BaseEnvironment:

    def __init__(self):

        """
        Initializes the BaseEnvironment class. This is where the Dynamics are directly implemented.
        """

        #self.initial_state = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.state = np.empty(16)

        for i in range(12):
            if i == 0:
                self.state[i] = random.uniform(-3.5, 3.5) 
            
            elif i == 1:
                self.state[i] = random.uniform(-3.5, 3.5) 

            elif i == 2:
                self.state[i] = random.uniform(1.0, 2.1) 

            else:
                self.state[i] = random.uniform(-0.001, 0.001)
        #self.state = np.array([2.0, 0.0, 1.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        
        self.obs_dim = 13
        self.act_dim = 4
        self.old_control = np.array([0.0, 0.0, 0.0, 0.0])
        #self.initial_quat = drone_dynamics.euler_to_quaternion(self.state[6], self.state[7], self.state[8])
        #self.quaternion = self.initial_quat.copy()


        self.gates = GateFinder()
        
        #for i in range(8):
        #    x, y, z, k = random.uniform(-2.5, 2.5), random.uniform(-2.5, 2.5), 1.0, random.choice([-2.0, -1.0, 1.0, 2.0])
        #    self.gates.add_gate([x, y, z], [k])

        self.gates.add_gate([-3.0, 3.0, 1.0], [1.0]) # 1.0 (x up), -1.0 (x down), 2.0 (y up), -2.0 (y down)
        self.gates.add_gate([3.0, 3.0, 1.0], [-2.0])
        self.gates.add_gate([2.0, -2.0, 2.0], [-1.0])
        self.gates.add_gate([-2.0, -3.0, 2.0], [-1.0])
        self.gates.add_gate([-2.0, -3.0, 1.0], [1.0])
        self.gates.add_gate([0.0, 0.0, 1.0], [-1.0])
        self.gates.add_gate([-2.0, -2.0, 1.0], [-1.0])
        #self.gates.add_gate([-3.0, -3.0, 1.0], [2.0])

        self.gates.move_closest_pair_to_front([self.state[0], self.state[1], self.state[2]])

        #for i in range(12, 15):
        #    self.state[i] = self.gates.gates[1][i-12] - self.state[i-12]

        #for i in range(15, 18):
        #    self.state[i] = self.gates.gates[2][i-15] - self.state[i-15]
        
        self.dic = {-2.0: np.array([0.0, -1.0, 0.0]), -1.0: np.array([-1.0, 0.0, 0.0]), 1.0: np.array([1.0, 0.0, 0.0]), 2.0: np.array([0.0, 1.0, 0.0])}
        """
        self.vec = np.empty(3)

        for i in range(3):
            x, y, z, k = random.uniform(-3.5, 3.5), random.uniform(-3.5, 3.5), random.choice([1.0, 2.0]), random.choice([-2.0, -1.0, 1.0, 2.0])
            self.gates.add_gate([x, y, z], [k])

        for i in range(12, 15):
            self.state[i] = - self.gates.gates[1][i-12] + self.state[i-12]
            self.vec[i-12] = self.state[i]"""
        


        vec = np.empty(3)

        for i in range(12, 15):
            vec[i-12] = self.gates.gates[1][i-12] - self.state[i-12]
            
        r = np.linalg.norm(vec)
        theta = np.arctan2(vec[1], vec[0])
        phi = np.arccos(vec[2]/r)

        self.state[12], self.state[13], self.state[14] = r, theta, phi


        self.state[15] = np.arccos((np.dot(vec, self.dic[self.gates.gates[1][3]])/(np.linalg.norm(vec)*np.linalg.norm(self.dic[self.gates.gates[1][3]])))) # alpha

        
        self.initial_state = self.state
        #print(self.initial_state)

        self.return_state = np.empty(13)


    def get_observation_dimension(self):

        """
        Returns the dimension of the observation space.

        Returns:
            int: Dimension of the observation space.
        """

        return self.obs_dim

    def get_action_dimension(self):

        """
        Returns the dimension of the action space.

        Returns:
            int: Dimension of the action space.
        """

        return self.act_dim

    def reset(self):

        """
        Resets the environment and returns the last State.

        Returns:
            np.ndarray: Sate of the environment.
        """

        #self.gates = GateFinder()

        self.state = np.empty(16)
        for i in range(12):
            if i == 0:
                self.state[i] = random.uniform(-3.5, 3.5) 
            
            elif i == 1:
                self.state[i] = random.uniform(-3.5, 3.5) 

            elif i == 2:
                self.state[i] = random.uniform(1.0, 2.1) 

            else:
                self.state[i] = random.uniform(-0.001, 0.001)

        #self.state = np.array([2.0, 0.0, 1.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


        #self.quaternion = self.initial_quat.copy()


        #for i in range(8):
        #    x, y, z, k = random.uniform(-2.5, 2.5), random.uniform(-2.5, 2.5), 1.0, random.choice([-2.0, -1.0, 1.0, 2.0])
        #    self.gates.add_gate([x, y, z], [k])
        
        self.gates.move_closest_pair_to_front([self.state[0], self.state[1], self.state[2]])
        
        #for i in range(12, 15):
        #    self.state[i] = self.gates.gates[1][i-12] - self.state[i-12]

        #for i in range(15, 18):
        #    self.state[i] = self.gates.gates[2][i-15] - self.state[i-15]
        

        """for i in range(3):
            x, y, z, k = random.uniform(-3.5, 3.5), random.uniform(-3.5, 3.5), random.choice([1.0, 2.0]), random.choice([-2.0, -1.0, 1.0, 2.0])
            self.gates.add_gate([x, y, z], [k])"""
        
        vec = np.empty(3)

        for i in range(12, 15):
            vec[i-12] = self.gates.gates[1][i-12] - self.state[i-12]
            
        r = np.linalg.norm(vec)
        theta = np.arctan2(vec[1], vec[0])
        phi = np.arccos(vec[2]/r)

        self.state[12], self.state[13], self.state[14] = r, theta, phi

        self.state[15] = np.degrees(np.arccos((np.dot(vec, self.dic[self.gates.gates[1][3]])/(np.linalg.norm(vec)*np.linalg.norm(self.dic[self.gates.gates[1][3]]))))) # alpha


        self.return_state = self.state[3:]

        

        return self.return_state

    def step(self, act):

        """
        Executes a step in the environment and updates the state.

        Args:
            action (np.ndarray): Action to be performed.

        Returns:
            tuple: New state and reward.

        """

        #print(self.state)

        self.state, reward, done, next, good = drone_dynamics2.dynamics(self.state, act, self.old_control, self.gates.gates[0][0:3], self.gates.gates[0][3], self.gates.gates[1][0:3], self.gates.gates[1][3], self.gates.gates[2][0:3], self.gates.gates[2][3], self.dic[self.gates.gates[1][3]])

        self.old_control = act.copy()

        self.return_state = self.state[3:]

        if next:
            self.gates.next()

        #if next and not done:
        #    self.state = self.initial_state
        #    print(self.state)
        #elif next and not good and not done:
        #    done = True
        print(self.state[:3].tolist(), ",")

        return self.return_state, reward, done

    def is_terminal_state(self):

        """
        Checks if the environment is in a terminal state. Not used at the time given, directy implemented in dynamics (see step_env())

        Returns:
            bool: True if in a terminal state, False otherwise.
        """

        return self.state[0] > 3 or self.state[0] < -3 or self.state[1] > 3 or self.state[1] < -3 or self.state[2] > 6 or self.state[2] < 0.02


    def get_observation(self, observation):

        """
        Returns the current observation of the environment.

        Args:
            observation (np.ndarray): Array to store the observation.

        Returns:
            np.ndarray: The current state of the environment.
        """

        return self.state

