#!/usr/bin/env python3
import gym
import numpy as np
import time
import math


class EnvWrapper(gym.Env):
    def __init__(self, env):

        self.env = env
        self.env.init()
        self.num_obs = env.getObsDim()
        self.num_act = env.getActDim()

        self._observation_space = gym.spaces.Box(
            np.ones(self.num_obs) * -np.Inf,
            np.ones(self.num_obs) * np.Inf,
            dtype=np.float32)
        # the actions are eventually constrained by the action space.
        self._action_space = gym.spaces.Box(
            low=np.ones(self.num_act) * -1.,
            high=np.ones(self.num_act) * 1.,
            dtype=np.float32)
        self.observation = np.zeros(self.num_obs, dtype=np.float32)
        self.reward = np.float32(0.0)
        self.gatestate = np.array()
        self.done = False

        gym.Env.__init__(self)
        #
        self._max_episode_steps = 300

    def seed(self, seed=None):
        self.env.setSeed(seed)

    def step(self, action):
	    """
        pos_before = self.observation[:3]

        _ = self.env.step(action, self.observation)

        pos_now = self.obs()[:3]

        self.reward = self.rewardfunc(action, pos_before, pos_now)
	    """
        self.reward = self.env.step(action, self.observation)
        terminal_reward = 0.0
        self.done = self.env.isTerminalState(terminal_reward)
        self.reward -= terminal_reward
        return self.observation.copy(), self.reward, \
            self.done, [dict(reward_run=self.reward, reward_ctrl=0.0)]
    
    def rewardfunc(self, action, pos_before, pos_now):

        gate = self.getGateState() 
        g2 = gate[:3] # next gate
        alpha = gate[3]
        beta = gate[4]

        prev_gate_direction = np.array(math.cos(beta) + math.sin(beta), -math.sin(beta) + math.cos(beta))

        g1 = 10*prev_gate_direction + g2

        

        """
        ------------------------------------------------------------------ POSITIONAL REWARD ------------------------------------------------------------------------------------------------------------------------------------------

        Belohnung durch Bewegung entlang der Centerline, welche das vorherige und nachherige Gate verbindet (mit Beta ist diese Richtung gespeichert)

        Mittels Skalarprodukt wird ein Fortschritt an einer projizierten geraden Linie gemessen
        """

        norm_vec = np.linalg.norm(g2-g1) # Distanz für Normierung

        reward_ = ((pos_now[0] - g1[0])*(g2[0]-g1[0])+(pos_now[1]-g1[1])*(g2[1]-g1[1])+(pos_now[2]-g1[2])*(g2[2]-g1[2]))/norm_vec 
        rewardprev_ = ((pos_before[0] - g1[0])*(g2[0]-g1[0])+(pos_before[1]-g1[1])*(g2[1]-g1[1])+(pos_before[2]-g1[2])*(g2[2]-g1[2]))/norm_vec # Skalarprodukt der jeweiligen vorherigen und jetzigen Position

        pos_reward = reward_ - rewardprev_

        

        """
        ------------------------------------------------------------------ SAFETY REWARD ------------------------------------------------------------------------------------------------------------------------------------------------

        Belohnung durch genug weiten Abstand vom Äusseren des Gates --> Belohnung, umso näher es in der Mitte ist (ist wegen Sicherheitsgründen)

        Mittels Exponentialfunktion 
        """

        gate_normal_direction = np.array(math.cos(alpha) + math.sin(alpha), - math.sin(alpha) + math.cos(alpha), 0) # alpha ist mit Rotationsmatrix zusammen ausgerichtet für die Richtung

        D_ = - (gate_normal_direction[0]*g2[0] + gate_normal_direction[1]*g2[1] + gate_normal_direction[2]*g2[2])
        plane_of_gate = np.array(gate_normal_direction[0], gate_normal_direction[1], gate_normal_direction[2], D_) # Ebene von dem Gate, hier dargestellt als Koordinatengleichung; D_ noch berechnet

        gate_side_length = 1 # in Meter
        distance_max = 1
        distance_plane = abs((g2[0]*plane_of_gate[0] + g2[1]*plane_of_gate[1] + g2[2]*plane_of_gate[2] + plane_of_gate[3])/(math.sqrt(plane_of_gate[0]^2 + plane_of_gate[1]^2 + plane_of_gate[2]^2))) # Formel für Abstandberechnung von Punkt zu Ebene

        
        projection = ((gate_normal_direction[0]*pos_now[0] + gate_normal_direction[1]*pos_now[1] + gate_normal_direction[2]*pos_now[2] - gate_normal_direction[0]*g2[0] + gate_normal_direction[1]*g2[1] - gate_normal_direction[2]*g2[2]) 
                      /(gate_normal_direction[0]^2 + gate_normal_direction[1]^2 + gate_normal_direction[2]^2)) * gate_normal_direction + g2 # Abstandbestimmung, diese multipliziert und addiert mit zugehörigen Vektoren
        
        distance_to_normal = np.linalg.norm(pos_now - projection)

        f = 1 - (distance_plane/distance_max) # Safety-Reward soll nur zum Zuge kommen, wenn die Drohne genug nahe ist!

        if (f < 0): # untere Limite
            f = 0

        v = (1 - f)*(gate_side_length/5)
        
        if (v < 0.05): # untere Limite
            v = 0.05

        # Safetyrewardfunktion:

        safety_reward = -f * f * (1 - math.exp((-0.5 * distance_to_normal * distance_to_normal)/v))

        a, b = 10, 0.25 # Hyperparameter für Gesamt-Reward

        total_reward = a * pos_reward + b * safety_reward
        total_reward += 0.2

        return total_reward


    def reset(self):
        self.reward = 0.0
        self.env.reset(self.observation)
        return self.observation.copy()

    def reset_and_update_info(self):
        return self.reset(),

    def obs(self):
        self.env.getObs(self.observation)
        return self.observation

    def close(self,):
        return True

    def getQuadState(self,):
        quad_state = np.zeros(10, dtype=np.float32)
        self.env.getQuadState(quad_state)
        quad_correct = np.zeros(10, dtype=np.float32)
        quad_correct[0:3] = quad_state[0:3]
        quad_correct[3] = quad_state[9]
        quad_correct[4] = quad_state[6]
        quad_correct[5] = quad_state[7]
        quad_correct[6] = quad_state[8]
        quad_correct[7:10] = quad_state[3:6]
        return quad_correct

    def getGateState(self,):

        self.obs()

        self.gatestate = np.asarray(self.observation[12:])
        return self.gatestate
    
        #gate_state = np.zeros(9, dtype=np.float32)
        #self.env.getGateState(gate_state)
        #return gate_state

    def connectUnity(self):
        self.env.connectUnity()

    def disconnectUnity(self):
        self.env.disconnectUnity()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

# def main():
#   import os
#   from ruamel.yaml import YAML, dump, RoundTripDumper
#   cfg_path = os.path.abspath("../configs/env.yaml")
#   cfg = YAML().load(open(cfg_path, 'r'))
#   #
#   env = DynamicGate_v0(dump(cfg["env"], Dumper=RoundTripDumper))
#   env = EnvWrapper(env)

#   obs = env.reset()

#   obs = env.obs()
#   print(obs)

#   for i in range(10000):
#     act = np.array([10.0, 0.0, 0.0, 0.0])
#     next_obs, rew, done, _ = env.step(act)
#     time.sleep(0.01)

# if __name__ == "__main__":
#   main()
