#!/usr/bin/env python3
import os
import argparse
import numpy as np
import tensorflow as tf

from stable_baselines import logger

from baselines.common.policies import MlpPolicy
from baselines.ppo.ppo2 import PPO2
from sim.run.ml_test import test
import baselines.common.util as U

from sim.vec_env import vec_env as wrapper
from sim.vec_env.vec_processor import VectorEnvironment


def main():

    args = {"train" : False, "model" : "models/2024-08-07-18-21-51_Iteration_1267.zip"} #extremely good model (dynamics 2, with L/sqrt(2)): models/2024-07-15-10-51-42_Iteration_11261.zip # working model: 2024-07-13-19-39-26_Iteration_2660.zip    2024-07-13-21-38-05_Iteration_1998.zip

    if args["train"]:
        args["n_envs"] = 100
    else:
        args["n_envs"] = 1


    env = wrapper.MainVecEnv(VectorEnvironment({"nenvs" : args["n_envs"]}))   

    
    if args["train"]:
        
        root = os.path.dirname(os.path.abspath(__file__))
        log_dir = root + '/models'
        saver = U.ConfigurationSaver(log_dir=log_dir)

        if args["model"] != "models/":
            model = PPO2.load(args["model"])
            model.set_env(env)
            
        else: 
            model = PPO2(
                tensorboard_log=saver.data_dir,
                policy=MlpPolicy,
                policy_kwargs=dict(
                    net_arch=[dict(pi=[128, 128], vf=[128, 128])], act_fun=tf.nn.relu),
                env=env,
                lam=0.95,
                gamma=0.99,
                n_steps=250,
                ent_coef=0.00,
                learning_rate=3e-4,
                vf_coef=0.5,
                max_grad_norm=0.5,
                nminibatches=1,
                noptepochs=10,
                cliprange=0.2,
                verbose=1,
            )

        logger.configure(folder=saver.data_dir)
        model.learn(
            total_timesteps=int(1000000000), # max timesteps
            log_dir=saver.data_dir, logger=logger)
        model.save(saver.data_dir)

    else:
        model = PPO2.load(args["model"])
        test(env, model)


if __name__ == "__main__":
    main()

