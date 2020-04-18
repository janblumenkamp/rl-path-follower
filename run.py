import gym
from gym.spaces import Box
from sim_env import SimEnv, FeedbackNormalizedSimEnv
#from sim_env_nodrone import SimEnv, FeedbackNormalizedSimEnv
import tensorflow as tf

from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2, A2C
from stable_baselines.results_plotter import load_results, ts2xy

from stable_baselines.common.policies import FeedForwardPolicy, MlpLstmPolicy, MlpPolicy


import time
import os
import numpy as np

log_dir = "/auto/homes/jb2270/gym/0005"

def make_env(rank, seed=0):
    def _init():
        '''
        env = SimEnv({
            'render': True,
            'path_state_waypoints_lookahead': 10,
            'waypoints_lookahead_skip': 3,
            'ep_end_after_n_waypoints': 1000,
            'max_timesteps_between_checkpoints': 5000,
            'dist_waypoint_abort_ep': 5,
            'minimum_drone_height': 0.2,
            'dist_waypoint_proceed': 1.0,
        })
        '''
        env = FeedbackNormalizedSimEnv({
            'render': True,
            'path_state_waypoints_lookahead': 10,
            'waypoints_lookahead_skip': 3,
            'ep_end_after_n_waypoints': 1000,
            'max_timesteps_between_checkpoints': 5000,
            'dist_waypoint_abort_ep': 5,
            'minimum_drone_height': 0.2,
            'dist_waypoint_proceed': 1.0,
        })
        
        env = Monitor(env, None if rank == 0 else None, allow_early_resets=True)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

from stable_baselines.common.callbacks import BaseCallback

from stable_baselines.common.vec_env import VecEnvWrapper
import numpy as np
import time
from collections import deque
import os.path as osp
import json
import csv

import pybullet as p
distr = [[],[]]
def run():
    #model = PPO2.load("./results/end2end")
    model = PPO2.load("./results/feedbacknormalized")
    env = make_env(0)() #SubprocVecEnv([make_env(i) for i in range(1)])
    obs = env.reset()
    cum_rew = 0
    env.render()
    for i in range(20000):
        action, _states = model.predict(obs)
        distr[0].append(action[0])
        distr[1].append(action[1])
        obs, rewards, dones, info = env.step(action)
        #env.render()
        cum_rew += rewards
        p.resetDebugVisualizerCamera(5, 50, -35, env.drone.position, env.client)
        #f = env.render()
        #f.savefig(f"./results/render/{i:05d}")
        print(obs, rewards, cum_rew, dones, action)
        if dones:
            input("Done")
            import matplotlib.pyplot as plt
            plt.hist(distr[0])
            plt.show()
            plt.hist(distr[1])
            plt.show()
            env.reset()
            env.render()
            #break

    env.close()
    
if __name__ == '__main__':
    run()
    #train()
