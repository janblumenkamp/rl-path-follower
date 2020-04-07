import time
import numpy as np  
import yaml
import ray
from ray import tune
from ray.tune.registry import register_env
import pybullet as p
import gym
from gym.spaces import Box
from demo_env import DemoEnv

if __name__ == '__main__':

    env = DemoEnv({})

    try:
        obs = env.reset()
        cum_reward = 0
        for i in range(100):
            goal_vec = -obs[:3]
            goal_d = np.linalg.norm(goal_vec, ord=2)
            if goal_d > 0:
                goal_vec /= np.abs(goal_vec).max()
            print("a", goal_vec)
            #goal_vec /= 100
            if goal_d < 0.5:
                goal_vec /= 5
            obs, reward, done, _ = env.step(goal_vec)#env.action_space.sample())
            cum_reward += reward
            print(obs, reward, cum_reward, done)
            if done:                
                input(f"{i} Done")
                obs = env.reset()
                cum_reward = 0
                #break
    except Exception as e:
        del env
        print("Exit", e)
