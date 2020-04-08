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

    env = DemoEnv({
        "dim": 2,
        "waypoints": 4
    })

    try:
        obs = env.reset()
        cum_reward = 0
        for i in range(1000):
            goal_vec = -obs[:2]
            goal_d = np.linalg.norm(goal_vec, ord=2)
            if goal_d > 0:
                goal_vec /= np.abs(goal_vec).max()*3
            #print("a", goal_vec)
            #goal_vec /= 100

            obs, reward, done, _ = env.step(goal_vec)#env.action_space.sample())
            cum_reward += reward
            #print(obs)
            print(obs, reward, cum_reward, done)
            if done:                
                input(f"{i} Done")
                obs = env.reset()
                cum_reward = 0
                #break
    except Exception as e:
        del env
        print("Exit", e)
