import numpy as np  
import gym
from gym.spaces import Box

class DemoEnv(gym.Env):
    def __init__(self, config):
        self.action_space = Box(-np.inf, np.inf, shape=(3,))
        self.observation_space = Box(-np.inf, np.inf, shape=(3,))
        
        self.reset()

    def reset(self):
        self.last_pos = self.pos = np.array([0.0,0.0,0.0])
        rand_box = lambda s: np.random.uniform(-s, s, 3) + np.array([0, 0, 0.5 + s])
        self.goal = rand_box(2)
        self.last_dist = np.linalg.norm(self.pos - self.goal, ord=2)
        self.cnt_timesteps_goal_reached = 0
        return self.step([0,0,0])[0]

    def step(self, action):
        self.pos += np.clip(action, -0.5, 0.5)
        #self.speed = (self.pos - self.last_pos)*240 # Pybullet timestep length
        #self.last_pos = self.pos.copy()

        reward = 0
        dist_to_goal = np.linalg.norm(self.pos - self.goal, ord=2)
        #reward = self.last_dist - dist_to_goal
        #self.last_dist = dist_to_goal
        if dist_to_goal < 0.5:
            self.cnt_timesteps_goal_reached += 1
            reward = 5
        else:
            self.cnt_timesteps_goal_reached = 0
            reward = -0.1

        done = dist_to_goal > 10 or self.cnt_timesteps_goal_reached > 30
        state = self.pos - self.goal#np.concatenate([self.pos - self.goal, self.speed], axis=0)
        return state, reward, done, {}

