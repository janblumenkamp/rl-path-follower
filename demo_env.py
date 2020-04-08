import numpy as np  
import gym
from gym.spaces import Box

class DemoEnv(gym.Env):
    def __init__(self, config):
        self.cfg = config
        self.action_space = Box(-np.inf, np.inf, shape=(self.cfg['dim'],))
        self.observation_space = Box(-np.inf, np.inf, shape=(2 * self.cfg['dim'],)) # current and next waypoint
        
        self.reset()

    def reset(self):
        self.pos = np.array([0.0]*self.cfg['dim'])
        rand_box = lambda s: np.random.uniform(-s, s, self.cfg['dim'])
        self.waypoints = np.array([rand_box(3) for _ in range(self.cfg['waypoints'])])
        self.current_waypoint_index = 0
        return self.step([0]*self.cfg['dim'])[0]

    def step(self, action):
        self.pos += np.clip(action, -0.5, 0.5)

        reward = 0
        current_waypoint_rel = self.pos - self.waypoints[self.current_waypoint_index]
        next_waypoint_index = self.current_waypoint_index
        if next_waypoint_index < len(self.waypoints)-1:
            next_waypoint_index += 1
        next_waypoint_rel = self.pos - self.waypoints[next_waypoint_index]
        
        dist_to_current = np.linalg.norm(current_waypoint_rel, ord=2)
        done = dist_to_current > 10
        if dist_to_current < 0.5:
            reward = 10
            if self.current_waypoint_index < len(self.waypoints) - 1:
                self.current_waypoint_index += 1
            else:
                done = True

        state = np.concatenate([current_waypoint_rel, next_waypoint_rel], axis=0)
        return state, reward, done, {}

