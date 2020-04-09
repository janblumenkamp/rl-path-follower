import numpy as np
import gym
import pybullet as p
import pybullet_data
from gym.spaces import Box
from pybullet_multicopter.copters.quadcopter import Quadcopter

class SimEnv(gym.Env):
    def __init__(self, config):
        self.cfg = config
        self.action_space = Box(-12, 12, shape=(3,), dtype=float)
        self.observation_space = Box(-np.inf, np.inf, shape=(6,), dtype=np.float32) # current destination, next destination

        self.client = p.connect(p.GUI if self.cfg['render'] else p.DIRECT)
        p.setGravity(0, 0, -10, physicsClientId=self.client) 
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)

        self.drone = Quadcopter(self.client)
        
        self.reset()

    def __del__(self):
        p.disconnect()

    def reset(self):
        self.drone.reset()
        rand_box = lambda s: np.random.uniform(-s, s, 3) + np.array([0, 0, s])
        self.waypoints = np.array([rand_box(3) for _ in range(self.cfg['waypoints'])])
        self.current_waypoint_index = 0
        return self.step([0,0,0])[0]

    def step(self, action):
        
        self.drone.step_speed(action[0], -action[1], action[2])
        self.drone.step()
        p.stepSimulation(physicsClientId=self.client)

        m = np.array(p.getMatrixFromQuaternion(self.drone.orientation)).reshape(3,3)
                
        reward = 0
        current_waypoint_rel = self.drone.position - self.waypoints[self.current_waypoint_index] @ m
        next_waypoint_index = self.current_waypoint_index
        if next_waypoint_index < len(self.waypoints)-1:
            next_waypoint_index += 1
        next_waypoint_rel = self.drone.position - self.waypoints[next_waypoint_index] @ m
        
        dist_to_current = np.linalg.norm(current_waypoint_rel, ord=2)
        done = dist_to_current > 10
        if dist_to_current < 0.5:
            reward = 10
            if self.current_waypoint_index < len(self.waypoints) - 1:
                self.current_waypoint_index += 1
            else:
                done = True

        return np.concatenate([current_waypoint_rel, next_waypoint_rel], axis=0), reward, done, {}

    def render(self):
        for i in range(1, len(self.waypoints)):
            p.addUserDebugLine(self.waypoints[i-1], self.waypoints[i], lineColorRGB=[1,0,0], lineWidth=3, physicsClientId=self.client)
            p.addUserDebugText(str(i), self.waypoints[i-1], textColorRGB=[1,0,0])
        
