import numpy as np
import gym
import pybullet as p
import pybullet_data
from gym.spaces import Box
from pybullet_multicopter.copters.quadcopter import Quadcopter
from path_generator import PathGenerator

class SimEnv(gym.Env):
    def __init__(self, config):
        self.cfg = config
        self.action_space = Box(-np.inf, np.inf, shape=(3,), dtype=float)
        self.observation_space = Box(-np.inf, np.inf, shape=(7,), dtype=np.float32) # orientation, relative current destination, next destination
        self.path_track_length_episode = 100
        self.client = p.connect(p.GUI if self.cfg['render'] else p.DIRECT)
        p.setGravity(0, 0, -10, physicsClientId=self.client) 
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)

        self.drone = Quadcopter(self.client)
        CONFIG_SPACE_SIZE = 4
        self.path_generator = PathGenerator(np.array([[-CONFIG_SPACE_SIZE, CONFIG_SPACE_SIZE], [-CONFIG_SPACE_SIZE, CONFIG_SPACE_SIZE], [1, CONFIG_SPACE_SIZE]]))

        self.reset()

    def __del__(self):
        p.disconnect()

    def get_drone_pose(self):
        return np.array(list(self.drone.position)+[self.drone.orientation_euler[2]])

    def _reset(self):
        self.drone.reset()
        while True:
            self.waypoints = self.path_generator.get_path(self.get_drone_pose(), self.path_generator.sample_configuration_space())
            if len(self.waypoints) >= self.path_track_length_episode:
                break
        self.current_waypoint_index = 0
        self.next_waypoint_index = 1
        
    def reset(self):
        self._reset()
        return self.step([0,0,0])[0]

    def step(self, action):
        assert(not any(np.isnan(action)))
        self.drone.step_speed(action[0], 0, action[1])
        self.drone.set_yaw(action[2])
        self.drone.step()
        p.stepSimulation(physicsClientId=self.client)

        reward = 0
        current_waypoint_rel = (self.waypoints[self.current_waypoint_index] - self.drone.position)
        self.next_waypoint_index = self.current_waypoint_index
        if self.next_waypoint_index < len(self.waypoints)-1:
            self.next_waypoint_index += 1
        next_waypoint_rel = (self.waypoints[self.next_waypoint_index] - self.drone.position)
        
        dist_to_current = np.linalg.norm(current_waypoint_rel, ord=2)
        done = dist_to_current > 2
        if dist_to_current < 0.1:
            reward = 1
            if self.current_waypoint_index < self.path_track_length_episode:
                self.current_waypoint_index += 1
            else:
                done = True

        return np.concatenate([[self.drone.orientation_euler[2]], current_waypoint_rel, next_waypoint_rel], axis=0), reward, done, {}

    def render(self):
        for i in range(1, len(self.waypoints)):
            p.addUserDebugLine(self.waypoints[i-1], self.waypoints[i], lineColorRGB=[1,0,0], lineWidth=3, physicsClientId=self.client)

class FeedbackNormalizedSimEnv(SimEnv):
    def __init__(self, cfg):
        SimEnv.__init__(self, cfg)
        self.action_space = Box(0.001, np.inf, shape=(2,), dtype=float)

    def feedback_linearized(self, orientation, velocity, epsilon):
        u = velocity[0]*np.cos(orientation) + velocity[1]*np.sin(orientation)  # [m/s]
        w = (1/epsilon)*(-velocity[0]*np.sin(orientation) + velocity[1]*np.cos(orientation))  # [rad/s] going counter-clockwise.
        return u, w

    def reset(self):
        super()._reset()
        return self.step([0.1,1])[0]

    def step(self, action):
        orientation = self.drone.orientation_euler[2]
        next_pos = (self.waypoints[self.next_waypoint_index] - self.drone.position)
        
        position = np.array([
            action[0] * np.cos(orientation),
            action[0] * np.sin(orientation),
            0], dtype=np.float32)
        v = (next_pos - position)*action[1]
        u, w = self.feedback_linearized(orientation, v, epsilon=action[0])
        h = v[2]

        return super().step(np.array([u, h, w]))
