import numpy as np
import gym
import pybullet as p
import pybullet_data
from gym.spaces import Box
from pybullet_multicopter.copters.quadcopter import Quadcopter
#from sim_env_nodrone import Quadcopter
from path_generator import PathGenerator

class SimEnv(gym.Env):
    def __init__(self, config):
        self.cfg = config
        self.action_space = Box(-1, 1, shape=(3,), dtype=np.float32)
        self.observation_space = Box(-np.inf, np.inf, shape=(3*self.cfg['path_state_waypoints_lookahead'],), dtype=np.float32)
        self.client = p.connect(p.GUI if self.cfg['render'] else p.DIRECT)
        p.setGravity(0, 0, -10, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)

        self.drone = Quadcopter(self.client)
        self.path_generator = PathGenerator()

        self.reset()

    def __del__(self):
        p.disconnect()

    def get_simple_rotation_matrix(self):
        s, c = np.sin(self.drone.orientation_euler[2]), np.cos(self.drone.orientation_euler[2])
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]])

    def map_action(self, value, in_min, in_max, out_min, out_max):
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def get_drone_pose(self):
        return np.array(list(self.drone.position)+[self.drone.orientation_euler[2]])

    def _reset(self):
        self.drone.reset()
        self.waypoints_lookahead_amount = self.cfg['path_state_waypoints_lookahead'] * self.cfg['waypoints_lookahead_skip']
        self.generated_path_len = 20
        while True:
            self.waypoints = self.path_generator.get_path(self.get_drone_pose(), self.generated_path_len)
            if len(self.waypoints) >= self.cfg['ep_end_after_n_waypoints']+self.waypoints_lookahead_amount:
                self.waypoints = self.waypoints[:self.cfg['ep_end_after_n_waypoints']+self.waypoints_lookahead_amount]
                break
            else:
                # increase length of generated path
                self.generated_path_len *= 2
        self.current_waypoint_index = 0
        self.timestep = 0
        self.last_timestep_new_waypoint = 0

    def reset(self):
        self._reset()
        return self.step([0,0,0])[0]

    def internal_step(self, u, w, h):
        self.timestep += 1

        self.drone.step_speed(u, 0, h)
        self.drone.set_yaw_rate(w)
        self.drone.step_angle()
        self.drone.step()
        p.stepSimulation(physicsClientId=self.client)

        reward = 0
        current_waypoint_rel = (self.waypoints[self.current_waypoint_index] - self.drone.position)

        dist_to_current = np.linalg.norm(current_waypoint_rel, ord=2)
        done = dist_to_current > self.cfg['dist_waypoint_abort_ep'] or (self.timestep - self.last_timestep_new_waypoint) > self.cfg['max_timesteps_between_checkpoints'] or self.drone.position[2] < self.cfg['minimum_drone_height']
        if dist_to_current < self.cfg['dist_waypoint_proceed']:
            reward = 1
            self.last_timestep_new_waypoint = self.timestep
            if self.current_waypoint_index < self.cfg['ep_end_after_n_waypoints']:
                self.current_waypoint_index += 1
            else:
                done = True
        m = self.get_simple_rotation_matrix() #self.drone.get_rotation_matrix()
        state = (np.tensordot(
            self.waypoints[
                self.current_waypoint_index
                :self.current_waypoint_index+self.waypoints_lookahead_amount
                :self.cfg['waypoints_lookahead_skip']
            ] - self.drone.position, m, axes=([1],[0])
        ) - np.array([0,0,0])).flatten()

        return state, reward, done, {}

    def step(self, action):
        assert(not any(np.isnan(action)) and all(np.array(action) <= 1) and all(np.array(action) >= -1))
        u = self.map_action(action[0], -1, 1, -1, 3)
        w = self.map_action(action[1], -1, 1, -4*np.pi, 4*np.pi)
        h = self.map_action(action[2], -1, 1, -1, 1)
        return self.internal_step(u, w, h)

    def render(self, _=None):
        for i in range(1, len(self.waypoints)):
            p.addUserDebugLine(self.waypoints[i-1], self.waypoints[i], lineColorRGB=[1,0,0], lineWidth=3, physicsClientId=self.client)

class FeedbackNormalizedSimEnv(SimEnv):
    def __init__(self, cfg):
        SimEnv.__init__(self, cfg)
        self.action_space = Box(-1, 1, shape=(2,), dtype=float)

    def feedback_linearized(self, orientation, velocity, epsilon):
        u = velocity[0]*np.cos(orientation) + velocity[1]*np.sin(orientation)  # [m/s]
        w = (1/epsilon)*(-velocity[0]*np.sin(orientation) + velocity[1]*np.cos(orientation))  # [rad/s] going counter-clockwise.
        return u, w

    def reset(self):
        super()._reset()
        return self.step([1,0, 0])[0]

    def step(self, action):
        assert(not any(np.isnan(action)) and all(np.array(action) <= 1) and all(np.array(action) >= -1))
        epsilon = self.map_action(action[0], -1, 1, 0.1, 1.5)
        gamma = self.map_action(action[1], -1, 1, 0.1, 10.0)
        kappa_h = 1#mapping(action[1], -0.5, 0.5)
        #assert(epsilon >= 0.05 and epsilon <= self.cfg['dist_waypoint_proceed']+0.05)
        #assert(gamma >= 0.1 and gamma <= 2.1)
        #assert(kappa_h >= -1 and kappa_h <= 1)

        orientation = self.drone.orientation_euler[2]
        next_pos = (self.waypoints[self.current_waypoint_index] - self.drone.position)
        position = np.array([
            epsilon * np.cos(orientation),
            epsilon * np.sin(orientation),
        ], dtype=np.float32)
        v = (next_pos[:2] - position)*gamma
        h = next_pos[2]*kappa_h
        u, w = self.feedback_linearized(orientation, v, epsilon=epsilon)
        return super().internal_step(u, w, h)
