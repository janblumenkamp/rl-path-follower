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
        self.action_space = Box(-1, 1, shape=(3,), dtype=np.float32)
        self.observation_space = Box(-np.inf, np.inf, shape=(6,), dtype=np.float32) # orientation, relative current destination, next destination
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

    def get_drone_pose(self):
        return np.array(list(self.drone.position)+[self.drone.orientation_euler[2]])

    def _reset(self):
        self.drone.reset()
        while True:
            self.waypoints = self.path_generator.get_path(self.get_drone_pose())
            if len(self.waypoints) >= self.cfg['ep_end_after_n_waypoints']:
                break
        self.current_waypoint_index = 0
        self.next_waypoint_index = 1
        self.timestep = 0
        self.last_timestep_new_waypoint = 0

    def reset(self):
        self._reset()
        return self.step([0,0,0])[0]

    def step(self, action):
        assert(not any(np.isnan(action)))
        action *= np.array([1, 0.5, 1])
        self.timestep += 1

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
        done = dist_to_current > self.cfg['dist_waypoint_abort_ep'] or (self.timestep - self.last_timestep_new_waypoint) > self.cfg['max_timesteps_between_checkpoints'] or self.drone.position[2] < self.cfg['minimum_drone_height']
        if dist_to_current < self.cfg['dist_waypoint_proceed']:
            reward = 1
            self.last_timestep_new_waypoint = self.timestep
            if self.current_waypoint_index < self.cfg['ep_end_after_n_waypoints']:
                self.current_waypoint_index += 1
            else:
                done = True
        m = self.drone.get_rotation_matrix()
        state = np.concatenate([
            #np.array([[np.sin(o), np.cos(o)] for o in self.drone.orientation_euler]).flatten(),
            #np.array([[np.sin(o), np.cos(o)] for o in [self.drone.orientation_euler[2]]]).flatten(),
            #current_waypoint_rel,
            #next_waypoint_rel,
            current_waypoint_rel @ m,
            next_waypoint_rel @ m,
            #self.drone.absolute_speed,
            #self.drone.lateral_speed,
            #self.drone.ang_speed,
            #self.drone.position,
            #[self.drone.thrust, self.drone.setpoint_pitch, self.drone.setpoint_roll, self.drone.setpoint_yaw]
        ], axis=0)
        return state, reward, done, {}

    def render(self, _):
        for i in range(1, len(self.waypoints)):
            p.addUserDebugLine(self.waypoints[i-1], self.waypoints[i], lineColorRGB=[1,0,0], lineWidth=3, physicsClientId=self.client)

class FeedbackNormalizedSimEnv(SimEnv):
    def __init__(self, cfg):
        SimEnv.__init__(self, cfg)
        self.action_space = Box(-np.inf, np.inf, shape=(2,), dtype=float)

    def feedback_linearized(self, orientation, velocity, epsilon):
        u = velocity[0]*np.cos(orientation) + velocity[1]*np.sin(orientation)  # [m/s]
        w = (1/epsilon)*(-velocity[0]*np.sin(orientation) + velocity[1]*np.cos(orientation))  # [rad/s] going counter-clockwise.
        return u, w

    def reset(self):
        super()._reset()
        return self.step([1,0])[0]

    def step(self, action):
        if action[0] < 0.001:
            action[0] = 0.001
            
        #action[0] = np.clip(action[0], 0.1, 0.2)
        #action[1] = np.clip(action[1], 3, 4)
        #action[0] = 0.1
        #action[1] = 0.3
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
