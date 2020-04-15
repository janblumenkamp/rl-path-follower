import numpy as np
import gym
from gym.spaces import Box
from path_generator import PathGenerator

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_CONFIG = {
    'step_freq': 240,
    'reset_pos': [0, 0, 1],
    'pid': {
        'roll': [0.2, 0.01, 0.1],
        'pitch': [0.2, 0.01, 0.1],
        'yaw': [1, 0, 0],
        'vx': [0.5, 0.2, 0.01],
        'vy': [0.5, 0.2, 0.01],
        'vz': [100, 1, 0.1]
    },
    'limits': {
        'thrust': (0, 6),
        'pitch': (-np.pi/8, np.pi/8),
        'roll': (-np.pi/8, np.pi/8),
        'yaw': (-2*np.pi, 2*np.pi),
        'pid_pitch_out': (-3, 3),
        'pid_roll_out': (-3, 3),
        'pid_yaw_out': (-3, 3),
        'motor_force': (0, 15),
        'yaw_torque': (-0.5, 0.5),
        'vx': (-12, 12),
        'vy': (-12, 12),
        'vz': (-7, 1.5), # if we allow higher rising speeds, there are some issues with the speed controllers
    },
    'wind_vector': [0.2, 0.1, 0.03],
    'force_noise_fac': 2,
    'torque_noise_fac': 0.1
}

class Quadcopter():
    def __init__(self, cfg=DEFAULT_CONFIG):
        self.cfg = cfg
        self.initial_pos = self.position = np.array(self.cfg['reset_pos'], dtype=np.float)

        self.reset()	

    def reset(self):
        self.yaw = 0
        self.orientation_euler = np.zeros(3, dtype=np.float)
        self.position = self.initial_pos.copy()

    def get_rotation_matrix(self):
        s, c = np.sin(self.orientation_euler[2]), np.cos(self.orientation_euler[2])
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]])

    def step(self):
        self.orientation_euler[2] += self.yaw/self.cfg['step_freq']
        while self.orientation_euler[2] > np.pi:
            self.orientation_euler[2] -= 2*np.pi
        while self.orientation_euler[2] < -np.pi:
            self.orientation_euler[2] += 2*np.pi

    def set_yaw(self, yaw):
        self.yaw = np.clip(yaw, *self.cfg['limits']['yaw'])

    def step_speed(self, vx, vy, vz):
        dt = 1/self.cfg['step_freq']
        v = self.get_rotation_matrix() @ np.array([vx, vy, vz])*dt
        self.position += v


class SimEnv(gym.Env):
    def __init__(self, config):
        self.cfg = config
        self.action_space = Box(-1, 1, shape=(3,), dtype=np.float32)
        self.observation_space = Box(-np.inf, np.inf, shape=(17,), dtype=np.float32) # orientation, relative current destination, next destination
        self.path_track_length_episode = 250

        self.drone = Quadcopter()
        CONFIG_SPACE_SIZE = 4
        self.path_generator = PathGenerator(np.array([[-CONFIG_SPACE_SIZE, CONFIG_SPACE_SIZE], [-CONFIG_SPACE_SIZE, CONFIG_SPACE_SIZE], [1, CONFIG_SPACE_SIZE]]))

        self.reset()

    def get_drone_pose(self):
        return np.array(list(self.drone.position)+[self.drone.orientation_euler[2]])

    def _reset(self):
        self.drone.reset()
        while True:
            self.waypoints = self.path_generator.get_path(self.get_drone_pose(), self.path_generator.sample_configuration_space())
            if len(self.waypoints) > self.path_track_length_episode:
                break
        self.current_waypoint_index = 0
        self.next_waypoint_index = 1
        self.timestep = 0
        self.fig = None

    def reset(self):
        self._reset()
        return self.step([0,0,0])[0]

    def step(self, action):
        assert(not any(np.isnan(action)))
        action *= np.array([2, 1, 1])
        self.timestep += 1
        self.drone.step_speed(action[0], 0, action[1])
        self.drone.set_yaw(action[2])
        self.drone.step()

        reward = 0
        current_waypoint_rel = (self.waypoints[self.current_waypoint_index] - self.drone.position)
        self.next_waypoint_index = self.current_waypoint_index
        if self.next_waypoint_index < len(self.waypoints)-1:
            self.next_waypoint_index += 1
        next_waypoint_rel = (self.waypoints[self.next_waypoint_index] - self.drone.position)
        
        dist_to_current = np.linalg.norm(current_waypoint_rel, ord=2)
        done = dist_to_current > 2 or self.timestep > 10000
        if dist_to_current < 0.2:
            reward = 1
            if self.current_waypoint_index < self.path_track_length_episode:
                self.current_waypoint_index += 1
            else:
                done = True

        m = self.drone.get_rotation_matrix()
        state = np.concatenate([
            np.array([[np.sin(o), np.cos(o)] for o in [self.drone.orientation_euler[2]]]).flatten(),
            current_waypoint_rel,
            next_waypoint_rel,
            current_waypoint_rel @ m,
            next_waypoint_rel @ m,
            self.drone.position
        ], axis=0)
        return state, reward, done, {}

    def render(self):
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(constrained_layout=True, figsize=(16, 10))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.view_init(90, 0)
            self.im_pos, = self.ax.plot([], [], [])

            self.ax.scatter(self.waypoints[...,0], self.waypoints[...,1], self.waypoints[...,2])
        
        if self.timestep % 100 == 0:
            self.im_pos.set_data(
                [self.drone.position[0], self.drone.position[0] + 0.1*np.cos(self.drone.orientation_euler[2])],
                [self.drone.position[1], self.drone.position[1] + 0.1*np.sin(self.drone.orientation_euler[2])]
            )
            self.im_pos.set_3d_properties([self.drone.position[2], self.drone.position[2]])

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        return self.fig

class FeedbackNormalizedSimEnv(SimEnv):
    def __init__(self, cfg):
        SimEnv.__init__(self, cfg)
        self.action_space = Box(0.01, 5, shape=(2,), dtype=np.float32)

    def feedback_linearized(self, orientation, velocity, epsilon):
        u = velocity[0]*np.cos(orientation) + velocity[1]*np.sin(orientation)  # [m/s]
        w = (1/epsilon)*(-velocity[0]*np.sin(orientation) + velocity[1]*np.cos(orientation))  # [rad/s] going counter-clockwise.
        return u, w

    def reset(self):
        super()._reset()
        return self.step([1,0])[0]

    def step(self, action):
        action = np.array(action) * [0.2, 1]
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
