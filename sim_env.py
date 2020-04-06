import numpy as np
import gym
import pybullet as p
import pybullet_data
from gym.spaces import Box

class PID():
    def __init__(self, kp, ki, kd):
        self.k = {'P': kp, 'I': ki, 'D': kd}
        self.reset()

    def reset(self):
        self.integral = 0
        self.last_error = 0
        
    def step(self, setpoint, feedback):
        error = setpoint - feedback
        self.integral += error
        p = self.k['P']*error
        i = self.k['I']*self.integral
        d = self.k['D']*(error - self.last_error)
        self.last_error = error
        return p+i+d

class Drone():
    def __init__(self, initial_pos=np.zeros(3)):
        self.initial_pos = initial_pos
        self.pos = self.initial_pos
        self.body_id = p.loadURDF("/auto/homes/jb2270/l310_project/drone.urdf", basePosition=self.pos)
        self.pids = [
            PID(10000, 0, 0),
            PID(10000, 0, 0),
            PID(10000, 0, 0)
        ]
        self.reset()	

    def update_pos(self):
        return np.array(p.getBasePositionAndOrientation(self.body_id)[0])

    def reset(self):
        self.last_pos = self.pos = self.update_pos()
        p.resetBasePositionAndOrientation(self.body_id, self.initial_pos, [0, 0, 0, 1])

    def step(self, desired_speed):
        desired_speed = np.clip(desired_speed, -0.05, 0.05)
        self.pos = self.update_pos()
        self.speed = self.pos - self.last_pos
        self.last_pos = self.pos

        pid_vec = np.array([ctrl.step(setpoint, feedback) for ctrl, setpoint, feedback in zip(self.pids, np.array(desired_speed), self.speed)])
        pid_vec_clipped = np.clip(pid_vec, -30, 30) # simulate physical constraints
        p.applyExternalForce(self.body_id, 0, pid_vec_clipped, self.pos, p.WORLD_FRAME)

class SimEnv(gym.Env):
    def __init__(self, config):
        self.cfg = config
        self.action_space = Box(-1, 1, shape=(3,), dtype=float)
        self.observation_space = Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)

        self.client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -10, physicsClientId=self.client) 
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")

        self.drone = Drone(config['init'])
        
        self.reset()

    def __del__(self):
        p.disconnect()

    def reset(self):
        self.drone.reset()
        rand_box = lambda s: np.random.uniform(-s, s, 3) + np.array([0, 0, 0.5 + s])
        self.goal = rand_box(self.cfg['goal_box'])
        self.cnt_timesteps_goal_reached = 0
        return self.step([0,0,0])[0]

    def step(self, action):
        self.drone.step(action)
        p.stepSimulation()

        #reward = 0
        dist_to_goal = np.linalg.norm(self.drone.pos - self.goal, ord=2)
        reward = -dist_to_goal/100
        if dist_to_goal < 0.1:
            self.cnt_timesteps_goal_reached += 1
        #    reward += 1
        else:
            self.cnt_timesteps_goal_reached = 0

        done = dist_to_goal > 5 or self.cnt_timesteps_goal_reached > 100
        return np.concatenate([self.drone.pos - self.goal, self.drone.speed], axis=0), reward, done, {}

