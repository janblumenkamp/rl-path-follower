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
        self.body_id = p.loadURDF("./drone.urdf", basePosition=self.pos)
        self.pids = [
            PID(10000, 200, 50000),
            PID(10000, 200, 50000),
            PID(10000, 200, 50000)
        ]
        self.reset()	

    def update_pos(self):
        return np.array(p.getBasePositionAndOrientation(self.body_id)[0])

    def reset(self):
        self.last_pos = self.pos = self.update_pos()
        p.resetBasePositionAndOrientation(self.body_id, self.initial_pos, [0, 0, 0, 1])

    def step(self, desired_speed):
        self.pos = self.update_pos()
        self.speed = self.pos - self.last_pos
        self.last_pos = self.pos

        pid_vec = np.array([ctrl.step(setpoint, feedback) for ctrl, setpoint, feedback in zip(self.pids, np.array(desired_speed), self.speed)])
        pid_vec_clipped = np.clip(pid_vec, -30, 30) # simulate physical constraints
        p.applyExternalForce(self.body_id, 0, pid_vec_clipped, self.pos, p.WORLD_FRAME)

class SimEnv(gym.Env):
    def __init__(self, config):
        self.cfg = config
        self.action_space = Box(-0.05, 0.05, shape=(3,), dtype=float)
        self.observation_space = Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)

        self.client = p.connect(p.GUI)
        p.setGravity(0, 0, -10, physicsClientId=self.client) 
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")

        self.drone = Drone(config['init'])
        
        self.reset()

    def __del__(self):
        p.disconnect()

    def reset(self):
        self.last_pos = self.drone.pos
        self.drone.reset()
        rand_box = lambda s: np.random.uniform(-s, s, 3) + np.array([0, 0, 0.5 + s])
        self.goal = rand_box(self.cfg['goal_box'])
        return self.step([0,0,0])[0]

    def step(self, action):
        self.drone.step(action)
        p.stepSimulation()

        new_pos = self.drone.pos
        d_last_pos_to_goal = np.linalg.norm(self.last_pos - self.goal, ord=2)
        d_new_pos_to_goal = np.linalg.norm(new_pos - self.goal, ord=2)
        reward = d_last_pos_to_goal - d_new_pos_to_goal
        if reward < 0.0:
            reward *= 2
        self.last_pos = new_pos
        speed = np.linalg.norm(self.drone.speed, ord=2)
        done = (d_new_pos_to_goal < 0.1 and speed < 0.001) or d_new_pos_to_goal > 5
        return np.concatenate([new_pos, self.drone.speed], axis=0), reward*100, done, {}

