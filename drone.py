import numpy as np
import pybullet as p
from pid import PID

class Drone():
    def __init__(self, pybullet_client, initial_pos=np.zeros(3)):
        self.pybullet_client = pybullet_client
        self.initial_pos = self.position = np.array(initial_pos)
        self.body_id = p.loadURDF("./drone.urdf", basePosition=self.initial_pos, physicsClientId=self.pybullet_client)

        self.simulated_wind_vector = np.array([0.2, 0.1, 0.03])

        self.thrust = 0
        self.setpoint_roll = 0
        self.setpoint_pitch = 0
        self.setpoint_yaw = 0
        self.roll_pid = PID(0.3, 0.001, 8)
        self.pitch_pid = PID(0.3, 0.001, 8)
        self.yaw_pid = PID(1, 0, 0)
        
        self.pid_vx = PID(0.5, 0.0004, 8)
        self.pid_vy = PID(0.5, 0.0004, 8)
        self.pid_vz = PID(40, 0.2, 10)

        self.reset()	

    def reset(self):
        p.resetBasePositionAndOrientation(self.body_id, self.initial_pos, [0, 0, 0, 1], physicsClientId=self.pybullet_client)
        self.update_state()
        self.last_orientation = self.orientation
        self.last_position = self.position
        self.compute_speed()
        
    def update_state(self):
        position, orientation = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.pybullet_client)
        self.position = np.array(position)
        self.orientation = np.array(orientation)
        self.orientation_euler = np.array(p.getEulerFromQuaternion(orientation))

    def compute_speed(self):
        self.lateral_speed = (self.last_position - self.position)*240
        delta_speed_quat = p.getDifferenceQuaternion(self.last_orientation, self.orientation)
        self.ang_speed = np.array(p.getEulerFromQuaternion(delta_speed_quat))*240
        self.last_orientation = self.orientation
        self.last_position = self.position

    def set_thrust(self, thrust):
        self.thrust = np.clip(thrust, 0, 6)

    def set_pitch(self, pitch):
        self.setpoint_pitch = np.clip(pitch, -np.pi/4, np.pi/4)

    def set_roll(self, roll):
        self.setpoint_roll = np.clip(roll, -np.pi/4, np.pi/4)

    def set_yaw(self, yaw):
        self.setpoint_yaw = np.clip(yaw, -2*np.pi, 2*np.pi)
        
    def step(self):
        self.update_state()
        self.compute_speed()
        
        ctrl_pitch = self.pitch_pid.step(self.setpoint_pitch, self.orientation_euler[1])
        ctrl_roll = self.roll_pid.step(self.setpoint_roll, self.orientation_euler[0])
        ctrl_yaw = self.yaw_pid.step(self.setpoint_yaw, self.ang_speed[2])
        
        translation_forces = np.clip(np.array([
            self.thrust + ctrl_roll - ctrl_pitch,
            self.thrust - ctrl_roll - ctrl_pitch,
            self.thrust + ctrl_roll + ctrl_pitch,
            self.thrust - ctrl_roll + ctrl_pitch,
        ]), -8, 8)
        yaw_torque = np.clip(ctrl_yaw, -0.5, 0.5)

        motor_points = np.array([[1,1,0], [1,-1,0], [-1,1,0], [-1,-1,0]])*0.2
        for force_point, force in zip(motor_points, translation_forces):
            p.applyExternalForce(self.body_id, -1, [0,0,force], force_point, p.LINK_FRAME, physicsClientId=self.pybullet_client)
        p.applyExternalTorque(self.body_id, 0, [0,0,yaw_torque], p.LINK_FRAME, physicsClientId=self.pybullet_client)

        # apply noise and wind
        p.applyExternalForce(self.body_id, -1, (np.random.rand(3)-0.5)*2, self.position, p.WORLD_FRAME, physicsClientId=self.pybullet_client)
        p.applyExternalTorque(self.body_id, 0, (np.random.rand(3)-0.5)*0.1, p.LINK_FRAME, physicsClientId=self.pybullet_client)
        p.applyExternalForce(self.body_id, -1, self.simulated_wind_vector, self.position, p.WORLD_FRAME, physicsClientId=self.pybullet_client)

    def step_speed(self, vx, vy, vz):
        self.set_pitch(self.pid_vx.step(np.clip(vx, -12, 12), -self.lateral_speed[0]))
        self.set_roll(self.pid_vy.step(np.clip(vy, -12, 12), self.lateral_speed[1]))
        self.set_thrust(self.pid_vz.step(np.clip(vz, -7, 7), -self.lateral_speed[2]))

