from drone import Drone
import numpy as np
import time
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

import threading
import keyboard

current_control = {'thrust': 14, 'pitch': 0, 'roll': 0, 'yaw': 0}
def keyboard_command(command, change):
    global current_control
    limits = {'thrust': (0, 50), 'pitch': (-np.pi, np.pi), 'roll': (-np.pi, np.pi), 'yaw': (-np.pi, np.pi)}
    new_control = current_control[command]+change
    if new_control >= limits[command][0] and new_control <= limits[command][1]:
        current_control[command] = new_control

keyboard.add_hotkey('up', keyboard_command, args=('pitch', -0.01))
keyboard.add_hotkey('down', keyboard_command, args=('pitch', 0.01))
keyboard.add_hotkey('left', keyboard_command, args=('roll', 0.01))
keyboard.add_hotkey('right', keyboard_command, args=('roll', -0.01))
keyboard.add_hotkey('page up', keyboard_command, args=('thrust', 1))
keyboard.add_hotkey('page down', keyboard_command, args=('thrust', -1))
keyboard.add_hotkey(',', keyboard_command, args=('yaw', -0.1))
keyboard.add_hotkey('.', keyboard_command, args=('yaw', 0.1))
threading.Thread(target=lambda : keyboard.wait(hotkey="alt+f4"), args=()).start()

client = p.connect(p.GUI)
p.setGravity(0, 0, -10, physicsClientId=client) 
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF("plane.urdf", physicsClientId=client)
drone = Drone(client, np.array([0,0,1]))

def print_info():
    while True:
        print(current_control['roll'], drone.orientation[0])
        #print(drone.orientation)
        time.sleep(0.05)

#threading.Thread(target=print_info, args=()).start()

from pid import PID

def eval_pid_roll():
    pif_h = PID(2, 0.01, 200)
    desired_height = 3
    angle_sign = -1
    setpoints = []
    real_data = []
    timestep = []
    current_time = 0
    start_logging = False
    while True:
        drone.set_pitch(0)
        setpoint = 0
        if start_logging:
            setpoint = np.pi/8*angle_sign
        setpoints.append(setpoint)
        real_data.append(drone.orientation_euler[0])
        timestep.append(current_time)
        current_time += 1/240
        drone.set_roll(setpoint)
        drone.set_yaw(0)
        drone.set_thrust(pif_h.step(desired_height, drone.position[2]))
        drone.step()

        if drone.position[2] >= desired_height:
            start_logging = True
        if drone.position[1] > 6:
            angle_sign = 1
        elif drone.position[1] < -1:
            break
        
        if np.linalg.norm(drone.position, ord=2) > 15:
            drone.reset()
    
        p.stepSimulation(physicsClientId=client)
        #time.sleep(1/240)
    
    plt.plot(timestep, setpoints)
    plt.plot(timestep, real_data)
    #plt.ylim(-0.3,1)
    plt.grid()
    plt.show()

def eval_pid_yaw():
    pif_h = PID(4, 0.01, 100)
    desired_height = 2
    angle_sign = 1
    setpoints = []
    real_data = []
    timestep = []
    current_time = 0
    start_logging = False
    while True:
        drone.set_pitch(0)
        setpoint = 0
        if start_logging:
            setpoint = np.pi*angle_sign
        setpoints.append(setpoint)
        real_data.append(drone.ang_speed[2])
        timestep.append(current_time)
        current_time += 1/240
        drone.set_roll(0)
        drone.set_yaw(setpoint)
        drone.set_thrust(pif_h.step(desired_height, drone.position[2]))
        if current_time > 0.1:
            start_logging = True
        if angle_sign == 1 and drone.orientation[2] > np.pi/16:
            angle_sign = -1
        elif angle_sign == -1 and drone.orientation[2] < -np.pi/16:
            break
        
        drone.step()
        
        if np.linalg.norm(drone.position, ord=2) > 15:
            drone.reset()
    
        p.stepSimulation(physicsClientId=client)
        #time.sleep(1/240)
    
    plt.plot(timestep, setpoints)
    plt.plot(timestep, real_data)
    #plt.ylim(-0.3,1)
    plt.grid()
    plt.show()

def remote_control():
    pid_h = PID(3, 0.001, 150)
    desired_height = 2
    while True:
        drone.step_speed(0, 0, pid_h.step(desired_height, drone.position[2]))
        drone.set_pitch(current_control['pitch'])
        drone.set_roll(current_control['roll'])
        drone.set_yaw(current_control['yaw'])
        drone.step()
        if np.linalg.norm(drone.position, ord=2) > 5:
            drone.reset()
        p.stepSimulation(physicsClientId=client)
        time.sleep(1/240)

def eval_pid_speed_hor():
    angle_sign = 1
    setpoints = []
    real_data = []
    timestep = []
    current_time = 0
    start_logging = False
    while True:
        setpoint = 0
        if start_logging:
            setpoint = -12*angle_sign
        setpoints.append(setpoint)
        real_data.append(drone.lateral_speed[1])
        timestep.append(current_time)
        current_time += 1/240
        
        drone.step_speed(0, setpoint, 0)
        drone.step()
        
        if current_time > 5:
            start_logging = True

        if drone.position[1] > 40:
            #break
            angle_sign = -1
        elif drone.position[1] < -10:
            break
        
        if np.linalg.norm(drone.position, ord=2) > 100:
            drone.reset()
    
        p.stepSimulation(physicsClientId=client)
        #time.sleep(1/240)
    
    plt.plot(timestep, setpoints)
    plt.plot(timestep, real_data)
    #plt.ylim(-0.3,1)
    plt.grid()
    plt.show()

def eval_pid_speed_ver():
    angle_sign = 1
    setpoints = []
    real_data = []
    timestep = []
    current_time = 0
    start_logging = True
    start_hover = 0
    while True:
        setpoint = 0
        if start_logging:
            setpoint = 4*angle_sign
        setpoints.append(setpoint)
        real_data.append(-drone.lateral_speed[2])
        timestep.append(current_time)
        current_time += 1/240
        
        drone.step_speed(0, 0, setpoint)
        drone.step()

        if current_time > 1:
            start_logging = True

        if angle_sign == 1 and drone.position[2] > 4:
            angle_sign = -1
        elif angle_sign == -1 and drone.position[2] < 2:
            angle_sign = 0
            start_hover = current_time
        elif angle_sign == 0 and current_time - start_hover > 2:
            break
        
        if np.linalg.norm(drone.position, ord=2) > 15:
            drone.reset()
    
        p.stepSimulation(physicsClientId=client)
        time.sleep(1/240)
    
    plt.plot(timestep, setpoints)
    plt.plot(timestep, real_data)
    #plt.ylim(-0.3,1)
    plt.grid()
    plt.show()

def eval_pid_pos_ver():
    pif_h = PID(3, 0.001, 150)
    
    setpoints = []
    out = []
    real_data = []
    timestep = []
    current_time = 0
    start_hover = 0
    setpoint = 0
    while True:
        if current_time < 1:
            setpoint = 1
        elif current_time < 5:
            setpoint = 6
        elif current_time < 9:
            setpoint = 3
        elif current_time < 13:
            setpoint = 0
        elif current_time > 15:
            break
            
        setpoints.append(setpoint)
        real_data.append(drone.position[2])
        timestep.append(current_time)
        current_time += 1/240
        
        o = pif_h.step(setpoint, drone.position[2])
        out.append(-drone.lateral_speed[2])
        print(o)
        drone.step_speed(0, 0, o)
        drone.step()
        
        if np.linalg.norm(drone.position, ord=2) > 15:
            drone.reset()
    
        p.stepSimulation(physicsClientId=client)
        time.sleep(1/240)
    
    plt.plot(timestep, setpoints)
    plt.plot(timestep, real_data)
    #plt.plot(timestep, out)
    #plt.ylim(-0.3,1)
    plt.grid()
    plt.show()

#eval_pid_speed_ver()
#eval_pid_speed_hor()
remote_control()
#eval_pid_pos_ver()
#eval_pid_pos_hor()
#eval_pid_yaw()
#eval_pid_roll()
