import numpy as np

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

