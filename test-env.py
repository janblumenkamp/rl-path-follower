from sim_env import SimEnv, FeedbackNormalizedSimEnv
import time
import numpy as np  
np.random.seed(1)

# Can alternatively pass in p.DIRECT 

X = 0
Y = 1
Z = 2
YAW = 3
EPSILON = 0.1

def feedback_linearized(orientation, velocity, epsilon):
    u = velocity[X]*np.cos(orientation) + velocity[Y]*np.sin(orientation)  # [m/s]
    w = (1/epsilon)*(-velocity[X]*np.sin(orientation) + velocity[Y]*np.cos(orientation))  # [rad/s] going counter-clockwise.
    return u, w
    
def run():
    cfg = {
        'render': True,
        "waypoints": 4
    }

    env = SimEnv(cfg)

    try:
        obs = env.reset()
        env.render()
        cum_reward = 0
        for i in range(30000):
            orientation = np.arctan2(obs[0], obs[1])
            next_pos = obs[5:8]
            
            position = np.array([
                EPSILON * np.cos(orientation),
                EPSILON * np.sin(orientation),
                0], dtype=np.float32)
            v = (next_pos - position)*3
            u, w = feedback_linearized(orientation, v, epsilon=EPSILON)
            h = v[2]
            
            obs, reward, done, _ = env.step(np.array([u, h, w]))#env.action_space.sample())
            cum_reward += reward
            print(obs, reward, cum_reward, done)
            #time.sleep(1./240.)
            #time.sleep(1./10)
            if done:                
                input(f"{i} Done")
                obs = env.reset()
                env.render()
                cum_reward = 0
                #break
    except Exception as e:
        del env
        print("Exit", e)

def runfb():
    cfg = {
        'render': True,
        "waypoints": 4
    }

    env = FeedbackNormalizedSimEnv(cfg)

    try:
        obs = env.reset()
        env.render()
        cum_reward = 0
        for i in range(20000):
            obs, reward, done, _ = env.step(np.array([EPSILON, 3]))#env.action_space.sample())
            cum_reward += reward
            print(obs, reward, cum_reward, done)
            time.sleep(1./240.)
            #time.sleep(1./10)
            if done:                
                input("Done")
                obs = env.reset()
                env.render()
                cum_reward = 0
                #break
    except Exception as e:
        del env
        print("Exit", e)

if __name__ == '__main__':
    run()
    exit()
'''
d = Drone()
d.set_speed([0,0,0.01])
for i in range(200): 
    d.step()
    p.stepSimulation()
    time.sleep(1./240.)

d.set_speed([0,0.01,0.0])

for i in range(1000): 
    d.step()
    p.stepSimulation()
    time.sleep(1./240.)

d.set_speed([0,0,-0.005])
for i in range(1000): 
    d.step()
    p.stepSimulation()
    time.sleep(1./240.)
#time.sleep(5)
p.disconnect()
'''
