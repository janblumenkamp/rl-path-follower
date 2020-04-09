from sim_env import SimEnv
import time
import numpy as np  
np.random.seed(1)

# Can alternatively pass in p.DIRECT 

def run():
    cfg = {
        'init': [0, 0, 1],
        'render': True,
        "waypoints": 4
    }

    env = SimEnv(cfg)

    try:
        obs = env.reset()
        cum_reward = 0
        for i in range(10000):
            goal_vec = -obs[:3]
            goal_d = np.linalg.norm(goal_vec, ord=2)
            goal_vec /= np.abs(goal_vec).max()*2

            #goal_vec /= 100
            #if goal_d < 0.5:
            #    goal_vec /= 3
            if i == 0:
                env.render()
            obs, reward, done, _ = env.step(goal_vec)#env.action_space.sample())
            cum_reward += reward
            #print(obs, reward, cum_reward, done)
            print(obs[6:])
            time.sleep(1./240.)
            #time.sleep(1./10)
            if done:                
                input("Done")
                obs = env.reset()
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
