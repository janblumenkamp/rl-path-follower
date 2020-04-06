from sim_env import SimEnv
import time
import numpy as np  
np.random.seed(1)

# Can alternatively pass in p.DIRECT 

def run():
    cfg = {
        'init': [0, 0, 1],
        'goal_box': 5
    }

    env = SimEnv(cfg)

    try:
        env.reset()
        time.sleep(2)
        reached_goal = False
        for i in range(1000):
            goal_vec = env.goal - env.drone.pos
            goal_vec /= np.abs(goal_vec).max()*100
            obs, reward, done, _ = env.step(np.array([0,0,0] if reached_goal else goal_vec, dtype=float))
            print(obs, reward, done)
            time.sleep(1./240.)
            if done:
                reached_goal = True
                print("DONE")
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
