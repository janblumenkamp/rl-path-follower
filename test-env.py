from sim_env import SimEnv, FeedbackNormalizedSimEnv
#from sim_env_nodrone import SimEnv, FeedbackNormalizedSimEnv
import time
import numpy as np  
np.random.seed(2)
import matplotlib.pyplot as plt

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
        'path_state_waypoints_lookahead': 3,
        'waypoints_lookahead_skip': 10,
        'ep_end_after_n_waypoints': 1000,
        'max_timesteps_between_checkpoints': 2000,
        'dist_waypoint_abort_ep': 2,
        'minimum_drone_height': 0.2,
        'dist_waypoint_proceed': 1.0,
    }

    env = SimEnv(cfg)

    obs = env.reset()
    #env.render()
    cum_reward = 0
    
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter([],[],[])
    ax.scatter([0],[0],[0], c='r', s=10)
    ax.set_xlim(0,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.view_init(elev=30, azim=180)
    
    for i in range(30000):
        print(obs)
        next_pos = obs[:3]
        obs_r = obs.reshape(cfg['path_state_waypoints_lookahead'], 3)
        if i % 100 == 0:
            sc._offsets3d = (obs_r[...,0], obs_r[...,1], obs_r[...,2])
            fig.canvas.draw()
            #fig.canvas.flush_events()
        
        epsilon = 0.7
        v = (next_pos[:2] - np.array([epsilon, 0]))*3
        u, w = feedback_linearized(0, v, epsilon=epsilon)
        h = next_pos[2]
        
        print(u,h,w)
        obs, reward, done, _ = env.step(np.array([
            env.map_action(np.clip(u, -1, 5), -1, 5, -1, 1),
            env.map_action(np.clip(h, -1, 1), -1, 1, -1, 1),
            env.map_action(np.clip(w, -4*np.pi, 4*np.pi), -4*np.pi, 4*np.pi, -1, 1),
        ]))#env.action_space.sample())
        cum_reward += reward
        #print(obs[8:11])
        #print(obs, reward, cum_reward, done)
        #env.render()
        #time.sleep(1./240.)
        #time.sleep(1./10)
        if done:                
            input(f"{i} Done")
            obs = env.reset()
            env.render()
            cum_reward = 0
            #break

def test():
    cfg = {
        'render': True,
        'path_state_waypoints_lookahead': 10,
        'waypoints_lookahead_skip': 3,
        'ep_end_after_n_waypoints': 400,
        'max_timesteps_between_checkpoints': 2000,
        'dist_waypoint_abort_ep': 2,
        'minimum_drone_height': 0.2,
        'dist_waypoint_proceed': 0.2,
    }

    env = SimEnv(cfg)

    obs = env.reset()
    #env.render()
    cum_reward = 0
    actions = []
    for i in range(30000):
        action = env.action_space.sample()
        print(action)
        obs, reward, done, _ = env.step(np.array([action[0], 0.1, 0]))
        actions.append(action[0])
        cum_reward += reward
        #print(obs, reward, cum_reward, done)
        #env.render()
        time.sleep(1./240.)
        #time.sleep(1./10)
        if done:              
            import matplotlib.pyplot as plt
            plt.hist(actions)
            plt.show()  
            input(f"{i} Done")
            obs = env.reset()
            #env.render()
            cum_reward = 0
            #break
        
def runfb():
    cfg = {
        'render': True,
        'path_state_waypoints_lookahead': 3,
        'waypoints_lookahead_skip': 10,
        'ep_end_after_n_waypoints': 400,
        'max_timesteps_between_checkpoints': 2000,
        'dist_waypoint_abort_ep': 2,
        'minimum_drone_height': 0.2,
        'dist_waypoint_proceed': 1.0,
    }

    env = FeedbackNormalizedSimEnv(cfg)

    try:
        obs = env.reset()
        env.render()
        cum_reward = 0
        for i in range(20000):
            obs, reward, done, _ = env.step([-0.5, 1]) #env.action_space.sample())
            cum_reward += reward
            #print(obs, reward, cum_reward, done)
            time.sleep(1./240.)
            #env.render()
            #time.sleep(1./10)
            if done:                
                input("Done")
                obs = env.reset()
                #env.render()
                cum_reward = 0
                #break
    except Exception as e:
        del env
        print("Exit", e)

if __name__ == '__main__':
    run()
    
