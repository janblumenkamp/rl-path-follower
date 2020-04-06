from sim_env import SimEnv
import time
import numpy as np  
import yaml
import ray
from ray import tune
from ray.tune.registry import register_env
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
    #run()
    #exit()
    
    register_env("pybullet", lambda config: SimEnv(config))
    with open("cfg.yaml", "rb") as f:
        cfg = yaml.load(f)
    
    ray.init()
    tune.run(
        "PPO",
        #restore="/home/jb2270/ray_results/PPO/PPO_world_0_2020-04-04_23-01-16c532w9iy/checkpoint_100/checkpoint-100",
        checkpoint_freq=10,
        config={
            "env": "pybullet",
            "lambda": 0.95,
            "kl_coeff": 0.5,
            "clip_rewards": True,
            "clip_param": 0.3,
            "vf_clip_param": 1000.0,
            #"vf_share_layers": True,
            #"vf_loss_coeff": 1e-4,
            "entropy_coeff": 0.01,
            "train_batch_size": 5000,
            "sample_batch_size": 100,
            "sgd_minibatch_size": 500,
            "num_sgd_iter": 10,
            "num_workers": 8,
            "num_envs_per_worker": 10,
            "lr": 1e-4,
            "gamma": 0.9,
            "batch_mode": "truncate_episodes",
            "observation_filter": "NoFilter",
            "num_gpus": 1,
            "env_config": {
                  'init': [0, 0, 1],
                   'goal_box': 5
            }})
    
    #tune.run(
    #    "SAC",
        #restore="/home/jb2270/ray_results/PPO/PPO_world_0_2020-04-02_21-44-56vzulq1wd/checkpoint_190",
        #checkpoint_freq=10,
    #    config=cfg)

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
