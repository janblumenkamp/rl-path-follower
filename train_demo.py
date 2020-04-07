import time
import numpy as np  
import yaml
import ray
from ray import tune
from ray.tune.registry import register_env
import pybullet as p
from demo_env import DemoEnv

if __name__ == '__main__':
    #run()
    #exit()
    
    register_env("demo", lambda config: DemoEnv(config))
    with open("cfg.yaml", "rb") as f:
        cfg = yaml.load(f)
    #p.connect(p.DIRECT)
    #p.setAdditionalSearchPath("/auto/homes/jb2270/rl-path-follower") # so that drone file can be found
    ray.init()
    
    tune.run(
        "PPO",
        #restore="/home/jb2270/ray_results/PPO/PPO_world_0_2020-04-04_23-01-16c532w9iy/checkpoint_100/checkpoint-100",
        checkpoint_freq=10,
        config={
            "env": "demo",
            "lambda": 0.95,
            "kl_coeff": 0.5,
            "clip_rewards": True,
            "clip_param": 0.2,
            "vf_clip_param": 10.0,
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
            "model": {
                "fcnet_activation": "relu",
                "fcnet_hiddens": [256, 256],
            },
            "env_config": {
                'init': [0, 0, 1],
                'goal_box': 3,
                'render': False
            }})
    '''
    tune.run(
        "SAC",
        checkpoint_freq=10,
        config=cfg)
    '''
    
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
