import time
import numpy as np  
import yaml
import ray
from ray import tune
from ray.tune.registry import register_env
from demo_env import DemoEnv

if __name__ == '__main__':
    register_env("demo", lambda config: DemoEnv(config))

    ray.init()
    
    tune.run(
        "PPO",
        checkpoint_freq=10,
        config={
            "env": "demo",
            "lambda": 0.95,
            "kl_coeff": 0.5,
            "clip_rewards": True,
            "clip_param": 0.2,
            "vf_clip_param": 10.0,
            "entropy_coeff": 0.01,
            "train_batch_size": 1000,
            "sample_batch_size": 100,
            "sgd_minibatch_size": 500,
            "num_sgd_iter": 10,
            "num_workers": 16,
            "num_envs_per_worker": 16,
            "lr": 1e-3,
            "gamma": 0.9,
            "batch_mode": "truncate_episodes",
            "observation_filter": "NoFilter",
            "num_gpus": 1,
            "env_config": {
                "dim": 2,
                "waypoints": 4
            },
            "model": {
                "fcnet_activation": "relu",
                "fcnet_hiddens": [32, 32],
            }
        }
    )

