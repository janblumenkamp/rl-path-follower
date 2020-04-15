import gym
from gym.spaces import Box
#from sim_env import SimEnv, FeedbackNormalizedSimEnv
from sim_env_nodrone import SimEnv, FeedbackNormalizedSimEnv
import tensorflow as tf

from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2, A2C
from stable_baselines.results_plotter import load_results, ts2xy

from stable_baselines.common.policies import FeedForwardPolicy


import time
import os
import numpy as np

log_dir = "/auto/homes/jb2270/gym/"
    
def make_env(rank, seed=0):
    def _init():
        env = SimEnv({
            'dim': 3,
            'init': [0, 0, 1],
            'render': False,
            "waypoints": 4
        })
        env = Monitor(env, log_dir if rank == 0 else None, allow_early_resets=True)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

from stable_baselines.common.callbacks import BaseCallback

from stable_baselines.common.vec_env import VecEnvWrapper
import numpy as np
import time
from collections import deque
import os.path as osp
import json
import csv

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)
        return True

# Custom MLP policy of three layers of size 128 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                            act_fun=tf.nn.tanh,
                                           net_arch=[dict(pi=[128, 128, 128, 128, 128, 128], vf=[128, 128, 128, 128])],
                                           feature_extraction="mlp")

def train():
    env = SubprocVecEnv([make_env(i) for i in range(16)])
    #env = VecMonitor(env, log_dir+"monitor.csv")
    '''
    env = SimEnv({
            'dim': 3,
            'init': [0, 0, 1],
            'render': False,
            "waypoints": 4
        })
    
    env = Monitor(env, log_dir, allow_early_resets=True)
    '''
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    model = PPO2(CustomPolicy, env, verbose=1, nminibatches=32, n_steps=512)#, cliprange=0.3, gamma=0.95)
    #model = A2C(CustomPolicy, env, verbose=1, n_steps=32, gamma=0.95)
    model.learn(total_timesteps=int(2e6),callback=callback)
    model.save(log_dir + "model")


def run():
    model = PPO2.load(log_dir + "best_model")
    env = make_env(0)() #SubprocVecEnv([make_env(i) for i in range(1)])
    obs = env.reset()
    cum_rew = 0
    for i in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        #env.render()
        cum_rew += rewards
        print(obs, rewards, cum_rew, dones)
        if dones:
            print("Done", cum_rew)
            break

    env.close()
    
if __name__ == '__main__':
    run()
    #train()
