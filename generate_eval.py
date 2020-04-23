from sim_env import SimEnv, FeedbackNormalizedSimEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle

def make_env(seed=0, env_class="sim"):
    def _init():
        cfg = {
            'render': False,
            'path_state_waypoints_lookahead': 10,
            'waypoints_lookahead_skip': 3,
            'ep_end_after_n_waypoints': 1000,
            'max_timesteps_between_checkpoints': 5000,
            'dist_waypoint_abort_ep': 5,
            'minimum_drone_height': 0.2,
            'dist_waypoint_proceed': 1.0,
        }
        env = {"fb": FeedbackNormalizedSimEnv, "sim": SimEnv}[env_class](cfg)
        return env
    return _init

def run_generate_ground_truth(n_trials):
    env_gt = make_env(0, env_class="sim")()
    paths_ground_truths = []
    for i in range(n_trials):
        env_gt.seed(i)
        env_gt.reset()
        paths_ground_truths.append({'path': env_gt.waypoints})
    return paths_ground_truths

def run_feedback_baseline(n_trials):
    env = make_env(0, env_class="fb")()
    trials = []
    for i in range(n_trials):
        path = []
        env.seed(i)
        obs = env.reset()
        while True:
            path.append(env.drone.position)
            obs, rewards, dones, info = env.step([0.1, 1])
            if dones:
                break
        trials.append({'path': np.array(path)})
    return trials

def run_policy(model, env, n_trials):
    trials = []
    for i in range(n_trials):
        path = []
        speed = []
        observations = []
        actions = []
        env.seed(i)
        obs = env.reset()
        while True:
            path.append(env.drone.position)
            speed.append(env.drone.lateral_speed)
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            observations.append(obs)
            actions.append(action)
            if dones:
                break
        trials.append({'actions': np.array(actions), 'observations': np.array(observations), 'path': np.array(path), 'speed': np.array(speed)})
    return trials

def run_complex(n_trials):
    model = PPO2.load("./results/all/DronePathComplex-v0_52/best_model.zip")
    env = make_env(0, env_class="sim")()
    return run_policy(model, env, n_trials)

def run_fb(n_trials):
    model = PPO2.load("./results/all/DronePathComplexFB-v0_2/best_model.zip")
    env = make_env(0, env_class="fb")()
    return run_policy(model, env, n_trials)

if __name__ == '__main__':
    n_trials = 20
    pickle.dump({
        "ground_truth": run_generate_ground_truth(n_trials),
        "fb_baseline": run_feedback_baseline(n_trials),
        "complex": run_complex(n_trials),
        "fb": run_fb(n_trials)
    }, open("eval_results.pkl", "wb"))

