import sys
sys.path.append('/auto/homes/jb2270/rl-path-follower')
import gym
from gym.envs.registration import registry, register, make, spec
#from sim_env_nodrone import SimEnv, FeedbackNormalizedSimEnv
from sim_env import SimEnv as SimEnv
from sim_env import FeedbackNormalizedSimEnv as FeedbackNormalizedSimEnv
from sim_env_nodrone import SimEnv as SimEnvSimple
from sim_env_nodrone import FeedbackNormalizedSimEnv as FeedbackNormalizedSimEnvSimple


def initialize_custom_envs():
    cfg = {
        'render': False,
        'path_state_waypoints_lookahead': 10,
        'waypoints_lookahead_skip': 3,
        'ep_end_after_n_waypoints': 1000,
        'max_timesteps_between_checkpoints': 2000,
        'dist_waypoint_abort_ep': 2.0,
        'minimum_drone_height': 0.2,
        'dist_waypoint_proceed': 1.0
    }

    def make_sim_env():
        def _init():
            env = SimEnv(cfg)
            return env
        return _init

    register(
        id='DronePathComplex-v0',
        entry_point=make_sim_env(),
        max_episode_steps=20000,
        reward_threshold=200,
    )

    def make_sim_feedback_env():
        def _init():
            env = FeedbackNormalizedSimEnv(cfg)
            return env
        return _init

    register(
        id='DronePathComplexFB-v0',
        entry_point=make_sim_feedback_env(),
        max_episode_steps=20000,
        reward_threshold=200,
    )

    def make_sim_env_simple():
        def _init():
            env = SimEnvSimple(cfg)
            return env
        return _init

    register(
        id='DronePathSimple-v0',
        entry_point=make_sim_env_simple(),
        max_episode_steps=20000,
        reward_threshold=200,
    )

    def make_sim_feedback_env_simple():
        def _init():
            env = FeedbackNormalizedSimEnvSimple(cfg)
            return env
        return _init

    register(
        id='DronePathSimpleFB-v0',
        entry_point=make_sim_feedback_env_simple(),
        max_episode_steps=20000,
        reward_threshold=200,
    )
