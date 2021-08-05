import os
import time

import opensim
import gym
import numpy as np
import torch as th
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from stable_baselines3.common.utils import set_random_seed

from osim.env.osimMod36d import L2RunEnvMod

params = {'reward_weight': [6.0, 1.0, 1.0, 0.4, 0.0, 1.0, 1.0, 0.0, 0.5, 4],
          #['forward', 'survival', 'torso', 'joint', 'stability', 'act', 'footstep', 'jerk', 'slide', 'mimic']
          'action_limit': [1]*18,
          'time_limit': 1000,
          'stepsize': 0.01,
          'integrator_accuracy': 5e-5,
          'seed': 0,
          'num_cpu': 1,
          'lr_a1': 1.0e-4,
          'lr_a2': 2, 
          'target_speed_range': [0.8,1.2],
          'total_timesteps': 4000000}

v = "s0"
d = "muscle"
log_dir = f"{d}/muscle_log_{v}/"

def learning_rate(frac):
    return 1.0e-4*(np.exp(6*(frac-1)))

def own_policy(obs):
    action = np.zeros(18)
    return action

def make_env(env_in, rank, time_limit, seed=0, stepsize=0.01, **kwargs):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses 
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    # if os.path.exists(log_dir + '/env_0/monitor.csv'):
    #     raise Exception("existing monitor files found!!!")
    
    def _init():
        env_in.time_limit = time_limit
        env = env_in(**kwargs) 
        env.osim_model.stepsize = stepsize
        # log_sub_dir = log_dir + '/env_{}'.format(str(rank))
        # os.makedirs(log_sub_dir, exist_ok=True)
        # env = Monitor(env, log_sub_dir, allow_early_resets=True)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


dir_path = os.path.dirname(os.path.realpath(__file__))
traj_path = dir_path + "\\" + "tracking_solution_fullStride.sto"



if __name__ ==  '__main__':
    env = SubprocVecEnv([make_env(L2RunEnvMod, i, params['time_limit'], 
                                seed=params['seed'], 
                                stepsize=params['stepsize'], 
                                reward_weight = params['reward_weight'], 
                                action_limit = params['action_limit'], 
                                visualize=True,
                                traj_path=traj_path,
                                integrator_accuracy=params['integrator_accuracy'], 
                                target_speed_range = params['target_speed_range'], 
                                own_policy=own_policy,
                                muscle_synergy=True) 
                        for i in range(params['num_cpu'])])

    # print(env.observation_space)    # Box(0.0, 0.0, (36,), float32)
    # print(env.init_space)

    # env = L2RunEnvMod(reward_weight=params['reward_weight'],
    #                   action_limit=params['action_limit'],
    #                   target_speed_range=params['target_speed_range'],
    #                   own_policy=own_policy,
    #                   visualize=True, 
    #                   traj_path=traj_path)
    obs = env.reset()


    '''
    policy_kwargs = dict(activation_fn=th.nn.Tanh,
                        net_arch=[dict(vf=[512,512,512,256], pi=[512,512,512,256])])     # v=5
    model = PPO('MlpPolicy', env, verbose=0, policy_kwargs=policy_kwargs, learning_rate=learning_rate, n_steps=128) # , tensorboard_log=log_dir
    # model = PPO.load(f"{d}/muscle_lv5", env = env)
    model.learn(total_timesteps=params['total_timesteps'])

    # Test saving and loading
    model.save(f"{d}/muscle_l{v}")
    del model
    '''
    # '''
    model = PPO.load(f"{d}/muscle_l{v}", env = env)
    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
    # '''

    # for i in range(1):
    #     o, r, d, i = env.step([[1, 1, 0, 0, 0, 0.1, 0.2, 0.3, 1, 1, 0, 0, 0, 0.4, 0.5, 0.6]])
    