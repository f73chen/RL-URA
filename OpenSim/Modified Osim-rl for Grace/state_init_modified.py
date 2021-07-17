import os
import time

import opensim
import gym
import pybullet_envs
import numpy as np
import torch as th
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from osim.env.osimMod36d import L2RunEnvMod

params = {'reward_weight': [6.0, 1.0, 1.0, 0.4, 0.0, 1.0, 1.0, 0.0, 0.5, 1.0],
          #['forward', 'survival', 'torso', 'joint', 'stability', 'act', 'footstep', 'jerk', 'slide']
          'action_limit': [1]*18,
          'time_limit': 10,
          'stepsize': 0.01,
          'integrator_accuracy': 5e-3,
          'seed': 0,
          'num_cpu': 12,
          'lr_a1': 1.0e-4,
          'lr_a2': 2, 
          'target_speed_range': [0.8,1.2],
          'total_timesteps': 800}

v = "v6"
d = "muscle"
log_path = f"{d}/muscle_log_{v}/"

def learning_rate(frac):
    return 1.0e-4*(np.exp(6*(frac-1)))

def own_policy(obs):
    action = np.zeros(18)
    return action


dir_path = os.path.dirname(os.path.realpath(__file__))
traj_path = dir_path + "\\" + "tracking_solution_fullStride.sto"

env = L2RunEnvMod(reward_weight=params['reward_weight'],
                  action_limit=params['action_limit'],
                  target_speed_range=params['target_speed_range'],
                  own_policy=own_policy,
                  visualize=True, 
                  traj_path=traj_path)
obs = env.reset()


# '''
policy_kwargs = dict(activation_fn=th.nn.Tanh,
                     net_arch=[dict(vf=[256,256,256], pi=[256,256,256,12])])     # v=6
model = PPO('MlpPolicy', env, verbose=0, policy_kwargs=policy_kwargs, learning_rate=learning_rate, n_steps=128)
model.learn(total_timesteps=params['total_timesteps'])

# Test saving and loading
model.save(f"{d}/muscle_l{v}")
del model
# '''
'''
model = PPO.load(f"{d}/muscle_l{v}", env = env)
obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=False)
    # print(action)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
'''

# for i in range(100):
#     o, r, d, i = env.step(np.zeros(18))