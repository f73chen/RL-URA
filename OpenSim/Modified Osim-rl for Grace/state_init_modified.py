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

v = "v1"
d = "muscle"
log_path = f"{d}/muscle_log_{v}/"

def learning_rate(frac):
    return 1.0e-4*(np.exp(6*(frac-1)))



dir_path = os.path.dirname(os.path.realpath(__file__))
traj_path = dir_path + "\\" + "tracking_solution_fullStride.sto"

env = L2RunEnvMod(visualize=True, traj_path=traj_path)
# env = L2RunEnvMod(visualize=False, traj_path=traj_path)
obs = env.reset()


'''
policy_kwargs = dict(activation_fn=th.nn.Tanh,
                     net_arch=[dict(pi=[256,256,192,128], vf=[256,256,192,128])])     # v=6
model = PPO('MlpPolicy', env, verbose=0, policy_kwargs=policy_kwargs, learning_rate=learning_rate)
model.learn(total_timesteps=80)

# Test saving and loading
model.save(f"{d}/muscle_l{v}")
del model
'''
# '''
model = PPO.load(f"{d}/muscle_l{v}", env = env)
obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
# '''

# for i in range(100):
#     o, r, d, i = env.step(np.zeros(18))