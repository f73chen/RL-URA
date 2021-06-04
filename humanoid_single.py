import os
import time

import numpy as np
import matplotlib.pyplot as plt

import gym
import pybullet_envs
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

v = "6"
d = "humanoid"
log_path = f"{d}/humanoid_log_{v}/"

env = gym.make('HumanoidBulletEnv-v0')
# env.render()

# '''
def learning_rate(frac):
    return 1.0e-4*(np.exp(6*(frac-1)))

policy_kwargs = dict(activation_fn=th.nn.Tanh,
                     net_arch=[dict(pi=[256,256,192,128], vf=[256,256,192,128])])
model = PPO('MlpPolicy', env, verbose=0, policy_kwargs=policy_kwargs, learning_rate=learning_rate, tensorboard_log=log_path)
# model.learn(total_timesteps=30000000)
model.learn(total_timesteps=300)

# Test saving and loading
model.save(f"{d}/humanoid_l{v}")
del model
# '''

'''
model = PPO.load(f"{d}/humanoid_l{v}", env = env)
obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    # env.render()
    time.sleep(1/200)
    if done:
        obs = env.reset()
'''