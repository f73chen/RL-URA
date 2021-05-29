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

v = "5"
d = "humanoid"
log_path = f"{d}/humanoid_log_{v}/"

env = gym.make('HumanoidBulletEnv-v0')
env.render()

'''
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[256,256,256,256], vf=[256,256,256,256])])
model = PPO('MlpPolicy', env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log=log_path)
model.learn(total_timesteps=500000)

# Test saving and loading
model.save(f"{d}/humanoid_l{v}")
del model
'''

model = PPO.load(f"{d}/humanoid_l{v}", env = env)
obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(1/200)
    if done:
        obs = env.reset()