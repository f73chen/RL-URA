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
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps

from osim.env.osimMod36d import L2RunEnvMod

params = {'reward_weight': [6.0, 1.0, 1.0, 0.4, 0.0, 1.0, 1.0, 0.0, 0.5, 1.0],  # Weights change between iterations (ignore this)
          #['forward', 'survival', 'torso', 'joint', 'stability', 'act', 'footstep', 'jerk', 'slide', 'mimic']
          'action_limit': [1]*18,
          'time_limit': 120,    # Time limit changes between iterations (ignore this)
          'stepsize': 0.01,
          'integrator_accuracy': 5e-5,
          'seed': 0,
          'num_cpu': 12,
          'lr_a1': 1.0e-4,
          'lr_a2': 2,
          'target_speed_range': [1.2, 1.2],
          'total_timesteps': 4000000}

v = "v21"       # Log version number: change this for each run
d = "muscle"    # Log dir for both tensorboard and model zip files
log_dir = f"{d}/muscle_log_{v}/"
tb_dir = log_dir + "tb/"
os.makedirs(log_dir, exist_ok=True)

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
    def _init():
        env_in.time_limit = time_limit
        env = env_in(**kwargs)
        env.osim_model.stepsize = stepsize
        log_sub_dir = log_dir + '/env_{}'.format(str(rank))
        os.makedirs(log_sub_dir, exist_ok=True)
        env = Monitor(env, log_sub_dir, allow_early_resets=True)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

# Path to trajectory file; make sure speeds match with target speed range
dir_path = os.path.dirname(os.path.realpath(__file__))
traj_path = dir_path + "\\traj\\" + "1.2_gaitPrediction_solution_fullStride.sto"


def extract_xy(log_dir, num_rollout):
    y = []
    for folder in os.listdir(log_dir):
        if folder.startswith('env_'):
            _, y_tmp = ts2xy(load_results(log_dir+folder), 'timesteps')
            if len(y_tmp) > 0:
                y.extend(list(y_tmp[-num_rollout:]))
    y = sum(y)/len(y) if len(y) > 0 else -np.inf
    return y

class LogCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, log_dir, verbose=0, num_rollout=5):
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf
        self.num_rollout = num_rollout
        super(LogCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        mean_reward = extract_xy(self.log_dir, self.num_rollout)
        if mean_reward != -np.inf:
            # clear_output(wait=True)
            print(self.num_timesteps, 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}". \
                  format(self.best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                self.model.save(self.log_dir + 'best_model')
                self.model.save(self.log_dir + 'latest_model')
            else:
                print("Saving latest model")
                self.model.save(self.log_dir + 'latest_model')
        return True

log_callback = LogCallback(log_dir, num_rollout=5)
event_callback = EveryNTimesteps(n_steps=2000, callback=log_callback)

def iter_env(time_limit, reward_weight):
    env = SubprocVecEnv([make_env(L2RunEnvMod, i, time_limit,
                                    seed=params['seed'],
                                    stepsize=params['stepsize'],
                                    reward_weight = reward_weight,
                                    action_limit = params['action_limit'],
                                    visualize=False,
                                    traj_path=traj_path,
                                    integrator_accuracy=params['integrator_accuracy'],
                                    target_speed_range = params['target_speed_range'],
                                    own_policy=own_policy)
                            for i in range(params['num_cpu'])])
    return env


if __name__ ==  '__main__':

    # '''   ### TRAIN ###
    # Makes a new environment for each iteration
    iter_params = [{'time_limit': 30,   'reward_weight': [6.0, 1.0, 1.0, 0.4, 0.0, 1.0, 1.0, 0.0, 0.5, 3.5]},
                   {'time_limit': 60,   'reward_weight': [6.0, 1.0, 1.0, 0.4, 0.0, 1.0, 1.0, 0.0, 0.5, 3.0]},
                   {'time_limit': 90,   'reward_weight': [6.0, 1.0, 1.0, 0.4, 0.0, 1.0, 1.0, 0.0, 0.5, 2.5]},
                   {'time_limit': 120,  'reward_weight': [6.0, 1.0, 1.0, 0.4, 0.0, 1.0, 1.0, 0.0, 0.5, 2.0]},
                   {'time_limit': 150,  'reward_weight': [6.0, 1.0, 1.0, 0.4, 0.0, 1.0, 1.0, 0.0, 0.5, 1.5]},
                   {'time_limit': 250,  'reward_weight': [6.0, 1.0, 1.0, 0.4, 0.0, 1.0, 1.0, 0.0, 0.5, 1.0]},
                   {'time_limit': 1000, 'reward_weight': [6.0, 1.0, 1.0, 0.4, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5]}]

    envs = [iter_env(**ip) for ip in iter_params]

    policy_kwargs = dict(activation_fn=th.nn.Tanh,
                        net_arch=[dict(vf=[512,512,512,256], pi=[512,512,512,256])])

    model = PPO('MlpPolicy', envs[0], verbose=0, policy_kwargs=policy_kwargs, learning_rate=learning_rate, n_steps=128, tensorboard_log=log_dir) # 
    for i in range(len(envs)):
        obs = envs[i].reset()
        if i > 0:
            model.set_env(envs[i])
        model.learn(total_timesteps=params['total_timesteps'], callback=event_callback)
        model.save(f"{d}/muscle_l{v}_{i}")

    del model
    # '''

    ''' ### DISPLAY ###
    env = iter_env(time_limit=params['time_limit'], reward_weight=params['reward_weight'])
    model = PPO.load(f"{d}/muscle_l{v}", env = env)
    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=False)
        # print(action)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
    '''


'''
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve', instances=1, same_plot=False):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    :instances: (int) the number of instances to average
    """
    x, y = ts2xy(load_results(log_folder+'/env_0'), 'timesteps')

    if instances > 1:
        for i in range(1,instances):
            _, y_tmp = ts2xy(load_results(log_folder+'/env_'+str(i)), 'timesteps')
            if len(y) > len(y_tmp):
                y = y[:len(y_tmp)] + y_tmp
            else:
                y = y + y_tmp[:len(y)]
        y = y/instances

    y = moving_average(y, window=5) # change window value to change level of smoothness
    # Truncate x
    x = x[len(x) - len(y):]
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    if same_plot is False:
        plt.show()

plot_results(log_dir, instances=6)
'''