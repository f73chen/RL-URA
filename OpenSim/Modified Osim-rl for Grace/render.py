import warnings
import bz2
import pickle
import os
import numpy as np
warnings.filterwarnings('ignore')

from osim.env.osimMod36d import L2RunEnvMod 
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from osim.env.utils import analysis as ana
from osim.env.utils.analysis import interpolate_gait_data 

if __name__ ==  '__main__':

    # Load agent as model_v
    model_v = PPO.load(f"muscle/muscle_lv21_3")

    params = {'reward_weight': [6.0, 1.0, 1.0, 0.4, 0.0, 1.0, 1.0, 0.0, 0.5, 1.5],
            #['forward', 'survival', 'torso', 'joint', 'stability', 'act', 'footstep', 'jerk', 'slide', 'mimic']
            'action_limit': [1]*18,
            'time_limit': 300,
            'stepsize': 0.01,
            'integrator_accuracy': 5e-5,
            'seed': 0,
            'num_cpu': 1,
            'lr_a1': 1.0e-4,
            'lr_a2': 2,
            'target_speed_range': [1.2, 1.2],
            'total_timesteps': 4000000}
    dir_path = os.path.dirname(os.path.realpath(__file__))
    traj_path = dir_path + "\\traj\\" + "1.2_gaitPrediction_solution_fullStride.sto"
    log_dir = f"muscle/muscle_log_v21_3/"

    def own_policy(obs):
        action = np.zeros(18)
        return action

    def make_env(env_in, rank, time_limit, seed=0, stepsize=0.01, **kwargs):
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

    # Load environment as env_v
    def iter_env(time_limit, reward_weight):
        env = L2RunEnvMod(reward_weight = reward_weight, 
                          action_limit = params['action_limit'], 
                          target_speed_range = params['target_speed_range'],
                          own_policy=own_policy,
                          visualize=True,
                          traj_path=traj_path,
                          integrator_accuracy=params['integrator_accuracy']) 
                                        
        return env

    env_v = iter_env(time_limit=params['time_limit'], reward_weight=params['reward_weight'])
    print(type(env_v))

    analysis = ana.gaitAnalysis(shift = -6.5)
    data = analysis.collect_traj_data(model_v, env_v, max_step = 1000, episode_length=1000, deterministic=False)

    # plt.plot(analysis.extract_state(['muscles','hamstrings_r','activation'])[0:100])
    # plt.plot(analysis.extract_state(['muscles','hamstrings_r','excitation'])[0:100])
    # plt.show()

    # analysis = ana.gaitAnalysis()
    # analysis.load_traj_data(log_dir+'traj_data')
    
    exp_data = analysis.load_exp_joint_data(dir_path + "\\traj\\" + "1.2_gaitPrediction.csv", mode='Moco')
    exp_legend = ["reference"]

    fig1 = analysis.plot_joint_angles(exp_data=exp_data, exp_legends=exp_legend, side='left', size=[10,3])
    plt.show()
    fig2 = analysis.plot_joint_angles(exp_data=exp_data, exp_legends=exp_legend, side='right', size=[10,3])
    plt.show()
    fig3 = analysis.plot_joint_angles(exp_data=exp_data, exp_legends=exp_legend, side='avg', size=[10,3])
    plt.show()

    fig = analysis.plot_excitation_raster(steps=500, figsize=[10,4], side='right')
    plt.show()