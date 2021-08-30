import numpy as np
import pandas as pd
import scipy.interpolate as interp
from scipy.signal import savgol_filter
import time
from IPython.display import clear_output

def get_mirror_expert_dataset(model, env, get_observation, mirror_obs, mirror_act, n_episodes = 20):
    '''
    model: stable-baselines model
    env: L2RunEnvMod environment
    get_obs: callable, get_obs(state_desc) -> obs
    mirror_obs: callable, mirror_obs(obs) -> mirrored_obs
    mirror_act: callable, mirror_act(action) -> mirrored_action
    '''
    actions = []
    observations = []
    rewards = []
    episode_returns = np.zeros((n_episodes,))
    episode_starts = []

    actions_mirror = []
    observations_mirror = []
    rewards_mirror = []
    episode_returns_mirror = np.zeros((n_episodes,))
    episode_starts_mirror = []

    ep_idx = 0
    obs = env.reset()
    state_desc = env.get_state_desc()
    obs_mod = get_observation(state_desc)
    episode_starts.append(True)
    episode_starts_mirror.append(True)
    reward_sum = 0.0
    idx = 0

    while ep_idx < n_episodes:
        clear_output(wait=True)
        print('progress: {:.0%}'.format((ep_idx+1)/n_episodes))
        obs_ = obs_mod
        observations.append(obs_)
        observations_mirror.append(mirror_obs(obs_))
        action, _ = model.predict(obs)

        obs, reward, done, _ = env.step(action)
        state_desc = env.get_state_desc()
        obs_mod = get_observation(state_desc)

        actions.append(action)
        actions_mirror.append(mirror_act(action))
        rewards.append(reward)
        rewards_mirror.append(reward)
        episode_starts.append(done)
        episode_starts_mirror.append(done)
        reward_sum += reward
        idx += 1
        if done:
            obs = env.reset()
            episode_returns[ep_idx] = reward_sum
            episode_returns_mirror[ep_idx] = reward_sum
            reward_sum = 0.0
            ep_idx += 1

    observations = np.concatenate(observations).reshape((-1,) + (len(obs_mod),))
    actions = np.concatenate(actions).reshape((-1,) + (len(action),))
    observations_mirror = np.concatenate(observations_mirror).reshape((-1,) + (len(obs_mod),))
    actions_mirror = np.concatenate(actions_mirror).reshape((-1,) + (len(action),))

    rewards = np.array(rewards)
    rewards_mirror = np.array(rewards_mirror)
    episode_starts = np.array(episode_starts[:-1])
    episode_starts_mirror = np.array(episode_starts_mirror[:-1])

    assert len(observations) == len(actions)

    numpy_dict = {
        'actions': np.concatenate((actions,actions_mirror), axis=0),
        'obs': np.concatenate((observations,observations_mirror), axis=0),
        'rewards': np.concatenate((rewards,rewards_mirror), axis=0),
        'episode_returns': np.concatenate((episode_returns,episode_returns_mirror), axis=0),
        'episode_starts': np.concatenate((episode_starts,episode_starts_mirror), axis=0)
    }  # type: Dict[str, np.ndarray]

    for key, val in numpy_dict.items():
        print(key, val.shape)
        
    return numpy_dict