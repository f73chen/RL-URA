import gym
import pybulletgym
import time
env = gym.make('HumanoidPyBulletEnv-v0')
for i_episode in range(500):
    env.render()
    observation = env.reset()
    for t in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation)
        time.sleep(1/30)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break