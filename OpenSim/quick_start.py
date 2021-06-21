from osim.env.osim import L2RunEnv

env = L2RunEnv(visualize=True)
observation = env.reset(project=True)
print(observation)
for i in range(200):
    o, r, d, i = env.step(env.action_space.sample())

# from osim.env import L2M2019Env

# env = L2M2019Env(visualize=False)
# observation = env.reset()
# for i in range(200):
#     observation, reward, done, info = env.step(env.action_space.sample())
