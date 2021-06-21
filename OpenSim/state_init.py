from osim.env.osim_rsi import L2RunEnvRSI
import opensim

env = L2RunEnvRSI(visualize=True)
obs = env.reset()
print(obs)
# print(env.osim_model.get_state())
# for i in range(200):
#     o, r, d, i = env.step(env.action_space.sample())
