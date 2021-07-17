import os
import opensim
import numpy as np
# from osim.env.osim_rsi import L2RunEnvRSI
from osim.env.osim import L2RunEnv

# dir_path = os.path.dirname(os.path.realpath(__file__))
# traj_path = dir_path + "\\" + "tracking_solution_fullStride.sto"

# env = L2RunEnvRSI(visualize=True, traj_path=traj_path)
env = L2RunEnv(visualize=True)
obs = env.reset()

for i in range(100):
    o, r, d, i = env.step(np.zeros(18))
    # print(env.get_state_desc())