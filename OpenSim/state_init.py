from osim.env.osim_rsi import L2RunEnvRSI
import opensim

params = {"Muscles": {"hamstrings_r": 0.5,
                      "bifemsh_r": 0.25},
          "Joints": {"ground_pelvis": [0.3, 0.4, 0.5],
                     "knee_r": [0.5]}}

env = L2RunEnvRSI(visualize=True)
obs = env.reset(params)
# env.ref_state_init(params)

for i in range(50):
    o, r, d, i = env.step(env.action_space.sample())

# print(env.get_state_desc())