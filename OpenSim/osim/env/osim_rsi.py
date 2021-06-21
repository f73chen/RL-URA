from .osim import L2RunEnv
import collections
import numpy as np
import pandas as pd
import os.path

class L2RunEnvRSI(L2RunEnv):
    model_path = os.path.join(os.path.dirname(__file__), '../models/gait9dof18musc_20170320.osim')
    
    MASS = 75.1646 # 11.777 + 2*(9.3014 + 3.7075 + 0.1 + 1.25 + 0.2166) + 34.2366
    G = 9.80665 

    def __init__(self, *args, **kwargs):
        super(L2RunEnvRSI, self).__init__(*args, **kwargs)

    def reset(self):
        return super().reset()

    def ref_state_init(self, params):
        m = self.osim_model
        s = m.get_state()
        muscle_preset = params['Muscles']
        joint_preset = params['Joints']
        print(f"Muscle preset: {muscle_preset}")
        print(f"Joint preset:  {joint_preset}")

        for name, position in joint_preset.items():
            for i in range(len(position)):
                j = m.get_joint(name).upd_coordinates(i)
                j.setValue(s, position[i])

        for name, activation in muscle_preset.items():
            m.get_muscle(name).setActivation(s, activation)