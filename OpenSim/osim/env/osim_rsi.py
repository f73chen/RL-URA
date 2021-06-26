from .osim import *
import collections
import numpy as np
import pandas as pd
import os.path

class OsimModelRSI(OsimModel):
    def reset(self, params):
        # L2RunRSI calls this reset function instead of the original
        # Able to receive params and load normally (if nothing else is changed)
        print(params)

        # Below is the same as the original
        self.state = self.model.initializeState()
        self.model.equilibrateMuscles(self.state)
        self.state.setTime(0)
        self.istep = 0
        self.reset_manager()

class OsimEnvRSI(OsimEnv):
    def load_model(self, model_path = None):
        if model_path:
            self.model_path = model_path
            
        # Changed this line to use OsimModelRSI instead
        self.osim_model = OsimModelRSI(self.model_path, self.visualize, integrator_accuracy = self.integrator_accuracy)

        # Create specs, action and observation spaces mocks for compatibility with OpenAI gym
        self.spec = Spec()
        self.spec.timestep_limit = self.time_limit

        self.action_space = ( [0.0] * self.osim_model.get_action_space_size(), [1.0] * self.osim_model.get_action_space_size() )
#        self.observation_space = ( [-math.pi*100] * self.get_observation_space_size(), [math.pi*100] * self.get_observation_space_s
        self.observation_space = ( [0] * self.get_observation_space_size(), [0] * self.get_observation_space_size() )
        
        self.action_space = convert_to_gym(self.action_space)
        self.observation_space = convert_to_gym(self.observation_space)
       
    def reset(self, params, project = True):
        # Changed this line to pass in params
        self.osim_model.reset(params)
        
        if not project:
            return self.get_state_desc()
        return self.get_observation()

class L2RunEnvRSI(OsimEnvRSI):
    model_path = os.path.join(os.path.dirname(__file__), '../models/gait9dof18musc.osim')    
    time_limit = 1000

    def is_done(self):
        state_desc = self.get_state_desc()
        return state_desc["body_pos"]["pelvis"][1] < 0.6

    ## Values in the observation vector
    def get_observation(self):
        state_desc = self.get_state_desc()

        # Augmented environment from the L2R challenge
        res = []
        pelvis = None

        res += state_desc["joint_pos"]["ground_pelvis"]
        res += state_desc["joint_vel"]["ground_pelvis"]

        for joint in ["hip_l","hip_r","knee_l","knee_r","ankle_l","ankle_r",]:
            res += state_desc["joint_pos"][joint]
            res += state_desc["joint_vel"][joint]

        for body_part in ["head", "pelvis", "torso", "toes_l", "toes_r", "talus_l", "talus_r"]:
            res += state_desc["body_pos"][body_part][0:2]

        res = res + state_desc["misc"]["mass_center_pos"] + state_desc["misc"]["mass_center_vel"]

        res += [0]*5

        return res

    def get_observation_space_size(self):
        return 41

    def reward(self):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0
        return state_desc["joint_pos"]["ground_pelvis"][1] - prev_state_desc["joint_pos"]["ground_pelvis"][1]


    # def ref_state_init(self, params):
    #     m = self.osim_model
    #     s = m.get_state()
    #     muscle_preset = params['Muscles']
    #     joint_preset = params['Joints']
    #     print(f"Muscle preset: {muscle_preset}")
    #     print(f"Joint preset:  {joint_preset}")

    #     ''' Didn't work: all joints at 0 '''
    #     # for name, position in joint_preset.items():
    #     #     for i in range(len(position)):
    #     #         j = m.get_joint(name).upd_coordinates(i)
    #     #         j.setValue(s, position[i])

    #     ''' Didn't work: m.get_activations() returns 0.05 for all '''
    #     # for name, activation in muscle_preset.items():
    #     #     m.get_muscle(name).setActivation(s, activation)
    #     #     print(m.get_activations())

    #     # m.getCoordinateSet().get(0).setValue(0.5)

    #     # coordinate_set = m.model.updCoordinateSet()

    #     # for i in range(10):
    #     #     coordinate_set.get(i).setValue(s, 0.0)
    #     # print(m.list_elements())