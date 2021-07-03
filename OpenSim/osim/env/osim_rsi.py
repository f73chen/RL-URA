from .osim import *
from .traj_reader import *
import collections
import numpy as np
import pandas as pd
import os.path

class OsimModelRSI(OsimModel):
    def __init__(self, init_coords, init_speeds, model_path, visualize, integrator_accuracy = 5e-5):
        self.integrator_accuracy = integrator_accuracy
        self.model = opensim.Model(model_path)

        # Load reference coordinates
        if init_coords:
            for i in range(9):
                self.model.getCoordinateSet().get(i).set_default_value(init_coords[i])

        # Load reference speeds
        if init_speeds:
            for i in range(9):
                self.model.getCoordinateSet().get(i).set_default_speed_value(init_speeds[i])

        self.model.initSystem()
        self.brain = opensim.PrescribedController()

        # Enable the visualizer
        self.model.setUseVisualizer(visualize)

        self.muscleSet = self.model.getMuscles()
        self.forceSet = self.model.getForceSet()
        self.bodySet = self.model.getBodySet()
        self.jointSet = self.model.getJointSet()
        self.markerSet = self.model.getMarkerSet()
        self.contactGeometrySet = self.model.getContactGeometrySet()

        if self.verbose:
            self.list_elements()

        # Add actuators as constant functions. Then, during simulations
        # we will change levels of constants.
        # One actuartor per each muscle
        for j in range(self.muscleSet.getSize()):
            func = opensim.Constant(1.0)
            self.brain.addActuator(self.muscleSet.get(j))
            self.brain.prescribeControlForActuator(j, func)

            self.maxforces.append(self.muscleSet.get(j).getMaxIsometricForce())
            self.curforces.append(1.0)

        self.noutput = self.muscleSet.getSize()
            
        self.model.addController(self.brain)
        self.model.initSystem()

    def reset(self, init_activations = None):
        self.state = self.model.initializeState()
        self.model.equilibrateMuscles(self.state)
        self.state.setTime(0)
        self.istep = 0
        self.reset_manager()

        # Set initial muscle activations
        if init_activations:
            self.set_activations(init_activations)

class OsimEnvRSI(OsimEnv):
    def __init__(self, traj_path, visualize = True, integrator_accuracy = 5e-5):
        # Load trajectory as an environment variable
        trajreader = trajReader(traj_path)
        self.traj = trajreader.get_traj()

        # Rename trajectory columns
        traj_columns = self.traj.columns.values
        new_columns = ["time"] + [col.split('/')[-2] + '/' + col.split('/')[-1] for col in traj_columns[1:]]
        self.traj.columns = new_columns

        self.visualize = visualize
        self.integrator_accuracy = integrator_accuracy
        self.load_model()

    def load_model(self, init_coords = None, init_speeds = None, model_path = None):
        if model_path:
            self.model_path = model_path
            
        # Changed this line to use OsimModelRSI instead
        self.osim_model = OsimModelRSI(init_coords = init_coords, init_speeds = init_speeds, model_path = self.model_path, visualize = self.visualize, integrator_accuracy = self.integrator_accuracy)

        # Create specs, action and observation spaces mocks for compatibility with OpenAI gym
        self.spec = Spec()
        self.spec.timestep_limit = self.time_limit

        self.action_space = ( [0.0] * self.osim_model.get_action_space_size(), [1.0] * self.osim_model.get_action_space_size() )
        self.observation_space = ( [0] * self.get_observation_space_size(), [0] * self.get_observation_space_size() )
        
        self.action_space = convert_to_gym(self.action_space)
        self.observation_space = convert_to_gym(self.observation_space)
       
    def reset(self, project = True):
        # Choose random state to init model with
        rand_idx = np.random.randint(0, len(self.traj))
        ref_state = self.traj.iloc[rand_idx, :]

        # At reset, load joint coords
        # Sorts list of coords and speeds according to osim_model joint order
        init_coords = ref_state.iloc[1:10]
        init_coords = [init_coords[self.osim_model.model.getCoordinateSet().get(i).getName() + "/value"] for i in range(9)]
        
        init_speeds = ref_state.iloc[10:19]
        init_speeds = [init_speeds[self.osim_model.model.getCoordinateSet().get(i).getName() + "/speed"] for i in range(9)]
        
        self.load_model(init_coords, init_speeds)

        # At reset, load reference muscle activations
        init_activations = list(ref_state.iloc[19:37])
        self.osim_model.reset(init_activations)

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