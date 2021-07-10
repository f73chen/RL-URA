from .osim import L2RunEnv
import collections
import numpy as np
import pandas as pd
import os.path

####################################################################################################### common constants

####################################################################################################### common functions

def get_observation_vec(state_desc, target_speed):
    # custom observation, scale vel down by 10, scale down force by 1000
    res = []
    pelvis = None

    # pelvis global pitch angle and linear position of y only
    res.append(state_desc["joint_pos"]["ground_pelvis"][0])
    res.append(state_desc["joint_pos"]["ground_pelvis"][2])
    # pelvis global pitch velocity and linear velocity including x and y
    res += state_desc["joint_vel"]["ground_pelvis"][0:3]

    # joint angles and velocity
    for joint in ["hip_l","hip_r","knee_l","knee_r","ankle_l","ankle_r",]:
        res += state_desc["joint_pos"][joint]
        res += list(np.array(state_desc["joint_vel"][joint])/10)

    # body part positions relative to pelvis
    for body_part in ["head", "torso", "toes_l", "toes_r", "talus_l", "talus_r"]:
        res.append(state_desc["body_pos"][body_part][0]-state_desc["body_pos"]["pelvis"][0])
        res.append(state_desc["body_pos"][body_part][1]-state_desc["body_pos"]["pelvis"][1])

    # vertical ground reaction forces
    res.append(state_desc['forces']['contactFront_r'][1]/1000)
    res.append(state_desc['forces']['contactHeel_r'][1]/1000)
    res.append(state_desc['forces']['contactFront_l'][1]/1000)
    res.append(state_desc['forces']['contactHeel_l'][1]/1000)
#     res.append(state_desc['forces']['contactFront_r'][1]/1000 + state_desc['forces']['contactHeel_r'][1]/1000)
#     res.append(state_desc['forces']['contactFront_l'][1]/1000 + state_desc['forces']['contactHeel_l'][1]/1000)
    
    # center of mass velocity
    res += list(np.array(state_desc["misc"]["mass_center_vel"][0:2])/10)
    res += [target_speed]
    return res

def get_BoS(state_desc):
    # get Base of Support
    feet_pos_x = [state_desc["body_pos"]["talus_l"][0], state_desc["body_pos"]["talus_r"][0],
                  state_desc["body_pos"]["toes_l"][0], state_desc["body_pos"]["toes_r"][0]]
    return [min(feet_pos_x), max(feet_pos_x)]

def get_eCoM(state_desc):
    # get extended Center of Mass eCoM, which is the frontal foot placement to bring CoM acceleration to 0
    G = 9.80665
    CoM_pos_x = state_desc["misc"]["mass_center_pos"][0]
    CoM_pos_y = state_desc["misc"]["mass_center_pos"][1]
    CoM_acc_x = state_desc["misc"]["mass_center_acc"][0]
    return CoM_pos_x + (CoM_pos_y / G) * CoM_acc_x


####################################################################################################### common rewards
def get_forward_reward(state_desc, target_speed, dt, mode="qua"):
    '''
    reward for moving forward, evaluated using speed at CoM
    '''
    if mode == "exp":
        forward_reward = np.exp(-8*(state_desc["misc"]["mass_center_vel"][0]-target_speed)**2)
    elif mode == "qua":
        forward_reward = -(state_desc["misc"]["mass_center_vel"][0]-target_speed)**2+1 
    elif mode == "simple":
        forward_reward = state_desc["misc"]["mass_center_vel"][0]-0.2
        # -0.2 is to ensure non-zero reward value when forward speed is 0
    else:
        raise NotImplementedError
    return forward_reward*dt

def get_survival_reward(state_desc, dt):
    '''
    reward for maintaining upward position
    '''
    upward_height = np.clip((state_desc["body_pos"]["head"][1] - 1.45), None, 0.02)
    return 100*upward_height*dt
    
def get_torso_reward(state_desc, dt):
    '''
    reward for upward torso orientation
    '''
    # offset the torso orientation by the initial angle of -0.187 rad
    torso_orientation = -(state_desc['body_pos_rot']['torso'][2]+0.187)**2
    
    return 100*torso_orientation*dt

def get_joint_reward(state_desc, dt):
    '''
    panelty for passive joint torque when going to extreme joint angle 
    '''
    total_passive_torque = abs(state_desc['forces']['HipLimit_r'][0]) + \
                           abs(state_desc['forces']['HipLimit_l'][0]) + \
                           abs(state_desc['forces']['KneeLimit_r'][0]) + \
                           abs(state_desc['forces']['KneeLimit_l'][0]) + \
                           abs(state_desc['forces']['AnkleLimit_r'][0]) + \
                           abs(state_desc['forces']['AnkleLimit_l'][0])
    return -0.1*total_passive_torque*dt

def get_stability_reward(state_desc, dt):
    '''
    panelty for the eCom deviating away from frontal BoS
    '''
    BoS = get_BoS(state_desc)
    eCoM = get_eCoM(state_desc)
    return -(eCoM - BoS[1])**2*dt

def get_act_reward(state_desc, dt):
    '''
    muscle activation panelty, NOT the excitation panelty
    '''
    
    # muscle volume ratio from https://www.sciencedirect.com/science/article/pii/S0021929013006234
    act_weight = [0.133, 0.020, 0.176, 0.094, 0.056, 0.318, 0.084, 0.091, 0.028,
                  0.133, 0.020, 0.176, 0.094, 0.056, 0.318, 0.084, 0.091, 0.028,]
    
    act = 0
    i = 0
    for muscle in sorted(state_desc['muscles'].keys()):
        act += state_desc['muscles'][muscle]['activation']**3 # * act_weight[i]
        i += 1

    return -act*dt

def get_exc_reward(state_desc, dt):
    '''
    muscle excitation panelty, NOT the activation panelty
    '''
    exc = 0
    for muscle in sorted(state_desc['muscles'].keys()):
        exc += state_desc['muscles'][muscle]['excitation']**3

    return -exc*dt

def get_jerk_reward(state_desc, prev_state_desc, dt):
    '''
    panelty of jerk at center of mass 
    '''
    jerkx = ((state_desc['misc']['mass_center_acc'][0]-prev_state_desc['misc']['mass_center_acc'][0])/dt)**2
    jerky = ((state_desc['misc']['mass_center_acc'][1]-prev_state_desc['misc']['mass_center_acc'][1])/dt)**2
    # return -1e-4*(jerkx+jerky)*dt
    return -1e-5*(jerkx+8*jerky)*dt

def get_slide_reward(state_desc, dt):
    l_heel_contact = state_desc['forces']['contactHeel_l'][1]<0
    l_toe_contact = state_desc['forces']['contactFront_l'][1]<0
    r_heel_contact = state_desc['forces']['contactHeel_r'][1]<0
    r_toe_contact = state_desc['forces']['contactFront_r'][1]<0
    l_slide = abs(state_desc['body_vel']['toes_l'][0])*l_toe_contact + abs(state_desc['body_vel']['calcn_l'][0])*l_heel_contact
    r_slide = abs(state_desc['body_vel']['toes_r'][0])*r_toe_contact + abs(state_desc['body_vel']['calcn_r'][0])*r_heel_contact
    return -(l_slide+r_slide)*dt

####################################################################################################### modified env

class L2RunEnvMod(L2RunEnv):
    model_path = os.path.join(os.path.dirname(__file__), '../models/gait9dof18musc_20170320.osim')
    
    MASS = 75.16460000000001 # 11.777 + 2*(9.3014 + 3.7075 + 0.1 + 1.25 + 0.2166) + 34.2366
    G = 9.80665 
    
    def __init__(self, reward_weight=None, action_limit=None, target_speed_range=[0.7, 1.3], own_policy=None, 
                 muscle_synergy=None, *args, **kwargs):

        self.target_speed_range = target_speed_range
        self.own_policy = own_policy
        self.muscle_synergy = muscle_synergy
        super(L2RunEnvMod, self).__init__(*args, **kwargs)
        self.test_own_policy()
        
        # added footstep related parameters
        self.footstep = {}
        self.footstep['n'] = 0
        self.footstep['side'] = 0.5 # 1 means right, 0 means left
        self.footstep['new'] = False
        self.footstep['r_contact'] = 1
        self.footstep['l_contact'] = 1
        self.footstep['d_reward'] = 0
        self.footstep['old_location'] = 0
        self.footstep['location'] = 0
        
        # read the reward_weight vector
        if reward_weight == None:
            self.reward_weight = np.ones(self.get_reward_size())
        elif np.array(reward_weight).all()>=0 and len(reward_weight)==self.get_reward_size():
            self.reward_weight = np.array(reward_weight)
        else:
            raise ValueError('Reward weight values should be positive with size {}.'.format(self.get_reward_size()))
        
        # read the action limit vector
        if action_limit == None:
            self.action_limit = np.ones(self.get_action_space_size())
        elif 0<=np.array(action_limit).all()<=1 and len(action_limit)==self.get_action_space_size():
            self.action_limit = np.array(action_limit)
        else:
            raise ValueError('Action limit value should be between 0 and 1 with size {}.'.format(self.get_action_space_size()))
            
    def test_own_policy(self):
        if self.own_policy is None:
            pass
        else:
            test_input = np.ones(self.get_observation_space_size())
            test_output = self.own_policy(test_input)
            if len(test_output) != self.osim_model.get_action_space_size():
                raise ValueError('The own_policy output should have length of {}'.format(self.osim_model.get_action_space_size()))
            
    def reset(self):
        self.footstep['n'] = 0
        self.footstep['side'] = 0.5 # 1 means right, 0 means left
        self.footstep['new'] = False
        self.footstep['r_contact'] = 1
        self.footstep['l_contact'] = 1
        self.footstep['d_reward'] = 0
        self.footstep['step_length'] = 0
        self.footstep['location'] = 0
        self.target_speed = np.random.uniform(self.target_speed_range[0], self.target_speed_range[1])
        return super().reset()
    
    def step(self, action, project = True):
        # perform action clipping
        if self.own_policy is not None:
            own_action = self.own_policy(self.get_observation())
            action = action + own_action
        clipedAction = np.clip(action, [0]*self.get_action_space_size(), self.action_limit)
        self.prev_state_desc = self.get_state_desc()        
        self.osim_model.actuate(clipedAction)
        self.osim_model.integrate()
        self.update_footstep()
        
        obs = self.get_observation()
        
        return [obs, 
                self.reward(self.osim_model.istep), 
                self.is_done() or (self.osim_model.istep >= self.spec.timestep_limit), 
                {}]
                  
    def update_footstep(self):
        state_desc = self.get_state_desc()

        # update foot contact conditions
        l_force = state_desc['forces']['contactFront_l'][1]+state_desc['forces']['contactFront_l'][1]
        r_force = state_desc['forces']['contactFront_r'][1]+state_desc['forces']['contactFront_r'][1]
        l_contact = True if l_force < -0.3*(self.MASS*self.G) else False
        r_contact = True if r_force < -0.3*(self.MASS*self.G) else False
        
        # footstep_side is updated at the moment when one foot touches the ground. 
        # footstep_side is not updated when both feet touches the same time or no touch down happens
        if (not self.footstep['r_contact'] and r_contact) and (not (not self.footstep['l_contact'] and l_contact)):
            footstep_side = 1
        elif (not (not self.footstep['r_contact'] and r_contact)) and (not self.footstep['l_contact'] and l_contact):
            footstep_side = 0
        else:
            footstep_side = self.footstep['side']
        
        # update the footstep count and new footstep status
        if footstep_side != self.footstep['side']:
            self.footstep['new'] = True
            self.footstep['n'] += 1
            if footstep_side == 1:
                step_location = state_desc['body_pos']['talus_r'][0]
            elif footstep_side == 0:
                step_location = state_desc['body_pos']['talus_l'][0]
            self.footstep['step_length'] = step_location - self.footstep['location']
            self.footstep['location'] = step_location
        else:
            self.footstep['new'] = False
        
        # overwrite footstep parameters
        self.footstep['side'] = footstep_side
        self.footstep['r_contact'] = r_contact
        self.footstep['l_contact'] = l_contact
    
    def get_observation(self):
        state_desc = self.get_state_desc()
        return get_observation_vec(state_desc, self.target_speed)
    
    def get_action_space_size(self):
        if self.muscle_synergy is not None:
            return 4
        else:
            return self.osim_model.get_action_space_size()
        
    def get_observation_space_size(self):
        return 36 # either list or int
    
    def get_reward_size(self):
        return 9 # number of individual terms in the reward function
    
    def get_reward_names(self):
        return ['forward_reward', 'survival_reward', 'torso_reward', 'joint_reward', 'stability_reward', 
                'act_reward', 'footstep_reward', 'jerk_reward', 'slide_reward']
    
    def reward(self, istep):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        dt = self.osim_model.stepsize
        if not prev_state_desc:
            return 0
        
        ######################################################### 
        forward_reward_exp = get_forward_reward(state_desc, self.target_speed, dt, mode='exp')
        forward_reward_qua = get_forward_reward(state_desc, self.target_speed, dt, mode='qua')
        forward_reward_simple = get_forward_reward(state_desc, self.target_speed, dt, mode='simple')
            
        ######################################################### 
        survival_reward = get_survival_reward(state_desc, dt)
        
        ######################################################### 
        torso_reward = get_torso_reward(state_desc, dt)
        
        ######################################################### 
        joint_reward = get_joint_reward(state_desc, dt)
        
        ######################################################### 
        stability_reward = get_stability_reward(state_desc, dt)
        
        ######################################################### 
        act_reward = get_act_reward(state_desc, dt)
        exc_reward = get_exc_reward(state_desc, dt)
        
        ######################################################### 
        self.footstep['d_reward'] += dt
        if self.footstep['new']:
            # footstep_reward = np.clip(self.footstep['d_reward'], 0, 1)
            footstep_reward = self.footstep['d_reward']
            self.footstep['d_reward'] = 0
        else:
            footstep_reward = 0

        ######################################################### 
        jerk_reward = get_jerk_reward(state_desc, prev_state_desc, dt)

        #########################################################
        slide_reward = get_slide_reward(state_desc, dt)
        
        #########################################################

        # @@@ Add function to compute tracking reward

        #########################################################
        if state_desc["misc"]["mass_center_pos"][0] < 1.:       # @@@ Comment out the first weighting scheme
            self.reward_list = np.array([1.*forward_reward_qua, 
                                         1.*survival_reward, 
                                         1.*torso_reward, 
                                         1.*joint_reward, 
                                         0.2*stability_reward,
                                         0.5*exc_reward, 
                                         1.*footstep_reward, 
                                         1.*jerk_reward,
                                         1.*slide_reward])
        else:
            self.reward_list = np.array([1.*forward_reward_exp, # @@@ Add new reward here
                                         1.*survival_reward, 
                                         1.*torso_reward, 
                                         1.*joint_reward, 
                                         1.*stability_reward, 
                                         1.*exc_reward, 
                                         1.*footstep_reward, 
                                         1.*jerk_reward,
                                         1.*slide_reward])
        
        return self.reward_weight@self.reward_list
        
        
# class L2RunEnvModDisturb(L2RunEnvDisturb):
#     # model_path = os.path.join(os.path.dirname(__file__), '../models/gait9dof18musc_disturb_20170320.osim')
#     model_path = os.path.join(os.path.dirname(__file__), '../models/gait9dof18musc_disturb.osim')   
#     ... same as the previous class