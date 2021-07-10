from tqdm.notebook import tqdm
import numpy as np
import os
import bz2
import pickle
import scipy.interpolate as interp
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
import pprint
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import clear_output, HTML


def single_plot(dataList, xlim=None, ylim=None, xlabel=None, ylabel=None, label=None):
    '''
    generate a single plot using multiple data in the dataList
    individual data shoould either have 
    - the same format as the output of get_gait_df
    - OR four-column format with gait%, mean, upper limit, lower limit
    dataList: a list of individual data
    axisLimit: [xlim, ylim]
    axisLabel: [xlabel, ylabel]
    '''
    color_index = 0
    data_num = len(dataList)
    if label == None:
        label = [None]*data_num
        
    for i in range(data_num):
        data = dataList[i]
        color = 'C'+ str(color_index)
        color_index += 1
        
        if not isinstance(data, pd.DataFrame):
            if data is None:
                pass
            else:
                raise ValueError('Data in dataList must be DataFrame')
        elif len(data.columns) == 2: # same style as the output of self.get_gait_df
            ax = sns.lineplot(x=data.columns[0], y=data.columns[1], data=data, ci='sd', color=color, label=label[i], legend=False)
        elif len(data.columns) == 4: # gait%, mean, upper limit, lower limit
            ax = sns.lineplot(x=data.columns[0], y=data.columns[1],data=data, ci='sd', color=color, label=label[i], legend=False)
            plt.fill_between(data[data.columns[0]], data[data.columns[2]], data[data.columns[3]], alpha=0.2, color=color)
        else:
            raise ValueError('DataFrame structure not understood')
    
    plt.locator_params(nbins=3)
    plt.xlim(0, 100)
    if xlim != None:
        plt.xlim(xlim)
    if ylim != None:
        plt.ylim(ylim)
    if xlabel != None:
        plt.xlabel(xlabel)
    if ylabel != None:
        plt.ylabel(ylabel)

    return ax
    
def multiple_plot(dataSetList, size=None, arrange=None, xlimls=None, ylimls=None, xlabells=None, ylabells=None, legends=None):
    '''
    generate multiple plots with provided dataSetList
    dataSetList: [Set_1_for_plot_1, Set_2_for_plot_2]
    arrange: a list including [# of rows, # of columns] for the subplots
    xlimls: a list of xlim values
    ylimls: a list of ylim values
    xlabells: a list of xlabel strings
    ylabells: a list of ylabel strings
    legends: a list of labels with len equal to # of lines in one single plot
    '''
    fig_num = len(dataSetList)
    if arrange == None:
        arrange = [1, fig_num]
    if xlimls == None:
        xlimls = [None]*fig_num
    if ylimls == None:
        ylimls = [None]*fig_num
    if xlabells == None:
        xlabells = [None]*fig_num
    if ylabells == None:
        ylabells = [None]*fig_num
            
    #print(fig_num, xlimls)
    if size != None:
        fig = plt.figure(figsize=size)
    else:
        fig = plt.figure()
        
    for i in range(fig_num):
        ax = plt.subplot(arrange[0], arrange[1], i+1)
        ax = single_plot(dataSetList[i], xlim=xlimls[i], ylim=ylimls[i], xlabel=xlabells[i], ylabel=ylabells[i], label=legends)
        
    if legends != None:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(legends))
    fig.tight_layout()
    return fig

def plot_single_event_raster(x_vec, intensity_vec, y_pos, line_width):
    '''
    generate a raster plot with many small vertical lines
    x_vec: a vector indicating the x location of vertical lines
    intensity_vec: a vector indicating the intensity of each vertical lines
    y_pos: the y position to generate the plot
    line_width: the length of each vertical line
    '''
    if len(x_vec) != len(intensity_vec):
        raise Exception('x_vec and intensity_vec have different lengths.')
    for i in range(len(x_vec)):
        plt.vlines(x_vec[i], y_pos-line_width/2, y_pos+line_width/2, colors=(0,0,0,intensity_vec[i]))
        
def plot_multi_event_raster(data, *args, **kwargs):
    '''
    plot raster plots
    data: a DataFrame with each column as a single raster plot
    '''
    names = data.columns
    fig, ax = plt.subplots(*args, **kwargs)
    for i in range(len(names)):
        pos, = np.where(data[names[i]])
        intensity = np.array(data[names[i]])[pos]
        plot_single_event_raster(pos, intensity, i, 1)
    plt.yticks(np.arange(0, len(names)))
    ax.set_yticklabels(names)
    
    # set colorbar
    cmap = plt.get_cmap('Greys')
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm)
    return fig

def interpolate_gait_data(data, t=None, length=101):
    '''
    interpolate a vector to a desired length
    data: a vector of data (same size as t)
    t: a vector of time, if None, assume data is evenly spaced
    length: desired output vector length
    '''
    data = np.array(data)
    if t is None: # assuming y has equal steps
        dataInterp = interp.interp1d(np.arange(data.size), data)
        dataStrech = dataInterp(np.linspace(0,data.size-1,length))
    else:
        t = np.array(t)
        dataInterp = interp.interp1d(t, data)
        dataStrech = dataInterp(np.linspace(t[0],t[-1],length))
    gaitPercent = np.linspace(0,100,length)
    return gaitPercent, dataStrech

def shift_gait(gait_df, res, shift):
    '''
    shift gait% by some number of points
    gait_df: output from gaitAnalysis.get_gait_df
    res [int]: full traj resolution
    shift [int]: number of points to shift
    '''
    gait_df['gait%'] = np.mod((gait_df['gait%']+shift), res)
    return gait_df 

def RMSE(a, b):
    return np.sqrt((a-b)@(a-b)/len(a))

def Pearson(a, b):
    return pearsonr(a, b)[0]

def DTW(a, b):
    '''
    The method of trajectory comparison is from this paper: 
    https://pdfs.semanticscholar.org/05a2/0cde15e172fc82f32774dd0cf4fe5827cad2.pdf
    '''
    dist, path = fastdtw(a, b, dist=euclidean)
    return dist

class gaitAnalysis():
    def __init__(self, GRF_treshold=100, init_crop=2, data_res=101, shift=0):
        '''
        initialize gaitAnalysis
        GRF_treshold: the force level used to seperate gait
        init_crop: # of initial gait cycles to disgard during analysis
        data_res: data resolution when interpolating the gait data
        '''
        self.data_res = data_res
        self.GRF_treshold = GRF_treshold
        self.init_crop = init_crop
        self.shift=shift
        
    def collect_traj_data(self, model, env, max_step = 1000, episode_length=1000, deterministic=True):
        '''
        generate and output gait trajectories in the form of a dictionary
        model: DRL agent model, output from Stable-baselines
        env: environment in Gym style (L2RunEnvMod)
        max_step: total number of time steps to run
        verbose: display of progress and the reward values during simulation
        '''
        
        self.traj_data = {'act_hist': [], 'state_hist': [], 'reward_hist': [], 'episode_hist': [], 'time':[],
                          'reward_names': env.get_reward_names(), 'reward_weight': env.reward_weight}
        
        obs = env.reset()
        episode_step = 0
        for i_step in tqdm(range(max_step)):
            if episode_step == 0:
                self.traj_data['episode_hist'].append(i_step)
            
            action,_ = model.predict(obs, deterministic=deterministic)
            obs, reward, dones, info = env.step(action)
            self.traj_data['state_hist'].append(env.get_state_desc())
            self.traj_data['reward_hist'].append(env.reward_list)
            self.traj_data['act_hist'].append(action)
            self.traj_data['time'].append(env.osim_model.get_state().getTime())
            episode_step += 1
            
            if dones is True or episode_step == episode_length:
                obs = env.reset()
                episode_step = 0
        
        self.get_gait_pos()

        return self.traj_data
    
    def extract_state(self, state_loc):
        '''
        extract data from self.traj_data
        data_loc: a list of strings / int to indicate the data location within the dictionary
        '''
        data = []
        for d in self.traj_data['state_hist']:
            for loc in state_loc:
                d = d[loc]
            data.append(d)
        return np.array(data)
    
    def save_traj_data(self, file_path):
        '''
        save the traj_data as a compressed pickle file at the file_path
        '''
        if not file_path.endswith('.pbz2'):
            file_path = file_path+'.pbz2'
        if os.path.exists(file_path):
            raise Exception("existing data found!")
        else:
            with bz2.BZ2File(file_path, 'w') as f: 
                pickle.dump(self.traj_data, f)
                
    def load_traj_data(self, file_path):
        '''
        load the traj_data from a compressed pickle file at the file_path
        '''
        if not file_path.endswith('.pbz2'):
            file_path = file_path+'.pbz2'
        data = bz2.BZ2File(file_path, 'rb')
        self.traj_data = pickle.load(data)
        self.get_gait_pos()
        
    def get_single_leg_gait_pos(self, GRF_traj): 
        '''
        locate the starting points of gait cycles based on ground reaction forces (GRF)
        GRF_traj: a vector of ground reaction forces
        '''
        filtered_data = -savgol_filter(GRF_traj, 19, 1) # smooth data

        # find gait rough starting point on the smooth data
        threshold_data = filtered_data > self.GRF_treshold
        gait_start = np.where(threshold_data * (1-np.roll(threshold_data,1)))[0]

        # exclude self.init_crop gait cycles at the beginning of each episode
        gait_pos = []
        episode_hist = self.traj_data['episode_hist']
        for i in range(1, len(gait_start)):
            j = np.clip(i-self.init_crop, 0, None) # limit left end of evaluation to >= 0
            hasEpisodeStart = [gait_start[j] <= episode_pos < gait_start[i] for episode_pos in episode_hist]
            if sum(hasEpisodeStart) == 0: # does not contain eqisode starting location
                gait_pos.append([gait_start[i-1], gait_start[i]])
        return gait_pos
        
    def get_gait_pos(self):
        '''
        compute the gait starting location based on GRFs
        '''
        GRF_l = self.extract_state(['forces', 'contactFront_l', 1]) + self.extract_state(['forces', 'contactHeel_l', 1])
        GRF_r = self.extract_state(['forces', 'contactFront_r', 1]) + self.extract_state(['forces', 'contactHeel_r', 1])
        self.gait_pos_l = self.get_single_leg_gait_pos(GRF_l)
        self.gait_pos_r = self.get_single_leg_gait_pos(GRF_r)
        
    def get_gait_df(self, state_loc, state_name=None, side=None, **kwargs):
        '''
        seperate and interpolate data specified in state_loc based on gait cycles
        data_loc: a list of strings / int to indicate the data location within the dictionary
        side: None, or left, or right
        '''
        
        # process side automatically
        state_loc_str = [loc for loc in state_loc if type(loc)==str]
        isR = sum([loc.endswith('_r') for loc in state_loc_str]) >= 1
        isL = sum([loc.endswith('_l') for loc in state_loc_str]) >= 1
        if (side == None and isR and not isL) or (side == 'right'):
            gait_pos = self.gait_pos_r
        elif (side == None and isL and not isR) or (side == 'left'):
            gait_pos = self.gait_pos_l
        else:
            raise ValueError('side needs to be specified: either left or right')        
        
        state = self.extract_state(state_loc)
        if state_name == None:
            state_name = state_loc[1]
        gait_df = pd.DataFrame({})
        for i in range(len(gait_pos)):
            gait_data = state[gait_pos[i][0]:gait_pos[i][1]]
            gaitPercent, dataInter  = interpolate_gait_data(gait_data, length=self.data_res)    
            gait_df = gait_df.append(pd.DataFrame({'gait%':gaitPercent, state_name:dataInter}), ignore_index=True)
            
        gait_df = shift_gait(gait_df, self.data_res, self.shift)
        return gait_df
    
    def get_avg_gait_df(self, gait_df):
        '''
        reduce the multi-gait df from get_gait_df to one avg gait cycle
        '''
        return gait_df.groupby('gait%').mean().reset_index()
    
    def cal_stride_time(self):        
        stride_time_l = [self.traj_data['time'][i[1]]-self.traj_data['time'][i[0]] for i in self.gait_pos_l]
        stride_time_r = [self.traj_data['time'][i[1]]-self.traj_data['time'][i[0]] for i in self.gait_pos_r]
        return stride_time_l, stride_time_r
    
    def cal_stance_percent(self):
        GRF_l = self.extract_state(['forces', 'contactFront_l', 1]) + self.extract_state(['forces', 'contactHeel_l', 1])
        stance_l = []
        for pos in self.gait_pos_l:
            isContact = savgol_filter(GRF_l[pos[0]:pos[1]], 7, 1) < -self.GRF_treshold
            stance_l.append(sum(isContact)/len(isContact))
            
        GRF_r = self.extract_state(['forces', 'contactFront_r', 1]) + self.extract_state(['forces', 'contactHeel_r', 1])
        stance_r = []
        for pos in self.gait_pos_r:
            isContact = savgol_filter(GRF_r[pos[0]:pos[1]], 7, 1) < -self.GRF_treshold
            stance_r.append(sum(isContact)/len(isContact))
        return stance_l, stance_r
    
    def cal_step_time(self):
        step_pos = np.sort([i[0] for i in self.gait_pos_l] + [i[0] for i in self.gait_pos_r])
        time = self.traj_data['time']
        episode_hist = self.traj_data['episode_hist']
        step_time = [time[step_pos[i]] - time[step_pos[i-1]] for i in range(1, len(step_pos)) 
                     if sum([step_pos[i-1]<=j<step_pos[i] for j in episode_hist])==0] # if contains episode start
        return step_time
    
    def cal_stride_len(self):
        calcn_l_x = self.extract_state(['body_pos', 'calcn_l', 0])
        calcn_r_x = self.extract_state(['body_pos', 'calcn_r', 0])
        stride_len_l = [calcn_l_x[self.gait_pos_l[i][1]]-calcn_l_x[self.gait_pos_l[i][0]] 
                        for i in range(len(self.gait_pos_l))]
        stride_len_r = [calcn_r_x[self.gait_pos_r[i][1]]-calcn_r_x[self.gait_pos_r[i][0]] 
                        for i in range(len(self.gait_pos_r))]
        return stride_len_l, stride_len_r
    
    def cal_speed(self):
        speed = self.extract_state(['misc', 'mass_center_vel', 0])
        return speed
    
    def cal_stride_params(self, verbose=True):
        '''
        calculate many gait related parameters
        '''
        params = {}
        
        speed = self.cal_speed()
        params['speed'] = [np.mean(speed), np.std(speed)]
        
        step_time = self.cal_step_time()
        params['step time'] = [np.mean(step_time), np.std(step_time)]
        
        stride_t_l, stride_t_r = self.cal_stride_time()
        stride_time_l = [np.mean(stride_t_l), np.std(stride_t_l)]
        stride_time_r = [np.mean(stride_t_r), np.std(stride_t_r)]
        stride_time_avg = [np.mean(stride_t_l+stride_t_r), np.std(stride_t_l+stride_t_r)]
        params['stride time'] = {'left': stride_time_l, 'right': stride_time_r, 'avg': stride_time_avg}
        
        stance_l, stance_r = self.cal_stance_percent()
        swing_l = [1-i for i in stance_l]
        swing_r = [1-i for i in stance_r]
        stance_p_l = [np.mean(stance_l), np.std(stance_l)]
        stance_p_r = [np.mean(stance_r), np.std(stance_r)]
        stance_p_avg = [np.mean(stance_l+stance_r), np.std(stance_l+stance_r)]
        swing_p_l = [np.mean(swing_l), np.std(swing_l)]
        swing_p_r = [np.mean(swing_r), np.std(swing_r)]
        swing_p_avg = [np.mean(swing_l+swing_r), np.std(swing_l+swing_r)]
        params['stance%'] = {'left': stance_p_l, 'right': stance_p_r, 'avg': stance_p_avg}
        params['swing%'] = {'left': swing_p_l, 'right': swing_p_r, 'avg': swing_p_avg}
        
        stance_t_l = [stance_l[i]*stride_t_l[i] for i in range(len(stance_l))]
        stance_t_r = [stance_r[i]*stride_t_r[i] for i in range(len(stance_r))]
        swing_t_l = [swing_l[i]*stride_t_l[i] for i in range(len(swing_l))]
        swing_t_r = [swing_r[i]*stride_t_r[i] for i in range(len(swing_r))]
        
        stance_time_l = [np.mean(stance_t_l), np.std(stance_t_l)]
        stance_time_r = [np.mean(stance_t_r), np.std(stance_t_r)]
        stance_time_avg = [np.mean(stance_t_l+stance_t_r), np.std(stance_t_l+stance_t_r)]
        swing_time_l = [np.mean(swing_t_l), np.std(swing_t_l)]
        swing_time_r = [np.mean(swing_t_r), np.std(swing_t_r)]
        swing_time_avg = [np.mean(swing_t_l+swing_t_r), np.std(swing_t_l+swing_t_r)]
        params['stance_time'] = {'left': stance_time_l, 'right': stance_time_r, 'avg': stance_time_avg}
        params['swing_time'] = {'left': swing_time_l, 'right': swing_time_r, 'avg': swing_time_avg}
        
        def print_params(data):
            '''
            recursively print out the parameters
            '''
            if type(data) == dict:
                for key in data.keys():
                    if key in ['left', 'right', 'avg']:
                        print(key, end = ' ')
                    else:
                        print('\n'+key)
                    print_params(data[key])
            elif type(data) == list:
                if len(data) == 2:
                    print('mean: {:.02f} with SD: {:.02f}'.format(data[0], data[1]))
            
        if verbose:
            print_params(params)
            
        return params

    def compare_joint_angles(self, exp_data, side='avg', verbose=True):
        '''
        compare joint angles between agent in simulation and experimental data
        side: a string of left, right, or avg
        exp_data: output of self.load_exp_joint_data (can only take one set of experimental data)
        '''
        simAngles = [self.get_avg_gait_df(angle) for angle in self.get_joint_angles(side)]
        comp = {}
        for i in range(len(simAngles)):
            rmse = RMSE(simAngles[i].iloc[:,1], exp_data[i][0].iloc[:,1])
            pear = Pearson(simAngles[i].iloc[:,1], exp_data[i][0].iloc[:,1])
            dtw = DTW(simAngles[i].iloc[:,1], exp_data[i][0].iloc[:,1])
            comp[simAngles[i].columns[1]] = {'RMSE': rmse, 'Pearson': pear, 'DTW': dtw}
            
        if verbose:
            for c in comp.keys():
                print('\n'+c)
                for t in comp[c].keys():
                    print('{}: {:0.2f}'.format(t, comp[c][t]))
            
        return comp
    
    def get_joint_angles(self, side):
        '''
        obtain joint angle data for one side or average between two legs
        side: a string of left, right, or avg
        '''
        hipAngle_l = self.get_gait_df(['joint_pos','hip_l',0], state_name='hip')
        kneeAngle_l = self.get_gait_df(['joint_pos','knee_l',0], state_name='knee')
        ankleAngle_l = self.get_gait_df(['joint_pos','ankle_l',0], state_name='ankle')
        hipAngle_r = self.get_gait_df(['joint_pos','hip_r',0], state_name='hip')
        kneeAngle_r = self.get_gait_df(['joint_pos','knee_r',0], state_name='knee')
        ankleAngle_r = self.get_gait_df(['joint_pos','ankle_r',0], state_name='ankle')
        if side == 'left':
            hipAngle = hipAngle_l
            kneeAngle = kneeAngle_l
            ankleAngle = ankleAngle_l
        elif side == 'right':
            hipAngle = hipAngle_r
            kneeAngle = kneeAngle_r
            ankleAngle = ankleAngle_r
        elif side == 'avg':
            hipAngle = hipAngle_l.append(hipAngle_r, ignore_index=True)
            kneeAngle = kneeAngle_l.append(kneeAngle_r, ignore_index=True)
            ankleAngle = ankleAngle_l.append(ankleAngle_r, ignore_index=True)
        return [hipAngle, kneeAngle, ankleAngle]
    
    def get_muscle_act(self, side):
        '''
        obtain muscle activation data for one side or average between two legs
        side: a string of left, right, or avg
        '''
        muscle_act = {}
        for muscle in self.traj_data['state_hist'][0]['muscles'].keys():
            muscle_act[muscle] = self.get_gait_df(['muscles',muscle,'activation'], state_name=muscle[0:len(muscle)-2])
            
        muscle_act_l = {muscle[0:len(muscle)-2]:muscle_act[muscle] for muscle in muscle_act.keys() if muscle.endswith('_l')}
        muscle_act_r = {muscle[0:len(muscle)-2]:muscle_act[muscle] for muscle in muscle_act.keys() if muscle.endswith('_r')}
        
        if side == 'left':
            muscle_activation = muscle_act_l
        elif side == 'right':
            muscle_activation = muscle_act_r
        elif side == 'avg':
            muscle_activation = {muscle:muscle_act_l[muscle].append(muscle_act_r[muscle], ignore_index=True) for muscle in muscle_act_l.keys()}
        return muscle_activation
    
    def plot_joint_angles(self, exp_data = None, exp_legends = None, side='avg', **kwargs):
        '''
        plot joint angles
        exp_data: output from self.load_exp_joint_data
        exp_legends: a list specifying the legends
        side: left, right, or avg
        '''
        jointAngles = self.get_joint_angles(side)
        data = [[i] for i in jointAngles]
        ylabells = ['hip [rad]','knee [rad]','ankle [rad]']
        if exp_data is None:
            fig = multiple_plot(data, legends=[side], ylabells=ylabells, **kwargs)
        else:
            data = [data[i]+exp_data[i] for i in range(3)]
            if exp_legends is None:
                exp_legends = ['exp'+str(i) for i in range(len(exp_data[0]))]
            fig = multiple_plot(data, legends=[side]+exp_legends, ylabells=ylabells, **kwargs)
            
        return fig
    
    def plot_muscle_activation(self, exp_data = None, exp_legends = None, side = 'avg', **kwargs):
        '''
        plot muscle activation
        exp_data: output from self.load_exp_activation_data
        exp_legends: a list specifying the legends
        side: left, right, or avg
        '''
        sim_data = self.get_muscle_act(side)
        sim_data = [[sim_data[muscle]] for muscle in sim_data.keys()]
        if exp_data is None:
            fig = multiple_plot(data, legends=[side], **kwargs)
        else:
            # generate data list by matching muscle names
            data = []
            for i in range(len(sim_data)):
                for j in range(len(exp_data)):
                    if sim_data[i][0].keys()[1] == exp_data[j][0].keys()[1]:
                        data.append(sim_data[i]+exp_data[j])
                        break
                    elif j == len(exp_data)-1:
                        data.append(sim_data[i]+[None])     
            if exp_legends is None:
                exp_legends = ['exp'+str(i) for i in range(len(exp_data[0]))]
            fig = multiple_plot(data, legends=[side]+exp_legends, ylimls=[[0,0.7]]*len(data), **kwargs)
            
        return fig
            
    def plot_reward_corr(self, weights=None, size=None):
        '''
        plot reward correlations among all reward terms
        weights: a list of reward weights (reward terms with weight=0 is not plotted)
        size: image size
        '''
        reward_hist = np.array(self.traj_data['reward_hist'])
        if weights == None:
            weights = [1]*reward_hist.shape[1]
        elif len(weights) != reward_hist.shape[1]:
            raise ValueError('weights should have the same dimension as the # of reward terms')
        reward_hist = reward_hist*np.array(weights)

        reward_names = np.array(self.traj_data['reward_names'])
        reward_hist = np.delete(reward_hist, np.where([weight==0 for weight in weights]), axis=1)
        reward_names = np.delete(reward_names, np.where([weight==0 for weight in weights]), axis=0)
        reward_cov = np.round(np.corrcoef(reward_hist.T), 2)

        fig = plt.figure(figsize=size)
        ax = plt.axes([0, 0.05, 0.9, 0.9 ])
        cax = plt.axes([0.95, 0.05, 0.05,0.9 ])
        im = ax.imshow(reward_cov)

        ax.set_xticks(np.arange(len(reward_names)))
        ax.set_yticks(np.arange(len(reward_names)))

        ax.set_xticklabels(reward_names)
        ax.set_yticklabels(reward_names)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(reward_names)):
            for j in range(len(reward_names)):
                text = ax.text(j, i, reward_cov[i, j],
                               ha="center", va="center", color="w")

        ax.set_title("reward term correlation")
        fig.colorbar(mappable=im, cax=cax)
        fig.tight_layout()

        return fig
    
    def plot_excitation_raster(self, steps=None, side='left', *args, **kwargs):
        if steps is None:
            steps = np.clip(len(self.traj_data['act_hist']), 0, 1000)
            
        excitation = np.array(self.traj_data['act_hist'])
        names = list(self.traj_data['state_hist'][0]['muscles'].keys())
        muscle_exc = {}
        if side == 'left':
            names = [name for name in names if name.endswith('_l')]
            for i in range(len(names)):
                muscle_exc[names[i]] = excitation[0:steps,i+9]
        elif side == 'right':
            names = [name for name in names if name.endswith('_r')]
            for i in range(len(names)):
                muscle_exc[names[i]] = excitation[0:steps,i]
        else:
            raise NotImplementedError
        
        fig = plot_multi_event_raster(pd.DataFrame(muscle_exc), *args, **kwargs)
        return fig
        
    def plot_activation_raster(self, steps=None, side='left', *args, **kwargs):
        if steps is None:
            steps = np.clip(len(self.traj_data['act_hist']), 0, 1000)
        
        names = list(self.traj_data['state_hist'][0]['muscles'].keys())    
        muscle_act = {}    
        if side == 'left':
            names = [name for name in names if name.endswith('_l')]
        elif side == 'right':
            names = [name for name in names if name.endswith('_r')]
        else:
            raise NotImplementedError
            
        for i in range(len(names)):
            muscle_act[names[i]] = self.extract_state(['muscles', names[i], 'activation'])[0:steps]
        
        fig = plot_multi_event_raster(pd.DataFrame(muscle_act), *args, **kwargs)
        return fig
    
    def plot_total_muscle_activation(self, exp_act_data):
        '''
        plot muscle activation comparison between the simulation results and the CMC from OpenSim
        exp_act_data: output from load_exp_activation_data
        '''
        sim_act = pd.DataFrame({})
        for muscle in self.traj_data['state_hist'][0]['muscles'].keys():
            sim_act[muscle] = [np.sum(self.extract_state(['muscles', muscle, 'activation']))]
        total_sim_act = np.sum(np.sum(sim_act))
        sim_act = sim_act/total_sim_act*2


        exp_act = pd.DataFrame({})
        for i in range(len(exp_act_data)):
            muscle = exp_act_data[i][0].keys()[1]
            exp_act[muscle] = [np.sum(exp_act_data[i][0].iloc[:, 1])]
        total_exp_act = np.sum(np.sum(exp_act))
        exp_act = exp_act/total_exp_act

        # Plot
        width=0.35

        plt.figure(figsize=[8,4])
        y_pos = np.arange(len(exp_act.columns))

        plt.subplot(121)
        plt.barh(y_pos, exp_act.iloc[0,:], width, label='exp')
        plt.barh(y_pos+width, sim_act.iloc[0,0:9], width, label='sim')
        plt.yticks(y_pos+width/2, exp_act.columns)
        plt.xlabel('Relative Muscle Activation')
        plt.title('Right Leg')
        plt.xlim([0,0.5])

        plt.subplot(122)
        plt.barh(y_pos, exp_act.iloc[0,:], width, label='exp')
        plt.barh(y_pos+width, sim_act.iloc[0,9:18], width, label='sim')
        plt.yticks(y_pos+width/2, exp_act.columns)
        plt.legend(loc='best')
        plt.xlabel('Relative Muscle Activation')
        plt.title('Left Leg')
        plt.xlim([0,0.5])

        plt.subplots_adjust(wspace=0.8)
        
    def plot_reward(self, weights=None, ylim=None, size=[15,3], steps=None):
        '''
        plot reward history of a trajectory
        skip_index: the index of individual reward terms to ignore during plotting (1:ignore, 0:plot)
        apply_weight: to plot reward with weighting factors or without weighting factors
        '''
        reward_names = self.traj_data['reward_names']
        reward_hist = np.array(self.traj_data['reward_hist'])

        if weights is None:
            weights = [1] * len(reward_names)
            
        if steps is None:
            steps = [0, reward_hist.shape[0]]

        fig = plt.figure(figsize=size)
        if ylim is not None:
            plt.ylim(ylim)
        for i in range(len(reward_names)):
            if weights[i] != 0:
                plt.plot(reward_hist[steps[0]:steps[1],i]*weights[i], label=reward_names[i])
        plt.legend()

        return fig
    
    def load_exp_data(self, file_path):
        '''
        load experimental data and interpolate them to desired length
        file_path: path to the experimental csv file
        '''
        exp_data = pd.read_csv(file_path)
        
        # interpolate data
        gaitPercent = np.linspace(0,100,self.data_res)
        data_inter = pd.DataFrame({'gait%':gaitPercent})
        for i in range(1, len(exp_data.columns)):
            column_name = exp_data.columns[i]
            _,b = interpolate_gait_data(exp_data.iloc[:,i], t=exp_data.iloc[:,0], length=self.data_res)
            data_inter[column_name] = b
        return data_inter

    def load_exp_joint_data(self, file_path, mode):
        '''
        extract joint angles from experimental data
        file_path: path to the experimental csv file
        mode: extracted, Moco, or Osim, different modes are for experimental data in different format
        '''
        data_inter = self.load_exp_data(file_path)
        
        if mode == 'extracted':
            hip_exp = pd.concat([data_inter.iloc[:,0:1], data_inter.iloc[:,1:4]], axis=1, join="inner")
            knee_exp = pd.concat([data_inter.iloc[:,0:1], data_inter.iloc[:,4:7]], axis=1, join="inner")
            ankle_exp = pd.concat([data_inter.iloc[:,0:1], data_inter.iloc[:,7:10]], axis=1, join="inner")
        elif mode == 'Moco':
            hip_exp = pd.concat([data_inter.iloc[:,0:1], data_inter.iloc[:,5:6]], axis=1, join="inner")
            knee_exp = pd.concat([data_inter.iloc[:,0:1], data_inter.iloc[:,8:9]], axis=1, join="inner")
            ankle_exp = pd.concat([data_inter.iloc[:,0:1], data_inter.iloc[:,10:11]], axis=1, join="inner")
        elif mode == 'Osim':
            hip_exp = pd.concat([data_inter.iloc[:,0:1], data_inter.iloc[:,7:8]], axis=1, join="inner")
            knee_exp = pd.concat([data_inter.iloc[:,0:1], data_inter.iloc[:,9:10]], axis=1, join="inner")
            ankle_exp = pd.concat([data_inter.iloc[:,0:1], data_inter.iloc[:,11:12]], axis=1, join="inner")
        
        return [[hip_exp], [knee_exp], [ankle_exp]]
    
    def load_exp_activation_data(self, file_path, mode):
        '''
        extract muscle activation from experimental data
        file_path: path to the experimental csv file
        mode: extracted, Moco, or Osim, different modes are for experimental data in different format
        '''
        data_inter = self.load_exp_data(file_path)
        
        if mode == 'extracted':
            exp_activation = data_inter.iloc[:,1:len(data_inter.columns)]
        elif mode == 'Moco':
            exp_activation = data_inter.filter(regex='_r/activation')
            exp_activation = exp_activation.rename(columns={name:name[1:-13] for name in exp_activation.columns})
        elif mode == 'Osim':
            exp_activation = data_inter.filter(regex='_r/activation')
            exp_activation = exp_activation.rename(columns={name:name[10:-13] for name in exp_activation.columns})
        
        exp_activation = [[pd.concat([data_inter.iloc[:,0:1], exp_activation.loc[:,name]], axis=1, join="inner")]
                          for name in exp_activation.keys()]
        return exp_activation
    
    def get_stick_animation_data(self, max_frame, eCoM_coeff=0.09):
        
        def joint_loc(pelvisAngle, pelvis_y, lumbarAngle, hipAngle, kneeAngle, ankleAngle):
            torso_len = 0.5
            thigh_len = 0.41
            shank_len = 0.43
            foot_len = 0.18

            torsoAngle_global = pelvisAngle+lumbarAngle
            thighAngle_global = hipAngle+pelvisAngle
            shankAngle_global = kneeAngle + thighAngle_global
            footAngle_global = np.pi/2+ankleAngle+shankAngle_global

            pelvis = np.array([0,pelvis_y])
            head = pelvis + torso_len*np.array([-np.sin(torsoAngle_global), np.cos(torsoAngle_global)])
            knee = pelvis + thigh_len*np.array([np.sin(thighAngle_global), -np.cos(thighAngle_global)])
            ankle = knee + shank_len*np.array([np.sin(shankAngle_global), -np.cos(shankAngle_global)])
            toe = ankle + foot_len*np.array([np.sin(footAngle_global), -np.cos(footAngle_global)])
            
            return np.vstack((head, pelvis, knee, ankle, toe))
            
        #simulation data
        pelvisAngle = self.extract_state(['joint_pos', 'ground_pelvis', 0])
        pelvisPos_x = self.extract_state(['joint_pos', 'ground_pelvis', 1])
        pelvisPos_y = self.extract_state(['joint_pos', 'ground_pelvis', 2])
        hipAngle_l = self.extract_state(['joint_pos', 'hip_l', 0])
        kneeAngle_l = self.extract_state(['joint_pos', 'knee_l', 0])
        ankleAngle_l = self.extract_state(['joint_pos', 'ankle_l', 0])
        hipAngle_r = self.extract_state(['joint_pos', 'hip_r', 0])
        kneeAngle_r = self.extract_state(['joint_pos', 'knee_r', 0])
        ankleAngle_r = self.extract_state(['joint_pos', 'ankle_r', 0])
        CoM_pos_x = self.extract_state(["misc", "mass_center_pos", 0])
        CoM_acc_x = self.extract_state(["misc", "mass_center_acc", 0])
        ZMP = CoM_pos_x + eCoM_coeff*CoM_acc_x - pelvisPos_x
        
        loc_r = []
        loc_l = []
        for k in range(max_frame):
            loc_r.append(joint_loc(pelvisAngle[k], pelvisPos_y[k], -0.14, hipAngle_r[k], kneeAngle_r[k], ankleAngle_r[k]))
            loc_l.append(joint_loc(pelvisAngle[k], pelvisPos_y[k], -0.14, hipAngle_l[k], kneeAngle_l[k], ankleAngle_l[k]))
        return loc_l, loc_r, ZMP
            
    
    def get_stick_animation(self, frame_gap=10, max_frame=None, **kwargs):
        '''
        generate animation in the form of sticks
        frame_gap: # of miliseconds between each frame
        max_frame: max frame counts to be rendered (if greater than length of sim_data, then the length of sim_data is used)
        '''
        def animate(k):
            human_r.set_data((loc_r[k][:,0], loc_r[k][:,1]))
            human_l.set_data((loc_l[k][:,0], loc_l[k][:,1]))
            ZMP_loc.set_data((ZMP[k],0))
            clear_output(wait=True)
            print('{}/{}'.format(k+1, max_frame))

        if max_frame is None:
            max_frame = len(self.traj_data['state_hist'])
        else:
            max_frame = np.clip(max_frame, None, len(self.traj_data['state_hist']))

        loc_l, loc_r, ZMP = self.get_stick_animation_data(max_frame, **kwargs)
        
        #animate
        fig = plt.figure(figsize=[5,7.5])
        ground, = plt.plot([-1,1],[0,0])
        human_l, = plt.plot([], color='#273f8a')
        human_r, = plt.plot([], color='#d6000d')
        ZMP_loc, = plt.plot([], marker="o", markersize=5)
        plt.xlim([-0.6,0.6])
        plt.ylim([-0.3,1.5])

        anim = FuncAnimation(fig, animate, frames=max_frame, interval=frame_gap)
        video = anim.to_html5_video()
        html = HTML(video)
        display(html)
        plt.close()
    
