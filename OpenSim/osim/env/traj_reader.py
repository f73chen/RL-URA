import numpy as np
import pandas as pd
import scipy.interpolate as interp

class trajReader():
    def __init__(self, file_dir):
        _, self.header, self.traj = self.read_sto(file_dir)
        
    def get_traj(self):
        return self.traj
        
    def read_sto(self, file_dir):
        '''
        read the sto file into DataFrame
        '''
        file_name = file_dir[file_dir.rfind("/")+1:]
        try:
            file_num = int(file_name[0:file_name.find("_")])-1
        except:
            file_num = None

        # extract file header
        log = pd.read_csv(file_dir, header=None)[0]
        header_end = log.index[log == "endheader"][0]
        header = log[0:header_end].str.split("=", n = -1, expand = True)
        header = header.set_index(0)
        header = header[1].to_dict()

        for key in header.keys():
            try:
                header[key] = float(header[key])
                if header[key].is_integer():
                    header[key] = int(header[key])
            except:
                pass

        # extract trajectories
        traj = log[header_end+1:].str.split("\t", n = -1, expand = True)
        traj = traj.reset_index(drop=True)
        traj.columns = traj.iloc[0]
        traj = traj.drop(traj.index[0])
        traj = traj.astype("float")
        return file_num, header, traj
    
    def interpolate_traj(self, dt=0.01):
        '''
        interpolate trajectory to match certain temporal resolution
        '''
        start_time = self.traj.iloc[0,0]
        end_time = self.traj.iloc[-1,0]
        length = int((end_time - start_time)/dt)+1
        
        out = {}
        out["time"] = np.arange(0, end_time-start_time, dt)
        for name in self.traj.iloc[:,1:].columns: # skip the first column which is time
            out[name] = self.interpolate_data(self.traj.iloc[:,0], self.traj[name], out["time"])
        return pd.DataFrame(out)
    
    def interpolate_data(self, t, data, target_t):
        '''
        interpolate a vector to a desired length
        data: a vector of data (same size as t)
        t: a vector of time, if None, assume data is evenly spaced
        target_t: desired output time vector
        '''
        data = np.array(data)
        t = np.array(t)
        target_t = np.array(target_t)
        dataInterp = interp.interp1d(t, data)
        dataStrech = dataInterp(target_t)
        return dataStrech