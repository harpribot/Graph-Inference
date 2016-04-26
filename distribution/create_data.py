from scipy.stats import rv_discrete
import pandas as pd
import numpy as np

class Creator:
    def __init__(self, num_samples=500):
        self.num_samples = num_samples
        self.conditioned_on = {}
        self.distribution = {}
        self.var_range = {}
        self.samples = {}
        self.df = pd.DataFrame()

    def add_variable(self,varname,var_rn):
        self.var_range[varname] = var_rn

    def sample(self, dist,condn_name, *condn_on):
        '''
            dist : Distribution should be such that the one distributed on is at the end
        '''
        self.distribution[condn_name] = dist
        self.conditioned_on[condn_name] = condn_on
        self.__create_samples(condn_name)

    def __create_samples(self,var_nm):
        '''
            Ensure that all the samples have been obtained for previous dist when
            trying to sample new ones
        '''
        dist = self.distribution[var_nm]
        v_range = self.var_range[var_nm]
        condn_on = list(self.conditioned_on[var_nm])
        condn_on_samples = np.array([self.samples[x].tolist() for x in condn_on]).T
        if(condn_on_samples.shape[0] != 0):
            sample_vals = np.array([],dtype=int)
            for condn_on_smpl in condn_on_samples:
                sampler = rv_discrete(name=var_nm,values=(v_range,dist[tuple(condn_on_smpl)]))
                sample_vals = np.append(sample_vals,sampler.rvs(size=1))
        else:
            sampler = rv_discrete(name=var_nm,values=(v_range,dist))
            sample_vals = sampler.rvs(size=self.num_samples)
        self.samples[var_nm] = sample_vals
        self.df[var_nm] = sample_vals

    def store_samples(self,outfile):
        self.df.to_csv(outfile, index=False)

    def get_frame(self):
        return self.df
