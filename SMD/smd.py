# -*- coding: utf-8 -*-
"""Model

Provides a class for consumption-saving models with methods for saving and loading
and interfacing with numba jitted functions and C++.

"""
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import minimize

import time
import glob

from types import SimpleNamespace
from collections import OrderedDict


# main
class SMDClass():
    
    def __init__(self,**kwargs):
        """ defines default attributes """

        # a. options
        self.options = SimpleNamespace() 
        self.options.solver = ['nelder-mead','bfgs'] # if a list is given the sequence of solvers will be applied 
        self.options.max_iter = [20,100] # the list of max_iter's must match the list of solvers
        self.options.disp = True

        self.datamoms = None #OrderedDict()
        self.moms = OrderedDict()

        self.datamoms_var = {} 
        self.datamoms_cov = {} 
        self.datamoms_scale = {}

        self.cov_moms = None

        self.info = SimpleNamespace() 
        self.info.momspecs = {} 
        self.info.names = []
        self.info.Ndata = None
        self.info.datafolder = None
        self.info.do_scale = False 

        self.est = {}


    def set_info(self,model,est_par,W):

        options = self.options
        info = self.info

        # number of solvers
        options.solver = options.solver if isinstance(options.solver, (list,tuple)) else [options.solver]
        options.max_iter = options.max_iter if isinstance(options.max_iter, (list,tuple)) else [options.max_iter]
        assert len(options.solver) == len(options.max_iter), f'The sequence of numerical solvers is {len(options.solver)} while the list of max_iter is only of length {len(self.options.max_iter)}'

        # check that model has the required modules
        for method in ('update_par','solve','simulate','calc_moments'):
            assert hasattr(model,method), f'model is missing required .{method}() method'

        # check the starting values are within bounds?

        # check that weighting matrix is positive semidefinite

        # save information
        info.W = W
        info.est_par = est_par
        info.names = [name for name,spec in est_par.items()]
        info.lower_bounds = [spec['bounds'][0] for name,spec in est_par.items()]
        info.upper_bounds = [spec['bounds'][1] for name,spec in est_par.items()]


    def estimate(self,model,est_par,W=None,print_progress=False):
        ''' If no weighting matrix is given, equal weighting is used'''
        global nits

        # update and check information given
        self.set_info(model,est_par,W)

        # setup estimation ? 

        if print_progress:
            print('Running optimization:')
            t0 = time.time()   

        # loop over multistarts

        # Callback function
        nits = 0
        callback_func = lambda theta: progress(theta,print_progress,self.info.names)
        callback_func([spec['init'] for name,spec in est_par.items()])     

        # loop over solvers
        for i,(solver,max_iter) in enumerate(zip(self.options.solver,self.options.max_iter)):

            if print_progress & (len(self.options.solver)>1): 
                print(f'\n   Number {i+1} in solver sequence of length {len(self.options.solver)}\n')

            # optimization settings
            optim_options = {'maxiter':max_iter,'disp':self.options.disp}
            initial = [spec['init'] for name,spec in est_par.items()] if i==0 else res.x
            args = (model,W,print_progress)

            # optimize
            res = minimize(self.obj_func,initial,method=solver,callback=callback_func,options=optim_options,args=args)
            
        return res

    def load_data(self,momspecs,datafolder='data',Ndata=None,do_scale=False):

        # TODO: add load-option that loads covariance

        # Check number of observations
        if Ndata is None:
            try: Ndata = np.genfromtxt(f'{datafolder}/N.txt',delimiter=',').reshape((1,))[0] # try to load txt in path
            except: raise ValueError('Number of observations, Ndata, must be specified in .load_data() or in .txt file in datafolder')

        self.info.datafolder = datafolder
        self.info.momspecs = momspecs
        self.info.Ndata = Ndata
        self.info.do_scale = do_scale

        # a. load txt-files
        datamoms_df = pd.read_excel(f'{datafolder}/moments.xls')

        # boostrap
        files = glob.glob(f'{datafolder}/moments_bootstrap*')
        moms_boot = np.concatenate([np.genfromtxt(file,delimiter=',') for file in files],axis=0).T

        # b. find chosen moments
        for name,infos in self.info.momspecs.items():

            for info in infos:
                
                if len(info) == 1:
                    spec = 0
                    ages = info[0]
                    key = (name,)
                else:
                    spec = info[0]
                    ages = info[1]
                    key = (name,info[0])
                    
                J = datamoms_df.momname == name
                J &= (datamoms_df.spec == spec)
                J &= (datamoms_df[J].age.isin(ages))
                
                self.datamoms_scale[key] = 1.0/np.fabs(np.nanmean(datamoms_df.loc[J,'value'].values))

                self.datamoms[key] = datamoms_df.loc[J,'value'].values
                self.datamoms_var[key] = np.var(moms_boot[J,:],axis=1)  
                # self.datamoms_w[key] = 1/(self.datamoms_var[key]*self.par.Ndata)

                for name_cov,infos_cov in self.info.momspecs.items():

                    for info_cov in infos_cov:
                        
                        if len(info_cov) == 1:
                            spec_cov = 0
                            ages_cov = info_cov[0]
                            key_cov = (name_cov,)
                        else:
                            spec_cov = info_cov[0]
                            ages_cov = info_cov[1]
                            key_cov = (name_cov,info_cov[0])
                            
                        J_cov = datamoms_df.momname == name_cov
                        J_cov &= (datamoms_df.spec == spec_cov)
                        J_cov &= (datamoms_df[J_cov].age.isin(ages_cov))
                        
                        # Scale moments to improve numerical behavior of the covariance matrix
                        if self.info.do_scale:
                            scale = self.datamoms_scale[key]
                            scale_cov = 1.0/np.fabs(np.nanmean(datamoms_df.loc[J_cov,'value'].values))
                        else: 
                            scale = scale_cov = 1.0

                        a = moms_boot[J,:]*scale
                        b = moms_boot[J_cov,:]*scale_cov
                        covmat = np.zeros((a.shape[0],b.shape[0]))
                        for i in range(a.shape[0]):
                            for j in range(b.shape[0]):
                                if key == key_cov and i == j:
                                    covmat[i,j] = np.var(a[i,:])
                                else:
                                    covmat[i,j] = np.cov(a[i,:],b[j,:])[0,1]

                        self.datamoms_cov[(key,key_cov)] = covmat
                
    def calc_cov_moms(self):

        def _increment(values):

            if type(values) == np.ndarray:
                I = ~np.isnan(values)              
                return I.sum()
            else:
                if ~np.isnan(values):        
                    return 1

        # a. number of moments
        N = 0
        for key,values in self.moms.items():
            N += _increment(values)

        # b. selected covariances
        self.cov_moms = np.zeros((N,N))

        i = 0
        for key,values in self.moms.items():
            
            iprev = i
            i += _increment(values)
            
            j = 0
            for key_cov,values_cov in self.moms.items():
            
                jprev = j
                j += _increment(values_cov)

                covmat = self.datamoms_cov[(key,key_cov)]
                covmat = covmat[~np.isnan(covmat)]
                self.cov_moms[iprev:i,jprev:j] = covmat.reshape((i-iprev,j-jprev))

        # c. compute Omega and W
        # i. construct indicator for close-to-zero correlations
        # corrmat = np.zeros((len(self.cov_moms),len(self.cov_moms)))
        # for i in range(len(self.cov_moms)):
        #     for j in range(len(self.cov_moms)):
        #         corrmat[i,j] = self.cov_moms[i,j]/np.sqrt(self.cov_moms[i,i]*self.cov_moms[j,j])

        # I = (corrmat<-0.05) | (corrmat>0.05)
        
        # ii. calculate covariance and weighting matrix NO NOT NOW: where close-to-zero correlations are imposed to improve numerical behavior
        self.cov_moms = self.info.Ndata*self.cov_moms

#        self.par.W = np.diag(1/np.diag(self.par.Omega))



    def obj_func(self,theta,model,W,print_progress=False):

        global nfuns
        global minfunval

        # a. setup print progress
        if print_progress:
            if nfuns == 0:
                minfunval = np.inf
            nfuns += 1
        
        # penalty
        _,penalty = self.bounds_penalty(theta)

        # calculate difference between the data and simulated moments
        objval = self.obj_func_vec(theta,model)
        if W is None:
            objval = np.sum(objval*objval)
        else:
            diff = np.expand_dims( objval ,axis=1)
            objval = np.squeeze(diff.T @ W @ diff) + penalty

        objval += penalty

        # d. print progress
        if print_progress:
            minfunval = np.fmin(objval,minfunval)

        # return objective function
        return objval

    def obj_func_vec(self,theta,model):

        # bounds and penalty
        theta_clipped,_ = self.bounds_penalty(theta)

        # update parameters (with clipped) and solve model
        model.update_par(theta_clipped,self.info.names)
        model.solve()

        # simulate data and calculate moments
        model.simulate()
        self.moms = model.calc_moments()

        return self.diff_moms_vec(self.moms,self.datamoms,self.datamoms_scale)
    
    def diff_moms_vec(self,moms,datamoms,scale_moms):

        diff_moms_vec = np.array([])
        for key,values in moms.items():

            scale = scale_moms[key] if self.info.do_scale else 1.0  
            data_values = datamoms[key]

            diff_moms = (data_values - values)*scale # this is done here to preserve the original scaling in the dictionaries

            # handle NaNs
            Idata = ~np.isnan(data_values)  
            Inan = np.isnan(values) & Idata
            if type(diff_moms) == np.ndarray:
                diff_moms[Inan] = 1000.0 * data_values[Inan]       # penalty for producing NaNs in model when not in data
                diff_moms_vec = np.insert(diff_moms_vec,len(diff_moms_vec),diff_moms[Idata]) 
            
            else:
                if Idata:
                    if Inan:
                        diff_moms = 1000.0 * data_values       # penalty for producing NaNs in model when not in data
                    diff_moms_vec = np.insert(diff_moms_vec,len(diff_moms_vec),diff_moms)  

        return diff_moms_vec

    def bounds_penalty(self,theta):

        lower = self.info.lower_bounds
        upper = self.info.upper_bounds

        # bounds and penalty
        penalty = 0
        theta_clipped = theta.copy()
        for i in range(theta.size):
            
            # i. clip
            if (lower[i] != None) or (upper[i] != None):
                theta_clipped[i] = np.clip(theta_clipped[i],lower[i],upper[i])
            
            # ii. penalty
            penalty += 10_000*(theta[i]-theta_clipped[i])**2

        return theta_clipped,penalty

    
    ##########
    ## print #
    ##########

    def __str__(self):
        """ called when SMD is printed """ 
        
        print(self.info)
        print(self.options)   

        return 'test'

def progress(theta,print_progress,names):
    """ print progress when estimating """

    global nits
    global nfuns
    global minfunval
    global tic

    if print_progress:

        if nits > 0 and nits%10 == 0:
            print(f'    obj = {minfunval:.4f}, {time.time()-tic:.1f} secs, {nfuns} func. evals')

        if nits%10 == 0:
            print(f'{nits:4d}:')
            for x,name in zip(theta,names):
                print(f'     {name} = {x:.4f}')

    # d. update
    nits += 1
    nfuns = 0
    tic = time.time()