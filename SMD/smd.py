# -*- coding: utf-8 -*-
"""Model

Provides a class for consumption-saving models with methods for saving and loading
and interfacing with numba jitted functions and C++.

"""
import numpy as np
from scipy import linalg
from scipy.optimize import minimize

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

        self.info = SimpleNamespace() 

        self.datamoms = OrderedDict()
        self.moms = OrderedDict()


    def assert_info(self,model,est_par,W):

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

        # check that moments in data is real numbers

        # check that weighting matrix is positive semidefinite
        


    def estimate(self,model,est_par,W=None,print_progress=False):
        ''' If no weighting matrix is given, equal weighting is used'''

        # update and check information given
        self.assert_info(model,est_par,W)

        # setup estimation

        # loop over multistarts

        # Callback function

        # loop over solvers
        for i,(solver,max_iter) in enumerate(zip(self.options.solver,self.options.max_iter)):
            
            # optimization settings
            optim_options = {'maxiter':max_iter,'disp':self.options.disp}
            initial = [spec[1]['init'] for spec in est_par.items()] if i==0 else res.x
            args = (model,est_par,W,print_progress)

            # optimize
            res = minimize(self.obj_func,initial,method=solver,options=optim_options,args=args)
            
        return res

    def load_data(self,path,N=None):

        # Check number of observations
        if N is None:
            try: N = np.load_txt(path+'N') # try to load txt in path
            except: raise ValueError('Number of observations, N, must be specified in .load_data()')



    def obj_func(self,theta,model,est_par,W,print_progress=False):

        global nfuns
        global minfunval

        # a. setup print progress
        if print_progress:
            if nfuns == 0:
                minfunval = np.inf
            nfuns += 1
        
        # penalty
        _,penalty = bounds_penalty(theta,est_par)

        # calculate difference between the data and simulated moments
        if W is None:
            objval = np.prod(self.obj_func_vec(theta,model,est_par))
        else:
            diff = np.expand_dims( self.obj_func_vec(theta,model,est_par) ,axis=1)
            objval = np.squeeze(diff.T @ W @ diff) + penalty

        objval += penalty

        # d. print progress
        if print_progress:
            minfunval = np.fmin(objval,minfunval)

        # return objective function
        return objval

    def obj_func_vec(self,theta,model,est_par):

        # bounds and penalty
        theta_clipped,_ = bounds_penalty(theta,est_par)

        # update parameters (with clipped)
        names = [name[0] for name in est_par.items()]
        model.update_par(theta_clipped,names)

        # solve model
        model.solve()

        # simulate data and calculate moments
        model.simulate()
        moms = model.calc_moments()

        # allocate memory
        diff_vec = np.nan + np.zeros(len(moms))

        # loop through used moments
        for i_mom,(key,mom_sim) in enumerate(moms.items()):
            diff_vec[i_mom] = self.datamoms[key] - mom_sim

        return diff_vec    

    
    ##########
    ## print #
    ##########

    def __str__(self):
        """ called when SMD is printed """ 
        
        print(self.info)
        print(self.options)   

        return 'test'

def bounds_penalty(theta,est_par):

        lower = [spec[1]['bounds'][0] for spec in est_par.items()]
        upper = [spec[1]['bounds'][1] for spec in est_par.items()]

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