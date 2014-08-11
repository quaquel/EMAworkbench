'''
Created on 22 nov. 2012

@author: localadmin
'''
from expWorkbench import util, ema_logging
import numpy as np
import os
from test.util import load_flu_data
from expWorkbench.util import save_results, load_results



def test_save_results():
    # test for 1d
    # test for 2d
    # test for 3d
    # test for very large
    
    nr_experiments = 10000
    experiments = np.recarray((nr_experiments,),
                           dtype=[('x', float), ('y', float)])
    outcome_a = np.random.rand(nr_experiments,1)
    
    results = (experiments, {'a': outcome_a})
    
    save_results(results, r'../data/test.tar.gz')
    os.remove('../data/test.tar.gz')
    ema_logging.info('1d saved successfully')
    
    nr_experiments = 10000
    nr_timesteps = 100
    experiments = np.recarray((nr_experiments,),
                           dtype=[('x', float), ('y', float)])
    outcome_a = np.random.rand(nr_experiments,nr_timesteps)
    
    results = (experiments, {'a': outcome_a})
    save_results(results, r'../data/test.tar.gz')
    os.remove('../data/test.tar.gz')
    ema_logging.info('2d saved successfully')
 
 
    nr_experiments = 10000
    nr_timesteps = 100
    nr_replications = 10
    experiments = np.recarray((nr_experiments,),
                           dtype=[('x', float), ('y', float)])
    outcome_a = np.random.rand(nr_experiments,nr_timesteps,nr_replications)
     
    results = (experiments, {'a': outcome_a})
    save_results(results, r'../data/test.tar.gz')
    os.remove('../data/test.tar.gz')
    ema_logging.info('3d saved successfully')
    

def test_load_results():
    # test for 1d
    # test for 2d
    # test for 3d
    # test for nd

    nr_experiments = 10000
    experiments = np.recarray((nr_experiments,),
                           dtype=[('x', float), ('y', float)])
    outcome_a = np.random.rand(nr_experiments,1)
    
    results = (experiments, {'a': outcome_a})
    
    save_results(results, r'../data/test.tar.gz')
    experiments, outcomes  = load_results(r'../data/test.tar.gz')
    
    logical = np.allclose(outcomes['a'],outcome_a)
    
    os.remove('../data/test.tar.gz')
    
    if logical:
        ema_logging.info('1d loaded successfully')
    
    
    
    nr_experiments = 1000
    nr_timesteps = 100
    nr_replications = 10
    experiments = np.recarray((nr_experiments,),
                           dtype=[('x', float), ('y', float)])
    outcome_a = np.random.rand(nr_experiments,nr_timesteps,nr_replications)
     
    results = (experiments, {'a': outcome_a})
    save_results(results, r'../data/test.tar.gz')
    experiments, outcomes = load_results(r'../data/test.tar.gz')
    
    logical = np.allclose(outcomes['a'],outcome_a)
    
    os.remove('../data/test.tar.gz')
    
    if logical:
        ema_logging.info('3d loaded successfully')
    

if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)
   
#     test_save_results()
    test_load_results()