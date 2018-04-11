'''
An example of the lake problem using the ema workbench. 

The model itself is adapted from the Rhodium example by Dave Hadka,
see https://gist.github.com/dhadka/a8d7095c98130d8f73bc

'''
from __future__ import (unicode_literals, print_function, absolute_import,
                        division)

import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from SALib.analyze import sobol


from ema_workbench import (Model, RealParameter, ScalarOutcome, Constant,
                           ema_logging, MultiprocessingEvaluator, Policy)
from ema_workbench.em_framework.evaluators import SOBOL
from ema_workbench.em_framework import get_SALib_problem

def lake_problem(
         b = 0.42,          # decay rate for P in lake (0.42 = irreversible)
         q = 2.0,           # recycling exponent
         mean = 0.02,       # mean of natural inflows
         stdev = 0.001,     # future utility discount rate
         delta = 0.98,      # standard deviation of natural inflows
         alpha = 0.4,       # utility from pollution
         nsamples = 100,    # Monte Carlo sampling of natural inflows
         **kwargs):   
    try:
        decisions = [kwargs[str(i)] for i in range(100)]
    except KeyError:
        decisions = [0, ] * 100
    
    Pcrit = brentq(lambda x: x**q/(1+x**q) - b*x, 0.01, 1.5)
    nvars = len(decisions)
    X = np.zeros((nvars,))
    average_daily_P = np.zeros((nvars,))
    decisions = np.array(decisions)
    reliability = 0.0

    for _ in range(nsamples):
        X[0] = 0.0
        
        natural_inflows = np.random.lognormal(
                math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
                math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
                size = nvars)
        
        for t in range(1,nvars):
            X[t] = (1-b)*X[t-1] + X[t-1]**q/(1+X[t-1]**q) + decisions[t-1] + natural_inflows[t-1]
            average_daily_P[t] += X[t]/float(nsamples)
    
        reliability += np.sum(X < Pcrit)/float(nsamples*nvars)
      
    max_P = np.max(average_daily_P)
    utility = np.sum(alpha*decisions*np.power(delta,np.arange(nvars)))
    inertia = np.sum(np.absolute(np.diff(decisions)) < 0.02)/float(nvars-1)

    return max_P, utility, inertia, reliability


def analyze(results, ooi):
    '''analyze results using SALib sobol, returns a dataframe
    
    '''
    
    _, outcomes = results
    
    problem = get_SALib_problem(lake_model.uncertainties)
    y = outcomes[ooi]
    sobol_indices = sobol.analyze(problem, y)
    sobol_stats = {key:sobol_indices[key] for key in ['ST', 'ST_conf', 'S1', 
                                                      'S1_conf']}
    sobol_stats = pd.DataFrame(sobol_stats, index=problem['names'])
    sobol_stats.sort_values(by='ST', ascending=False)
    s2 = pd.DataFrame(sobol_indices['S2'], index=problem['names'], 
                      columns=problem['names'])
    s2_conf = pd.DataFrame(sobol_indices['S2_conf'], index=problem['names'], 
                           columns=problem['names'])
    
    return sobol_stats, s2, s2_conf

if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)

    # instantiate the model
    lake_model = Model('lakeproblem', function=lake_problem)
    lake_model.time_horizon = 100
    
    #specify uncertainties
    lake_model.uncertainties = [RealParameter('b', 0.1, 0.45),
                                RealParameter('q', 2.0, 4.5),
                                RealParameter('mean', 0.01, 0.05),
                                RealParameter('stdev', 0.001, 0.005),
                                RealParameter('delta', 0.93, 0.99)]
    
    # set levers, one for each time step
    lake_model.levers = [RealParameter(str(i), 0, 0.1) for i in 
                         range(lake_model.time_horizon)]
    
    #specify outcomes 
    lake_model.outcomes = [ScalarOutcome('max_P',),
                           ScalarOutcome('utility'),
                           ScalarOutcome('inertia'),
                           ScalarOutcome('reliability')]
    
    # override some of the defaults of the model
    lake_model.constants = [Constant('alpha', 0.41),
                            Constant('nsamples', 150)]
        
    
    # generate sa single default no release policy
    policy = Policy('no release', **{str(i):0.1 for i in range(100)})
    
    n_scenarios = 1000
     
    with MultiprocessingEvaluator(lake_model) as evaluator:
        results = evaluator.perform_experiments(n_scenarios, policy, 
                                                uncertainty_sampling=SOBOL)
        
    sobol_stats, s2, s2_conf = analyze(results, 'max_P')
    print(sobol_stats)
    print(s2)
    print(s2_conf)
