'''
Created on Feb 28, 2012

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import division

import os

import numpy as np

from ..vensim_flu_for_testing import FluModel
from ema_workbench.em_framework import ModelEnsemble, MAXIMIZE
from ema_workbench.em_framework.ema_optimization import epsNSGA2
from ema_workbench.util import ema_logging


def obj_function_single(results):
    outcome = results['infected fraction R1']
    return np.max(outcome)

def obj_function_multi(results):
    outcome_1 = results['infected fraction R1']
    outcome_2 = results['deceased population region 1']
    return np.max(outcome_1), outcome_2[-1]

def test_optimization():
    if os.name != 'nt':
        return
    ema_logging.log_to_stderr(ema_logging.INFO)
    
    model = FluModel(r'../models', "fluCase")
    ensemble = ModelEnsemble()
    
    ensemble.model_structure = model
    ensemble.parallel=True
    
    pop_size = 8
    nr_of_generations = 10
    eps = np.array([1e-3, 1e6])

    stats, pop  = ensemble.perform_outcome_optimization(obj_function = obj_function_multi,
                                                    algorithm=epsNSGA2,
                                                    reporting_interval=100, 
                                                    weights=(MAXIMIZE, MAXIMIZE),
                                                    pop_size=pop_size,          
                                                    nr_of_generations=nr_of_generations,
                                                    crossover_rate=0.8,
                                                    mutation_rate=0.05,
                                                    eps=eps)
    

if __name__ == '__main__':
    if os.name == 'nt':
        test_optimization()