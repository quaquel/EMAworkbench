'''
Created on Feb 28, 2012

@author: jhkwakkel
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from expWorkbench import ema_logging
from test.test_vensim_flu import FluModel
from expWorkbench import ModelEnsemble, MAXIMIZE
from expWorkbench.ema_optimization import epsNSGA2, NSGA2
from expWorkbench.util import save_optimization_results


def obj_function_single(results):
    outcome = results['infected fraction R1']
    return np.max(outcome)

def obj_function_multi(results):
    outcome_1 = results['infected fraction R1']
    outcome_2 = results['deceased population region 1']
    return np.max(outcome_1), outcome_2[-1]

def test_optimization():
    ema_logging.log_to_stderr(ema_logging.INFO)
    
    model = FluModel(r'..\data', "fluCase")
    ensemble = ModelEnsemble()
    
    ensemble.set_model_structure(model)
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
    fn = '../data/test optimization save.bz2'
    save_optimization_results((stats,pop), fn)
    

if __name__ == '__main__':
    test_optimization()