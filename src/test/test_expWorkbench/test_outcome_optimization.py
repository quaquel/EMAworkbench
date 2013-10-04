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
        
    stats_callback, pop  = ensemble.perform_outcome_optimization(obj_function = obj_function_multi,
                                                    reporting_interval=100, 
                                                    weights=(MAXIMIZE, MAXIMIZE),
                                                    pop_size=1000,          
                                                    nr_of_generations=100,
                                                    crossover_rate=0.8,
                                                    mutation_rate=0.05,
                                                    caching=False)
    
    res = stats_callback.hall_of_fame.keys

    x = [entry.values[0] for entry in res]
    y = [entry.values[1] for entry in res]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x,y)
    ax.set_ylabel("deceased population")
    ax.set_xlabel("infected fraction")

    change = stats_callback.change
    added = [entry[0] for entry in change]
    removed = [entry[1] for entry in change]
    e_progress = [entry[2] for entry in change]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(added, label='added')
    ax.plot(removed, label='removed')
    ax.set_ylabel("changes")
    ax.set_xlabel("generation")
    ax.legend(loc='best')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(e_progress, label='$\epsilon$ progress')   
    ax.set_ylabel('$\epsilon$ progress')
    ax.set_xlabel("generation")
    
    plt.show()

if __name__ == '__main__':
    test_optimization()