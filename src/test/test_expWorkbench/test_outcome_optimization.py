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
    
    pop_size = 10
    nr_of_generations = 250

    stats_callback1, pop  = ensemble.perform_outcome_optimization(obj_function = obj_function_multi,
                                                    algorithm=NSGA2,
                                                    reporting_interval=100, 
                                                    weights=(MAXIMIZE, MAXIMIZE),
                                                    pop_size=pop_size,          
                                                    nr_of_generations=nr_of_generations,
                                                    crossover_rate=0.8,
                                                    mutation_rate=0.05)
        
    stats_callback2, pop  = ensemble.perform_outcome_optimization(obj_function = obj_function_multi,
                                                    algorithm=epsNSGA2,
                                                    reporting_interval=100, 
                                                    weights=(MAXIMIZE, MAXIMIZE),
                                                    pop_size=pop_size,          
                                                    nr_of_generations=nr_of_generations,
                                                    crossover_rate=0.8,
                                                    mutation_rate=0.05)
    del ensemble
    
    res1 = stats_callback1.algorithm.archive.keys
    x1 = [entry.values[0] for entry in res1]
    y1 = [entry.values[1] for entry in res1]

    res2 = stats_callback1.algorithm.archive.keys
    x2 = [entry.values[0] for entry in res2]
    y2 = [entry.values[1] for entry in res2]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1,y1, c='r')
    ax.scatter(x2,y2, c='b')
    ax.set_ylabel("deceased population")
    ax.set_xlabel("infected fraction")

    change = stats_callback2.change
    added = [entry[0] for entry in change]
    removed = [entry[1] for entry in change]


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(added, label='added')
    ax.plot(removed, label='removed')
    ax.set_ylabel("changes")
    ax.set_xlabel("generation")
    ax.legend(loc='best')
    
    e_progress = [entry[2] for entry in change] 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(e_progress, label='$\epsilon$ progress')   
    ax.set_ylabel('$\epsilon$ progress')
    ax.set_xlabel("generation")
    
    plt.show()

if __name__ == '__main__':
    test_optimization()