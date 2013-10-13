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
#     ensemble.parallel=True
    
    pop_size = 8
    nr_of_generations = 100
    eps = np.array([1e-3, 1e6])

#     stats_callback1, pop  = ensemble.perform_outcome_optimization(obj_function = obj_function_multi,
#                                                     algorithm=NSGA2,
#                                                     reporting_interval=100, 
#                                                     weights=(MAXIMIZE, MAXIMIZE),
#                                                     pop_size=pop_size,          
#                                                     nr_of_generations=nr_of_generations,
#                                                     crossover_rate=0.8,
#                                                     mutation_rate=0.05)
#         
    stats_callback2, pop  = ensemble.perform_outcome_optimization(obj_function = obj_function_multi,
                                                    algorithm=epsNSGA2,
                                                    reporting_interval=100, 
                                                    weights=(MAXIMIZE, MAXIMIZE),
                                                    pop_size=pop_size,          
                                                    nr_of_generations=nr_of_generations,
                                                    crossover_rate=0.8,
                                                    mutation_rate=0.05,
                                                    eps=eps)
    del ensemble
    
    stats_callback1 = stats_callback2
    
    res1 = stats_callback1.algorithm.archive.keys
    x1 = [entry.values[0] for entry in res1]
    y1 = [entry.values[1] for entry in res1]

    res2 = stats_callback2.algorithm.archive.keys
    x2 = [entry.values[0] for entry in res2]
    y2 = [entry.values[1] for entry in res2]

    min_x = min(np.min(x2), np.min(x1))
    max_x = max(np.max(x2), np.max(x1))
    min_y = min(np.min(y2), np.min(y1))
    max_y = max(np.max(y2), np.max(y1))

    min_y = np.floor(min_y/eps[1])*eps[1]-eps[1]
    max_y = np.floor(max_y/eps[1])*eps[1]+2*eps[1]

    min_x = np.floor(min_x/eps[0])*eps[0]-eps[0]
    max_x = np.floor(max_x/eps[0])*eps[0]+2*eps[0]

    def make_gridded_ax(min_x,max_x,min_y,max_y):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        nr_xsteps = int(np.floor((max_x-min_x)/eps[0]))
        ax.xaxis.set_ticks([min_x+(eps[0]*i) for i in range(nr_xsteps+3)], minor=True)
        
        nr_ysteps = int(np.floor((max_y-min_y)/eps[1]))
        ax.yaxis.set_ticks([min_y+(eps[1]*i) for i in range(nr_ysteps+3)], minor=True)
    
        ax.xaxis.grid(which='minor')
        ax.yaxis.grid(which='minor') 
        
        ax.set_xlim(xmin=min_x, xmax=max_x)
        ax.set_ylim(ymin=min_y, ymax=max_y)
        return fig, ax
    
    
    fig, ax = make_gridded_ax(min_x, max_x, min_y, max_y)

    ax.scatter(x1,y1, c='r', label='NSGA2')
    ax.scatter(x2,y2, c='b', label='$\epsilon$-NSGA2')
    
    ax.set_ylabel("deceased population")
    ax.set_xlabel("infected fraction")

    ax.legend(loc='best')

    change = stats_callback2.change
    added = [entry[0] for entry in change]
    removed = [entry[1] for entry in change]

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot(added, label='added')
#     ax.plot(removed, label='removed')
#     ax.set_ylabel("changes")
#     ax.set_xlabel("generation")
#     ax.legend(loc='best')
#      
#     e_progress = [entry[2] for entry in change] 
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot(e_progress, label='$\epsilon$ progress')   
#     ax.set_ylabel('$\epsilon$ progress')
#     ax.set_xlabel("generation")
    
    plt.show()

if __name__ == '__main__':
    test_optimization()