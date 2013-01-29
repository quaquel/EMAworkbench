'''
Created on Mar 1, 2012

@author: LocalAdmin
'''
import matplotlib.pyplot as plt
import numpy as np

from expWorkbench import load_results
from analysis.prim import perform_prim, write_prim_to_stdout
from analysis.prim import show_boxes_individually

def classify(data):
    
    result = data['total fraction new technologies']    
    classes =  np.zeros(result.shape[0])
    classes[result[:, -1] > 0.8] = 1
    
    return classes

if __name__ == '__main__':

    results = load_results(r'CESUN_optimized_1000_new.cPickle')
    experiments, results = results
    logicalIndex = experiments['policy'] == 'Optimized Adaptive Policy'
    newExperiments = experiments[ logicalIndex ]
    newResults = {}
    for key, value in results.items():
        newResults[key] = value[logicalIndex]
    results = (newExperiments, newResults)

    boxes = perform_prim(results,
                         'total fraction new technologies', 
                         threshold=0.6, 
                         threshold_type=-1)
    
    write_prim_to_stdout(boxes)
    show_boxes_individually(boxes, results)
    plt.show()  
