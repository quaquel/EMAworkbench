'''
Created on Mar 1, 2012

@author: LocalAdmin
'''
import matplotlib.pyplot as plt
import numpy as np

from expWorkbench import load_results
from analysis.prim import perform_prim, write_prim_to_stdout
from analysis.prim import show_boxes_individually, show_boxes_together

def classify(data):
    
    result = data['total fraction new technologies']    
    classes =  np.zeros(result.shape[0])
    classes[result[:, -1] < 0.6] = 1
    
    return classes

if __name__ == '__main__':

    results = load_results(r'CESUN_optimized_1000.cPickle')
    boxes = perform_prim(results,classify, threshold=0.6, threshold_type=1)
    
    write_prim_to_stdout(boxes)
    show_boxes_individually(boxes, results)
    plt.show()  
