'''
Created on Sep 3, 2012

@author: jhkwakkel
'''
import numpy as np
import matplotlib.pyplot as plt

from expWorkbench import load_results, ema_logging
from analysis import  prim

def classify(data):
    
    result = data['total fraction new technologies']    
    classes =  np.zeros(result.shape[0])
    classes[result[:, -1] < 0.6] = 1
    
    return classes

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)
    
    results = load_results(r'prim data 100 cases.cPickle')
    boxes = prim.perform_prim(results, 
                      classify=classify,
                      mass_min=0.05, 
                      threshold=0.8)
    prim.show_boxes_together(boxes, results)
    plt.show()