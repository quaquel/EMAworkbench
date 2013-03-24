'''
Created on 18 Oct 2012

@author: jhkwakkel
'''
import numpy as np
import matplotlib.pyplot as plt

from expWorkbench import load_results, ema_logging
from analysis import prim
from analysis import pca_prim

ema_logging.log_to_stderr(ema_logging.INFO)

def classify(data):
    #get the output for deceased population
    result = data['deceased population region 1']
    
    #make an empty array of length equal to number of cases 
    classes =  np.zeros(result.shape[0])
    
    #if deceased population is higher then 1.000.000 people, classify as 1 
    classes[result[:, -1] > 1500000] = 1
    
    return classes

results = load_results(r".\data\1000 flu cases no policy.cPickle")

#perform prim on modified results tuple
res = pca_prim.perform_pca_prim(results, 
                                classify,
                                mass_min=0.075, 
                                threshold=0.8, 
                                threshold_type=1)

rotation_matrix, row_names, column_names, rotated_experiments, boxes = res

#visualize results
prim.write_prim_to_stdout(boxes)

# we need to use the rotated experiments now
results = (rotated_experiments, results[1])

prim.show_boxes_together(boxes, results)

plt.show()
