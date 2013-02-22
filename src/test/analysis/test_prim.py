'''
Created on Mar 13, 2012

@author: jhkwakkel
'''
from __future__ import division

import numpy as np

from expWorkbench import ema_logging, vensim, load_results
from analysis import new_prim

def classify(data):
    #get the output for deceased population
    result = data['deceased population region 1']
    
    #make an empty array of length equal to number of cases 
    classes =  np.zeros(result.shape[0])
    
    #if deceased population is higher then 1.000.000 people, classify as 1 
    classes[result[:, -1] > 1000000] = 1
    
    return classes

def test_prim(results):

#    
#    #perform prim on modified results tuple
#    boxes = prim.perform_prim(results, classify, 
#                                        threshold=0.8, 
#                                        threshold_type=1,
#                                        pasting=True)
#    return boxes

    pass

def test_write_to_std_out(results):
#    boxes = test_prim(results)
#    prim.write_prim_to_stdout(boxes, results[0])
    pass

def test_prim_initialization(results):
    experiments, outcomes = results
    
    outcomes['death toll'] = outcomes['deceased population region 1'][:, -1]
    results = experiments, outcomes
    
    # string classify on 1D matrix
    new_prim.Prim(results, 'death toll')
    
    # string classify on 2D matrix should raise exception
    
    # classify function returning 1D matrix
    
    # classify function returning 2D matrix should raise exception

if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)
    results = load_results(r'../data/1000 flu cases no policy.bz2')
    test_write_to_std_out(results)