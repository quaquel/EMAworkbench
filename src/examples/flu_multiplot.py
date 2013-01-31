'''
Created on 20 sep. 2011

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
import numpy as np
import matplotlib.pyplot as plt

from analysis.pairs_plotting import pairs_lines, pairs_scatter, pairs_density
from expWorkbench.util import load_results
from expWorkbench import ema_logging

ema_logging.log_to_stderr(level=ema_logging.DEFAULT_LEVEL)

#load the data
experiments, outcomes = load_results(r'.\data\100 flu cases no policy.bz2')

#transform the results to the required format
newResults = {}

#get time and remove it from the dict
time = outcomes.pop('TIME')

for key, value in outcomes.items():
    if key == 'deceased population region 1':
        newResults[key] = value[:,-1] #we want the end value
    else:
        # we want the maximum value of the peak
        newResults['max peak'] = np.max(value, axis=1) 
        
        # we want the time at which the maximum occurred
        # the code here is a bit obscure, I don't know why the transpose 
        # of value is needed. This however does produce the appropriate results
        logicalIndex = value.T==np.max(value, axis=1)
        newResults['time of max'] = time[logicalIndex.T]
        
pairs_scatter((experiments, newResults))
pairs_lines((experiments, newResults))
pairs_density((experiments, newResults))
plt.show() 