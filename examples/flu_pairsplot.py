'''
Created on 20 sep. 2011

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
import numpy as np
import matplotlib.pyplot as plt

from ema_workbench import load_results, ema_logging

from ema_workbench.analysis.pairs_plotting import (pairs_lines, pairs_scatter, 
                                                   pairs_density)

ema_logging.log_to_stderr(level=ema_logging.DEFAULT_LEVEL)

# load the data
fh = './data/1000 flu cases no policy.tar.gz'
experiments, outcomes = load_results(fh)

# transform the results to the required format
# that is, we want to know the max peak and the casualties at the end of the 
# run
tr = {}

# get time and remove it from the dict
time = outcomes.pop('TIME')

for key, value in outcomes.items():
    if key == 'deceased population region 1':
        tr[key] = value[:,-1] #we want the end value
    else:
        # we want the maximum value of the peak
        max_peak = np.max(value, axis=1) 
        tr['max peak'] = max_peak
        
        # we want the time at which the maximum occurred
        # the code here is a bit obscure, I don't know why the transpose 
        # of value is needed. This however does produce the appropriate results
        logical = value.T==np.max(value, axis=1)
        tr['time of max'] = time[logical.T]
        
pairs_scatter((experiments, tr), filter_scalar=False)
pairs_lines((experiments, outcomes))
pairs_density((experiments, tr), filter_scalar=False)
plt.show() 