import numpy as np
import matplotlib.pyplot as plt

from analysis.pairs_plotting import pairs_scatter, pairs_density, pairs_lines
from expWorkbench.util import load_results

#load the data
experiments, results = load_results(r'../../../src/analysis/1000 flu cases.cPickle', zipped=False)

#transform the results to the required format
newResults = {}

#get time and remove it from the dict
time = results.pop('TIME')

for key, value in results.items():
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

pairs_density((experiments, newResults))
plt.show() 