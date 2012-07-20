import numpy as np
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec 

import analysis.prim as prim
from expWorkbench import load_results
import expWorkbench.EMAlogging as EMAlogging
from analysis.primCode.primDataTypeAware import in_box
 
#perform_prim logs information to the logger
EMAlogging.log_to_stderr(level=EMAlogging.INFO)

def classify(data):
    #get the output for deceased population
    result = data['deceased population region 1']
    
    #make an empty array of length equal to number of cases 
    classes =  np.zeros(result.shape[0])
    
    #if deceased population is higher then 1.000.000 people, classify as 1 
    classes[result[:, -1] > 1000000] = 1
    
    return classes

#load data
results = load_results(r'../analysis/1000 flu cases.cPickle')
experiments, results = results

#extract results for 1 policy
logicalIndex = experiments['policy'] == 'no policy'
newExperiments = experiments[ logicalIndex ]
newResults = {}
for key, value in results.items():
    newResults[key] = value[logicalIndex]

results = (newExperiments, newResults)

#perform prim on modified results tuple
boxes = prim.perform_prim(results, classify, 
                                    threshold=0.8, 
                                    threshold_type=1,
                                    pasting=True)



def plot_prim_time_series(boxes, data, outcome):
    x, results = data
    outcomes = results[outcome]
    
    try:
        time =  results.get('TIME')[0, :]
    except KeyError:
        time =  np.arange(0, outcomes.shape[1])
    
    #the plotting
    grid = gridspec.GridSpec(len(boxes)-1, 1)
    grid.update(wspace = 0.05,
                hspace = 0.4)
    
    figure = plt.figure()
    
    for i, box in enumerate(boxes[0:-1]):
        logical = in_box(x, box.box)
        
        #make the axes
        ax = figure.add_subplot(grid[i, 0])
        
        value = outcomes[logical]
        
        ax.plot(time.T[:, np.newaxis], value.T)
    return figure


fig = plot_prim_time_series(boxes, results, outcome='infected fraction R1')
plt.show()