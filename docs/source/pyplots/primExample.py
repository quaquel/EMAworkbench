import numpy as np
import matplotlib.pyplot as plt

import analysis.prim as prim
from expWorkbench import load_results
import expWorkbench.ema_logging as ema_logging
 
#perform_prim logs information to the logger
ema_logging.log_to_stderr(level=ema_logging.INFO)

def classify(data):
    #get the output for deceased population
    result = data['deceased population region 1']
    
    #make an empty array of length equal to number of cases 
    classes =  np.zeros(result.shape[0])
    
    #if deceased population is higher then 1.000.000 people, 
    #classify as 1 
    classes[result[:, -1] > 1000000] = 1
    
    return classes

#load data
results = load_results(r'../../../src/analysis/1000 flu cases.cPickle', zipped=False)
experiments, results = results

#extract results for 1 policy
logicalIndex = experiments['policy'] == 'no policy'
newExperiments = experiments[ logicalIndex ]
newResults = {}
for key, value in results.items():
    newResults[key] = value[logicalIndex]

results = (newExperiments, newResults)

#perform prim on modified results tuple
prims, uncertainties, x = prim.perform_prim(results, classify, 
                                            threshold=0.8, 
                                            threshold_type=1)

#visualize

figure = prim.visualize_prim(prims, 
                             uncertainties, 
                             x, 
                             filter=True)
plt.show()