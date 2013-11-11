import numpy as np
import matplotlib.pyplot as plt

import analysis.prim as prim
from expWorkbench import load_results, ema_logging

ema_logging.log_to_stderr(level=ema_logging.INFO)

def classify(data):
    #get the output for deceased population
    result = data['deceased population region 1']
    
    #make an empty array of length equal to number of cases 
    classes =  np.zeros(result.shape[0])
    
    #if deceased population is higher then 1.000.000 people, classify as 1 
    classes[result[:, -1] > 1000000] = 1
    
    return classes

#load data
results = load_results(r'./data/1000 flu cases.bz2')
experiments, results = results

#extract results for 1 policy
logical = experiments['policy'] == 'no policy'
new_experiments = experiments[ logical ]
new_results = {}
for key, value in results.items():
    new_results[key] = value[logical]

results = (new_experiments, new_results)

#perform prim on modified results tuple

prim = prim.Prim(results, classify, threshold=0.8, threshold_type=1)
box_1 = prim.find_box()
#box_1.show_ppt()
box_1.show_tradeoff()
box_1.write_ppt_to_stdout()

#print prim to std_out
prim.write_boxes_to_stdout()


##visualize
#prim.show_boxes()
#prim.show_boxes(together=False)
plt.show()