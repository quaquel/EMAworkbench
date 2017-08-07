'''

This file illustrated the use of the workbench for doing 
a PRIM analysis.

The data was generated using a system dynamics models implemented in Vensim.
See flu_example.py for the code. The dataset was generated using 32 bit Python.
Therefore, this example will not work if you are running 64 bit Python. 


.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
                chamarat <c.hamarat  (at) tudelft (dot) nl>

'''

import numpy as np
import matplotlib.pyplot as plt

from ema_workbench import ema_logging, load_results
import ema_workbench.analysis.prim as prim

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
fn = r'./data/1000 flu cases.tar.gz'
results = load_results(fn)
experiments, results = results

#extract results for 1 policy
logical = experiments['policy'] == 'no policy'
new_experiments = experiments[ logical ]
new_results = {}
for key, value in results.items():
    new_results[key] = value[logical]

results = (new_experiments, new_results)

#perform prim on modified results tuple

prim_obj = prim.setup_prim(results, classify, threshold=0.8, threshold_type=1)

box_1 = prim_obj.find_box()
box_1.show_ppt()
box_1.show_tradeoff()
box_1.inspect(5)
box_1.select(5)
box_1.write_ppt_to_stdout()
box_1.show_pairs_scatter()

#print prim to std_out
print prim_obj.stats_to_dataframe()
print prim_obj.boxes_to_dataframe()

#visualize
prim_obj.display_boxes()
plt.show()