'''
Created on May 26, 2015

@author: jhkwakkel
'''
import numpy as np
import matplotlib.pyplot as plt

import ema_workbench.analysis.cart as cart
from ema_workbench import ema_logging, load_results

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

#perform cart on modified results tuple

cart_alg = cart.setup_cart(results, classify, mass_min=0.05)
cart_alg.build_tree()

#print cart to std_out
print cart_alg.stats_to_dataframe()
print cart_alg.boxes_to_dataframe()

#visualize
cart_alg.display_boxes(together=True)
plt.show()