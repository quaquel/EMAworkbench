'''
Created on Mar 22, 2014

@author: jhkwakkel
'''
from matplotlib.mlab import rec2csv
import numpy as np
from expWorkbench import TIME

from expWorkbench import load_results

def write_results_to_csv(results, directory):

    experiments, outcomes = results
#     deceased_pop = outcomes['relative market price']
#     time = outcomes[TIME]
    
    rec2csv(experiments, directory+'/experiments.csv', withheader=True)
    
    for key, value in outcomes.iteritems():
        np.savetxt(directory+'/{}.csv'.format(key), value, delimiter=',')
#     np.savetxt('./data/scarcity/relative_market_price.csv', deceased_pop, delimiter=',')
#     np.savetxt('./data/scarcity/time.csv', time, delimiter=',')
#     
#     for entry in experiments.dtype.descr:
#         print entry

fn =r'./data/scarcity 1000.bz2'
results = load_results(fn)
write_results_to_csv(results, './data/scarcity')