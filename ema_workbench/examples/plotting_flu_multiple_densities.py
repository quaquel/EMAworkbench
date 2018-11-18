'''
Created on Jul 8, 2014

@author: jhkwakkel@tudelft.net
'''
import matplotlib.pyplot as plt
import numpy as np

from ema_workbench import ema_logging, load_results

from ema_workbench.analysis.plotting import multiple_densities

ema_logging.log_to_stderr(ema_logging.INFO)

file_name = './data/1000 flu cases with policies.tar.gz'
experiments, outcomes = load_results(file_name)

# pick points in time for which we want to see a 
# density subplot
time = outcomes["TIME"][0,:]
times = time[::int(np.ceil(time.shape[0]/6))].tolist()

multiple_densities(experiments, outcomes, log=True, points_in_time=times,
                   group_by='policy', density='kde', fill=True)
 
plt.show()