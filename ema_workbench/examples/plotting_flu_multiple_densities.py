'''
Created on Jul 8, 2014

@author: jhkwakkel@tudelft.net
'''
import math
import matplotlib.pyplot as plt

from ema_workbench import ema_logging, load_results
from ema_workbench.analysis import multiple_densities, Density

ema_logging.log_to_stderr(ema_logging.INFO)

file_name = './data/1000 flu cases with policies.tar.gz'
experiments, outcomes = load_results(file_name)

# pick points in time for which we want to see a
# density subplot
time = outcomes["TIME"][0, :]
times = time[1::math.ceil(time.shape[0] / 6)].tolist()

multiple_densities(experiments, outcomes, log=True, points_in_time=times,
                   group_by='policy', density=Density.KDE, fill=True)

plt.show()
