

'''
Created on Jul 8, 2014

@author: jhkwakkel@tudelft.net
'''
import matplotlib.pyplot as plt
import numpy as np

from ema_workbench import ema_logging, load_results
from ema_workbench.analysis import lines, Density

ema_logging.log_to_stderr(ema_logging.INFO)

file_name = r'./data/1000 flu cases with policies.tar.gz'
experiments, outcomes = load_results(file_name)


# let's specify a few scenarios that we want to show for
# each of the policies
scenario_ids = np.arange(0, 1000, 100)
experiments_to_show = experiments['scenario_id'].isin(scenario_ids)

# the plotting functions return the figure and a dict of axes
fig, axes = lines(experiments, outcomes, group_by='policy',
                  density=Density.KDE, show_envelope=True,
                  experiments_to_show=experiments_to_show)

# we can access each of the axes and make changes
for key, value in axes.items():
    # the key is the name of the outcome for the normal plot
    # and the name plus '_density' for the endstate distribution
    if key.endswith('_density'):
        value.set_xscale('log')

plt.show()
