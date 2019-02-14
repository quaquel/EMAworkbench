

'''
Created on Jul 8, 2014

@author: jhkwakkel@tudelft.net
'''
import matplotlib.pyplot as plt

from ema_workbench import ema_logging, load_results
from ema_workbench.analysis import envelopes, Density

ema_logging.log_to_stderr(ema_logging.INFO)

file_name = r'./data/1000 flu cases with policies.tar.gz'
experiments, outcomes = load_results(file_name)

# the plotting functions return the figure and a dict of axes
fig, axes = envelopes(experiments, outcomes, group_by='policy',
                      density=Density.KDE, fill=True)

# we can access each of the axes and make changes
for key, value in axes.items():
    # the key is the name of the outcome for the normal plot
    # and the name plus '_density' for the endstate distribution
    if key.endswith('_density'):
        value.set_xscale('log')

plt.show()
