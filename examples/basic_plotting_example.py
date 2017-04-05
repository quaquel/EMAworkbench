'''
Created on Jul 8, 2014

@author: jhkwakkel@tudelft.net
'''
import matplotlib.pyplot as plt

from ema_workbench import ema_logging, load_results

from ema_workbench.analysis.plotting import envelopes 
from ema_workbench.analysis.plotting_util import KDE

ema_logging.log_to_stderr(ema_logging.INFO)

file_name = r'./data/1000 flu cases.tar.gz'
results = load_results(file_name)

# the plotting functions return the figure and a dict of axes
fig, axes = envelopes(results, group_by='policy', density=KDE, fill=True)

# we can access each of the axes and make changes
for key, value in axes.iteritems():
    # the key is the name of the outcome for the normal plot
    # and the name plus '_density' for the endstate distribution
    if key.endswith('_density'):
        value.set_xscale('log')

plt.show()