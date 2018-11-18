'''
Created on Jul 8, 2014

@author: jhkwakkel@tudelft.net
'''
import matplotlib.pyplot as plt

from ema_workbench import ema_logging, load_results

from ema_workbench.analysis.plotting import kde_over_time

ema_logging.log_to_stderr(ema_logging.INFO)

file_name = r'./data/1000 runs scarcity.tar.gz'
results = load_results(file_name)

# the plotting functions return the figure and a dict of axes
fig, axes = kde_over_time(results, log=True,
                          outcomes_to_show='supply demand ratio')

plt.show()