'''

This file illustrated the use of the workbench for using dimensional
stacking for scenario discovery


.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

'''
import matplotlib.pyplot as plt

from ema_workbench import ema_logging, load_results
from ema_workbench.analysis import dimensional_stacking

ema_logging.log_to_stderr(level=ema_logging.INFO)

# load data
fn = './data/1000 flu cases no policy.tar.gz'
x, outcomes = load_results(fn)

y = outcomes['deceased population region 1'][:, -1] > 1000000

fig = dimensional_stacking.create_pivot_plot(x, y, 2, bin_labels=True)

plt.show()
