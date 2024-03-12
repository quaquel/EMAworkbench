""" A simple example of performing regional sensitivity analysis


"""

import matplotlib.pyplot as plt

from ema_workbench.analysis import regional_sa
from ema_workbench import ema_logging, load_results

fn = "./data/1000 flu cases with policies.tar.gz"
results = load_results(fn)
x, outcomes = results

y = outcomes["deceased population region 1"][:, -1] > 1000000

fig = regional_sa.plot_cdfs(x, y)

plt.show()
