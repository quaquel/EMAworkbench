"""Created on 26 sep. 2011

@author: jhkwakkel
"""

import matplotlib.pyplot as plt
from analysis.pairs_plotting import pairs_scatter
from expWorkbench import load_results

data = load_results(r"../../../src/analysis/1000 flu cases.cPickle", zipped=False)
fig = pairs_scatter(data, group_by="policy", legend=True)
plt.show()
