"""Created on 26 sep. 2011

@author: jhkwakkel
"""

import matplotlib.pyplot as plt
from analysis.plotting import envelopes
from expWorkbench import load_results

data = load_results(r"../../../src/analysis/1000 flu cases.cPickle", zipped=False)
fig = envelopes(data, group_by="policy")
plt.show()
