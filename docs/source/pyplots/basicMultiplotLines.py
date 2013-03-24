import numpy as np
import matplotlib.pyplot as plt

from analysis.pairs_plotting import pairs_lines
from expWorkbench.util import load_results


#load the data
data = load_results(r'../../../src/analysis/100 flu cases.cPickle', zipped=False)

pairs_lines(data, group_by='policy')
plt.show() 