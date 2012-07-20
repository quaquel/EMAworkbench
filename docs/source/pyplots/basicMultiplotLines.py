import numpy as np
import matplotlib.pyplot as plt

from analysis.graphs import multiplot_lines
from expWorkbench.util import load_results


#load the data
data = load_results(r'../../../src/analysis/100 flu cases.cPickle')

multiplot_lines(data, column='policy')
plt.show() 