'''
Created on 26 sep. 2011

@author: jhkwakkel
'''
import matplotlib.pyplot as plt

from expWorkbench import load_results
from analysis.graphs import multiplot_scatter

data = load_results(r'../../../src/analysis/1000 flu cases.cPickle')
fig = multiplot_scatter(data, column='policy', legend=True)
plt.show()
