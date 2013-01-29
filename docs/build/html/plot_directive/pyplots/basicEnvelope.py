'''
Created on 26 sep. 2011

@author: jhkwakkel
'''
import matplotlib.pyplot as plt

from expWorkbench import load_results
from analysis.graphs import envelopes

data = load_results(r'../../../src/analysis/1000 flu cases.cPickle')
fig = envelopes(data, column='policy')
plt.show()

