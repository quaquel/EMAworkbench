'''
Created on 26 sep. 2011

@author: jhkwakkel
'''
import matplotlib.pyplot as plt

from expWorkbench import load_results
from analysis.plotting import lines

data = load_results(r'../../../src/analysis/1000 flu cases.cPickle')
fig = lines(data, group_by='fatality ratio region 1')
plt.show()
