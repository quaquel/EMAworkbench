'''
Created on 26 sep. 2011

@author: jhkwakkel
'''
import matplotlib.pyplot as plt

from expWorkbench import load_results
from analysis.plotting import lines, KDE

data = load_results(r'../../../src/analysis/1000 flu cases.cPickle', zipped=False)
fig = lines(data, density=KDE, hist=True)
plt.show()
