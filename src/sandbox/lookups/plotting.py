'''
Created on Sep 18, 2012

@author: sibeleker
'''
import numpy as np
import matplotlib.pyplot as plt
from expWorkbench import load_results
from analysis.graphs import lines, envelopes

results = load_results('lookup_3methods.cpickle')
outcomes_distorted =['TF']
lines(results, outcomes=outcomes_distorted, density=True, hist=True)
plt.show()