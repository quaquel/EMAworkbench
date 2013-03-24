'''
Created on 7 Sep 2011

@author: chamarat
'''
import matplotlib.pyplot as plt

from expWorkbench import load_results
from analysis.graphs import envelopes

results = load_results(r'.\data\TFSC_policies.cPickle')

fig = envelopes(results, column='policy', fill=True, legend=False)
fig = plt.gcf()
fig.set_size_inches(15,5)
plt.savefig("policycomparison.png", dpi=75)