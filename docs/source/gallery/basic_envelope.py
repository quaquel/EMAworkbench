'''
Created on 26 sep. 2011

@author: jhkwakkel
'''
import matplotlib.pyplot as plt

from expWorkbench.util import load_results
from analysis.plotting import envelopes

data = load_results(r'./data/2000 flu cases no policy.bz2')
fig, axes_dict = envelopes(data, group_by='policy')
plt.savefig("./pictures/basic_envelope.png", dpi=75)