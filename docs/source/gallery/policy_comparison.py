'''
Created on 16 nov. 2012

@author: chamarat
'''

import matplotlib.pyplot as plt

from expWorkbench import load_results
from analysis.plotting import envelopes
import analysis.plotting_util as plottingUtil

# force matplotlib to use tight layout
# see http://matplotlib.sourceforge.net/users/tight_layout_guide.html 
# for details
plottingUtil.TIGHT= True

#get the data
results = load_results(r'.\data\TFSC_corrected.bz2')

# make an envelope
fig, axesdict = envelopes(results, 
                outcomes_to_show=['total fraction new technologies'], 
                group_by='policy', 
                grouping_specifiers=['No Policy',
                                     'Basic Policy',
                                     'Optimized Adaptive Policy'],
                legend=False,
                density='kde', fill=True,titles=None)

# set the size of the figure to look reasonable nice
fig.set_size_inches(8,5)

# save figure
plt.savefig("./pictures/policy_comparison.png", dpi=75)