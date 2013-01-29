'''
Created on Aug 21, 2012

@author: jhkwakkel
'''
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

from expWorkbench import load_results
from analysis.plotting import lines, envelopes
from analysis.b_and_w_plotting import set_fig_to_bw

# load results
results = load_results(r'.\data\JotKE 50000.bz2')
experiments, outcomes =  results

# pre process the data
new_outcomes = {}
new_outcomes['total capacity'] = outcomes['capa central'] + outcomes['capa decentral']
new_outcomes['total generation'] = outcomes['gen central'] + outcomes['gen decentral']
new_outcomes['total fossil'] = outcomes['central coal'] + outcomes['central gas'] + outcomes['decentral gas']
new_outcomes['total non-fossil'] = new_outcomes['total generation'] - new_outcomes['total fossil']
new_outcomes['avg. price'] = outcomes['avg price']
new_outcomes['fraction non-fossil'] = new_outcomes['total non-fossil'] / new_outcomes['total generation'] 

# create the time dimension including 2006 as a starting year
time = np.arange(0, new_outcomes['avg. price'].shape[1])+2006
time = np.tile(time, (new_outcomes['avg. price'].shape[0],1))
new_outcomes["TIME"] = time

results = (experiments, new_outcomes)

# create a lines plot on top of an envelope
fig, axes_dict = lines(results,
                       density='kde',
                       outcomes_to_show=['total capacity',
                                         'total generation',
                                         'avg. price',
                                         'fraction non-fossil'],
                       show_envelope=True,
                       experiments_to_show=np.random.randint(0, new_outcomes['avg. price'].shape[0], (5,)),
                       titles=None,
                       )

# use the returned axes dict to modify the ylim on one of the outcomes
axes_dict['fraction non-fossil'].set_ylim(ymin=0, ymax=1)
axes_dict['fraction non-fossil_density'].set_ylim(ymin=0, ymax=1)

# transform the figure to black and white
set_fig_to_bw(fig)

plt.savefig("./pictures/jotke_envelopes.png", dpi=75)



