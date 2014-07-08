'''
Created on Jul 8, 2014

@author: jhkwakkel@tudelft.net
'''
import matplotlib.pyplot as plt

from expWorkbench import load_results, ema_logging
from analysis.plotting import envelopes, KDE

ema_logging.log_to_stderr(ema_logging.INFO)
file_name = r'./data/1000 flu cases.tar.gz'
results = load_results(file_name)

experiments, outcomes = results
print experiments.shape
for key, value in outcomes.iteritems():
    print value.shape

# envelopes(results, group_by='policy', density=KDE, fill=True)
# plt.show()