'''
Created on 30 Oct 2018

@author: jhkwakkel
'''
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ema_workbench import ema_logging, load_results
from ema_workbench.analysis import feature_scoring

ema_logging.log_to_stderr(level=ema_logging.INFO)


# load data
fn = r'./data/1000 flu cases with policies.tar.gz'
x, outcomes = load_results(fn)


# we have timeseries so we need scalars
y = {'deceased population': outcomes['deceased population region 1'][:, -1],
     'max. infected fraction': np.max(outcomes['infected fraction R1'], axis=1)}


scores = feature_scoring.get_feature_scores_all(x, y)
sns.heatmap(scores, annot=True, cmap='viridis')
plt.show()
