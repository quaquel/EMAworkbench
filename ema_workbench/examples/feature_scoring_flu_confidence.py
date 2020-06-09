'''
Created on 30 Oct 2018

@author: jhkwakkel
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ema_workbench import ema_logging, load_results
from ema_workbench.analysis.feature_scoring import (get_ex_feature_scores,
                                                    RuleInductionType)

ema_logging.log_to_stderr(level=ema_logging.INFO)

# load data
fn = r'./data/1000 flu cases no policy.tar.gz'
x, outcomes = load_results(fn)

x = x.drop(['model', 'policy'], axis=1)
y = np.max(outcomes['infected fraction R1'], axis=1)


all_scores = []
for i in range(100):
    indices = np.random.choice(np.arange(0, x.shape[0]), size=x.shape[0])
    selected_x = x.iloc[indices, :]
    selected_y = y[indices]
    
    scores = get_ex_feature_scores(selected_x, selected_y,
                                   mode=RuleInductionType.REGRESSION)[0]
    all_scores.append(scores)
all_scores = pd.concat(all_scores, axis=1, sort=False)

sns.boxplot(data=all_scores.T)
plt.show()
