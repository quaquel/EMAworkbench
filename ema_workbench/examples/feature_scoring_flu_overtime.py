'''
Created on 30 Oct 2018

@author: jhkwakkel
'''
import matplotlib.pyplot as plt
import pandas as pd

from ema_workbench import ema_logging, load_results
from ema_workbench.analysis import (get_ex_feature_scores,
                                    RuleInductionType)

ema_logging.log_to_stderr(level=ema_logging.INFO)

# load data
fn = r'./data/1000 flu cases no policy.tar.gz'
x, outcomes = load_results(fn)

x = x.drop(['model', 'policy'], axis=1)
y = outcomes['deceased population region 1']


#
# 'infected fraction R1'

all_scores = []
for i in range(0, y.shape[1], 2):
    data = y[:, i]
    scores = get_ex_feature_scores(x, data,
                                   mode=RuleInductionType.REGRESSION)[0]

    all_scores.append(scores)
all_scores = pd.concat(all_scores, axis=1, sort=False)
all_scores[all_scores < 0.075] = 0  # cleans up results
normalized = all_scores.divide(all_scores.sum(axis=1), axis=0)
normalized = all_scores.divide(all_scores.sum(axis=1), axis=0)

labels = normalized.index.values
print(labels)
y = normalized.values

fig, ax = plt.subplots()
ax.stackplot(range(len(labels)), y.T, labels=labels)
# ax.legend()
plt.show()
