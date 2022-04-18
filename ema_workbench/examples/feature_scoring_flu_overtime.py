"""
Created on 30 Oct 2018

@author: jhkwakkel
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ema_workbench import ema_logging, load_results
from ema_workbench.analysis import get_ex_feature_scores, RuleInductionType

ema_logging.log_to_stderr(level=ema_logging.INFO)

# load data
fn = r"./data/1000 flu cases no policy.tar.gz"

x, outcomes = load_results(fn)
x = x.drop(["model", "policy"], axis=1)

y = outcomes["deceased population region 1"]

all_scores = []

# we only want to show those uncertainties that are in the top 5
# most sensitive parameters at any time step
top_5 = set()
for i in range(2, y.shape[1], 8):
    data = y[:, i]
    scores = get_ex_feature_scores(x, data, mode=RuleInductionType.REGRESSION)[0]
    # add the top five for this time step to the set of top5s
    top_5 |= set(scores.nlargest(5, 1).index.values)
    scores = scores.rename(columns={1: outcomes["TIME"][0, i]})
    all_scores.append(scores)

all_scores = pd.concat(all_scores, axis=1, sort=False)
all_scores = all_scores.loc[top_5, :]

fig, ax = plt.subplots()
sns.heatmap(all_scores, ax=ax, cmap="viridis")
plt.show()
