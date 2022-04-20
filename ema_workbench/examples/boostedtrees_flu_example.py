"""

"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import CircleCollection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from ema_workbench import load_results, ema_logging
from ema_workbench.analysis import feature_scoring

ema_logging.log_to_stderr(ema_logging.INFO)


def plot_factormap(x1, x2, ax, bdt, nominal):
    """helper function for plotting a 2d factor map"""
    x_min, x_max = x[:, x1].min(), x[:, x1].max()
    y_min, y_max = x[:, x2].min(), x[:, x2].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    grid = np.ones((xx.ravel().shape[0], x.shape[1])) * nominal
    grid[:, x1] = xx.ravel()
    grid[:, x2] = yy.ravel()

    Z = bdt.predict(grid)
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)  # @UndefinedVariable

    for i in (0, 1):
        idx = y == i
        ax.scatter(x[idx, x1], x[idx, x2], s=5)
    ax.set_xlabel(columns[x1])
    ax.set_ylabel(columns[x2])


def plot_diag(x1, ax):
    x_min, x_max = x[:, x1].min(), x[:, x1].max()
    for i in (0, 1):
        idx = y == i
        ax.hist(x[idx, x1], range=(x_min, x_max), alpha=0.5)


# load data
experiments, outcomes = load_results("./data/1000 flu cases with policies.tar.gz")

# transform to numpy array with proper recoding of cateogorical variables
x, columns = feature_scoring._prepare_experiments(experiments)
y = outcomes["deceased population region 1"][:, -1] > 1000000

# establish mean case for factor maps
# this is questionable in particular in case of categorical dimensions
minima = x.min(axis=0)
maxima = x.max(axis=0)
nominal = minima + (maxima - minima) / 2

# fit the boosted tree
bdt = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=3), algorithm="SAMME", n_estimators=200
)
bdt.fit(x, y)

# determine which dimensions are most important
sorted_indices = np.argsort(bdt.feature_importances_)[::-1]

# do the actual plotting
# this is a quick hack, tying it to seaborn Pairgrid is probably
# the more elegant solution, but is tricky with what arguments
# can be passed to the plotting function
fig, axes = plt.subplots(ncols=5, nrows=5, figsize=(15, 15))

for i, row in enumerate(axes):
    for j, ax in enumerate(row):
        if i > j:
            plot_factormap(sorted_indices[j], sorted_indices[i], ax, bdt, nominal)
        elif i == j:
            plot_diag(sorted_indices[j], ax)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")

        if j > 0:
            ax.set_yticklabels([])
            ax.set_ylabel("")
        if i < len(axes) - 1:
            ax.set_xticklabels([])
            ax.set_xlabel("")

# add the legend
# Draw a full-figure legend outside the grid
handles = [
    CircleCollection([10], color=sns.color_palette()[0]),
    CircleCollection([10], color=sns.color_palette()[1]),
]

legend = fig.legend(handles, ["False", "True"], scatterpoints=1)

plt.tight_layout()
plt.show()
