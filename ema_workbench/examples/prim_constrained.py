'''
a short example on how to use the constrained prim function.

for more details see Kwakkel (2019) A generalized many‐objective optimization
approach for scenario discovery, doi: https://doi.org/10.1002/ffo2.8

'''
import pandas as pd
import matplotlib.pyplot as plt

from ema_workbench.analysis import prim
from ema_workbench.util import ema_logging

ema_logging.log_to_stderr(ema_logging.INFO)

data = pd.read_csv('./data/bryant et al 2010 data.csv', index_col=False)
x = data.iloc[:, 2:11]
y = data.iloc[:, 15].values

box = prim.run_constrained_prim(x, y, peel_alpha=0.1)

box.show_tradeoff()
box.inspect(35)
box.inspect(35, style='graph')

plt.show()
