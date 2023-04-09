"""
Created on 12 Nov 2018

@author: jhkwakkel
"""
import matplotlib.pyplot as plt
import pandas as pd

from ema_workbench.analysis import prim
from ema_workbench.util import ema_logging

ema_logging.log_to_stderr(ema_logging.INFO)

data = pd.read_csv("./data/bryant et al 2010 data.csv", index_col=False)
x = data.iloc[:, 2:11]
y = data.iloc[:, 15].values

prim_alg = prim.Prim(x, y, threshold=0.8, peel_alpha=0.1)
box1 = prim_alg.find_box()

box1.show_tradeoff()
print(box1.resample(21))
box1.inspect(21)
box1.inspect(21, style="graph")
box1.show_pairs_scatter(21)

plt.show()
