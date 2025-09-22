"""Example demonstrates the use of PRIM for port of Rotterdam.

The dataset was generated using the world container model

(Tavasszy et al 2011; https://dx.doi.org/10.1016/j.jtrangeo.2011.05.005)


"""

import matplotlib.pyplot as plt

from ema_workbench import ema_logging, load_results
from ema_workbench.analysis import prim

# Created on Feb 13, 2014


ema_logging.log_to_stderr(ema_logging.INFO)

default_flow = 2.178849944502783e7


fn = r"./data/5000 runs WCM.tar.gz"
x, outcomes = load_results(fn)
y = (outcomes["throughput_Rotterdam"] / default_flow) < 1

prim_obj = prim.Prim(x, y)

# let's find a first box
box1 = prim_obj.find_box()

# let's analyze the peeling trajectory
box1.show_ppt()
box1.show_tradeoff()
box1.inspect_tradeoff()

box1.write_ppt_to_stdout()

# based on the peeling trajectory, we pick entry number 44
box1.select(44)

# show the resulting box
prim_obj.show_boxes()
prim_obj.boxes_to_dataframe()

plt.show()
