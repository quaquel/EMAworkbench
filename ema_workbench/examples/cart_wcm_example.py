'''
Created on May 26, 2015

@author: jhkwakkel
'''
import matplotlib.pyplot as plt

import ema_workbench.analysis.cart as cart
from ema_workbench import ema_logging, load_results

ema_logging.log_to_stderr(level=ema_logging.INFO)

default_flow = 2.178849944502783e7

# load data
fn = './data/5000 runs WCM.tar.gz'
results = load_results(fn)
x, outcomes = results

ooi = 'throughput Rotterdam'
outcome = outcomes[ooi] / default_flow
y = outcome < 1

cart_alg = cart.CART(x, y)
cart_alg.build_tree()

# print cart to std_out
print(cart_alg.stats_to_dataframe())
print(cart_alg.boxes_to_dataframe())

# visualize
cart_alg.show_boxes(together=False)
cart_alg.show_tree()
plt.show()
