"""Scenario discovery using CART."""

# Created on May 26, 2015
#
# @author: jhkwakkel
import matplotlib.pyplot as plt

import ema_workbench.analysis.cart as cart
from ema_workbench import ema_logging, load_results

ema_logging.log_to_stderr(level=ema_logging.INFO)


# load data
fn = "./data/1000 flu cases with policies.tar.gz"
results = load_results(fn)
experiments, outcomes = results

# extract results for 1 policy
logical = experiments["policy"] == "no policy"
new_experiments = experiments[logical]
new_outcomes = {}
for key, value in outcomes.items():
    new_outcomes[key] = value[logical]

x = new_experiments
y = new_outcomes["deceased_population_region_1"][:, -1] > 1000000

cart_alg = cart.CART(x, y, mass_min=0.05)
cart_alg.build_tree()

# print cart to std_out
print(cart_alg.stats_to_dataframe())
print(cart_alg.boxes_to_dataframe())

# visualize
cart_alg.show_boxes(together=False)
cart_alg.show_tree()
plt.show()
