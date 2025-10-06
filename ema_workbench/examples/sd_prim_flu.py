"""A basic example of using PRIM for scenario discovery.

The data was generated using a system dynamics models implemented in Vensim.
See flu_example.py for the code.

"""

# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
#                 chamarat <c.hamarat  (at) tudelft (dot) nl>

import matplotlib.pyplot as plt

import ema_workbench.analysis.prim as prim
from ema_workbench import ema_logging, load_results

ema_logging.log_to_stderr(level=ema_logging.INFO)


# load data
fn = "./data/1000 flu cases no policy.tar.gz"
x, outcomes = load_results(fn)
y = outcomes["deceased_population_region_1"][:, -1] > 1000000

# perform prim on modified results tuple
prim_obj = prim.Prim(x, y)

box_1 = prim_obj.find_box()
box_1.show_ppt()
box_1.show_tradeoff()
# box_1.inspect([5, 6], style="graph", boxlim_formatter="{: .2f}")

fig, axes = plt.subplots(nrows=2, ncols=1)

box_1.inspect([5, 6], style="graph", boxlim_formatter="{: .2f}", ax=axes)
plt.show()

box_1.inspect(5)
box_1.select(5)
box_1.write_ppt_to_stdout()
box_1.show_pairs_scatter(5)

# print prim to std_out
print(prim_obj.stats_to_dataframe())
print(prim_obj.boxes_to_dataframe())

# visualize
prim_obj.show_boxes()
plt.show()
