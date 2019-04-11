'''

This file illustrated the use of the workbench for doing
a PRIM analysis.

The data was generated using a system dynamics models implemented in Vensim.
See flu_example.py for the code.


.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
                chamarat <c.hamarat  (at) tudelft (dot) nl>

'''
import matplotlib.pyplot as plt

from ema_workbench import ema_logging, load_results
import ema_workbench.analysis.prim as prim

ema_logging.log_to_stderr(level=ema_logging.INFO)


def classify(data):
    # get the output for deceased population
    ooi = data['deceased population region 1']
    return ooi[:, -1] > 1000000


# load data
fn = r'./data/1000 flu cases no policy.tar.gz'
results = load_results(fn)

# perform prim on modified results tuple
prim_obj = prim.setup_prim(results, classify, threshold=0.8, threshold_type=1)

box_1 = prim_obj.find_box()
box_1.show_ppt()
box_1.show_tradeoff()
box_1.inspect(5, style='graph', boxlim_formatter="{: .2f}")
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
