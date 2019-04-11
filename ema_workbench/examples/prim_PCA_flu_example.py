'''

This file illustrated the use of the workbench for doing
a PRIM analysis with PCA preprocessing

The data was generated using a system dynamics models implemented in Vensim.
See flu_example.py for the code.


'''
import matplotlib.pyplot as plt

from ema_workbench import ema_logging, load_results
import ema_workbench.analysis.prim as prim

#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

ema_logging.log_to_stderr(level=ema_logging.INFO)

# load data
fn = r'./data/1000 flu cases no policy.tar.gz'
x, outcomes = load_results(fn)

# specify y
y = outcomes['deceased population region 1'][:, -1] > 1000000

rotated_experiments, rotation_matrix = prim.pca_preprocess(x, y,
                                                           exclude=['model', 'policy'])

# perform prim on modified results tuple
prim_obj = prim.Prim(rotated_experiments, y, threshold=0.8)
box = prim_obj.find_box()

box.show_tradeoff()
box.inspect(22)
plt.show()
