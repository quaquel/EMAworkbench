'''

'''
import matplotlib.pyplot as plt
import seaborn as sns


from ema_workbench import load_results
import ema_workbench.analysis.logistic_regression as logistic_regression

# Created on 14 Mar 2019
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


experiments, outcomes = load_results('./data/1000 flu cases no policy.tar.gz')

x = experiments.drop(['model', 'policy'], axis=1)
y = outcomes['deceased population region 1'][:, -1] > 1000000

logit = logistic_regression.Logit(x, y)
logit.run()

logit.show_tradeoff()

# when we change the default threshold, the tradeoff curve is
# recalculated
logit.threshold = 0.8
logit.show_tradeoff()

# we can also look at the tradeoff across threshold values
# for a given model
logit.show_threshold_tradeoff(3)

# inspect shows the threshold tradeoff for the model
# as well as the statistics of the model
logit.inspect(3)

# we can also visualize the performance of the model
# using a pairwise scatter plot
sns.set_style('white')
logit.plot_pairwise_scatter(3)

plt.show()
