'''
Created on 16 Mar 2019

@author: jhkwakkel
'''
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ema_workbench.analysis import logistic_regression as lr
from test import utilities


def flu_classify(data):
    #get the output for deceased population
    result = data['deceased population region 1'][:, -1]
    return result > 1000000

class LogitTestCase(unittest.TestCase):

    def test_logit(self):
        experiments, outcomes = utilities.load_flu_data()
        y = flu_classify(outcomes)

        logitmodel = lr.Logit(experiments, y)

        columns = set(experiments.drop(['scenario', 'policy', 'model'], axis=1).columns.values.tolist())

        # check init
        for entry in logitmodel.feature_names:
            self.assertIn(entry, columns)

        logitmodel.run()

        logitmodel.show_tradeoff()
        logitmodel.show_threshold_tradeoff(1)
        logitmodel.plot_pairwise_scatter(1)
        logitmodel.inspect(1)

        logitmodel.threshold = 0.8

        plt.draw()
        plt.close('all')


if __name__ == '__main__':
    unittest.main()