'''
Created on Nov 30, 2015

@author: jhkwakkel
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import unittest

from ema_workbench.analysis import dimensional_stacking
from test import utilities

class DimStackTestCase(unittest.TestCase):
    
    def test_discretize(self):
        
        float = np.random.rand(100,)  # @ReservedAssignment
        integer = np.random.randint(0, 5, size=(100,))
        categorical = [str(i) for i in np.random.randint(0, 3,
                                                         size=(100,))]
        data = {"float":float, "integer":integer,
                "categorical":categorical}
        
        data = pd.DataFrame(data)
        data['categorical'] = data['categorical'].astype('category')
        discretized = dimensional_stacking.discretize(data)
        
        self.assertTrue(np.all(discretized.apply(pd.Series.nunique)==3))
        
        nbins = 6
        discretized = dimensional_stacking.discretize(data, nbins=nbins)
        nunique = discretized.apply(pd.Series.nunique)
        
        self.assertTrue(nunique.loc["float"]==6)
        self.assertTrue(nunique.loc["integer"]==5)
        self.assertTrue(nunique.loc["categorical"]==3)

    def test_create_pivot_plot(self):
        x, outcomes = utilities.load_flu_data()
        y = outcomes['deceased population region 1'][:, -1] > 1000000

        dimensional_stacking.create_pivot_plot(x, y, 2)
        dimensional_stacking.create_pivot_plot(x, y, 2, labels=False,
                                               bin_labels=True )
        dimensional_stacking.create_pivot_plot(x, y, 1, labels=False)
        plt.draw()
        plt.close('all')
    
    def test_dim_ratios(self):
        ratios = dimensional_stacking.dim_ratios(0, (6,6))
        
        for x,y in zip(ratios,
                       [0.16, 0.04, 0.8]):
                self.assertAlmostEqual(x, y)

        ratios = dimensional_stacking.dim_ratios(1, (6,6))
        for x,y in zip(ratios,
                       [0.04, 0.16, 0.8]):
                self.assertAlmostEqual(x, y)

    
    def test_plot_pivot_table(self):
        pass

    
if __name__ == '__main__':
    unittest.main()
