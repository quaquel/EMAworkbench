'''
Created on Nov 30, 2015

@author: jhkwakkel
'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import unittest
import pandas as pd
import numpy as np

from ema_workbench.analysis import dimensional_stacking


class DimStackTestCase(unittest.TestCase):
    
    def test_discretize(self):
        
        float = np.random.rand(100,)  # @ReservedAssignment
        integer = np.random.randint(0, 5, size=(100,))
        categorical = [str(i) for i in np.random.randint(0, 3, size=(100,))]
        data = {"float":float, "integer":integer, "categorical":categorical}
        
        
        data = pd.DataFrame(data)
        discretized = dimensional_stacking.discretize(data)
        
        self.assertTrue(np.all(discretized.apply(pd.Series.nunique)==3))
        
        nbins = 6
        discretized = dimensional_stacking.discretize(data, nbins=nbins)
        nunique = discretized.apply(pd.Series.nunique)
        
        self.assertTrue(nunique.loc["float"]==6)
        self.assertTrue(nunique.loc["integer"]==5)
        self.assertTrue(nunique.loc["categorical"]==3)


    def test_create_pivot_plot(self):
        pass
    
    def test_dim_rations(self):
        pass
    
    def test_make_pivot_table(self):
        pass
    
    def test_plot_pivot_table(self):
        pass
    
if __name__ == '__main__':
    unittest.main()
