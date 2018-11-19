'''
Created on 19 Nov 2018

@author: jhkwakkel
'''
import unittest

import matplotlib.pyplot as plt

from ema_workbench.analysis import regional_sa
from test import utilities

class Test(unittest.TestCase):


    def test_plot_cdfs(self):
        x, outcomes = utilities.load_flu_data()
        y = outcomes['deceased population region 1'][:,-1] > 1000000
        
        regional_sa.plot_cdfs(x, y)
        regional_sa.plot_cdfs(x, y, ccdf=True)
        
        x = x.drop('scenario', axis=1)
        regional_sa.plot_cdfs(x, y, ccdf=True)


    def test_plot__individual_cdf(self):
        x, outcomes = utilities.load_flu_data()
        y = outcomes['deceased population region 1'][:,-1] > 1000000


        fig, ax = plt.subplots()
        unc = 'fatality ratio region 1'
        
        regional_sa.plot_individual_cdf(ax, unc, x[unc], y, 
                                        discrete=False, legend=True,
                                        xticklabels_on=True, 
                                        yticklabels_on=True)
        
        fig, ax = plt.subplots()
        unc = 'model'
        
        regional_sa.plot_individual_cdf(ax, unc, x[unc], y, 
                                        discrete=True, legend=True,
                                        xticklabels_on=True, 
                                        yticklabels_on=True)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()