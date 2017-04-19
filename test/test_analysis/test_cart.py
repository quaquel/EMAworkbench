'''
Created on May 22, 2015

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
import unittest

import numpy as np

from ema_workbench.analysis import cart
from .. import utilities


def flu_classify(data):
    #get the output for deceased population
    result = data['deceased population region 1']
    
    #make an empty array of length equal to number of cases 
    classes =  np.zeros(result.shape[0])
    
    #if deceased population is higher then 1.000.000 people, classify as 1 
    classes[result[:, -1] > 1000000] = 1
    
    return classes

def scarcity_classify(outcomes):
    outcome = outcomes['relative market price']
    change = np.abs(outcome[:, 1::]-outcome[:, 0:-1])
    
    neg_change = np.min(change, axis=1)
    pos_change = np.max(change, axis=1)
    
    logical = (neg_change > -0.6) & (pos_change > 0.6)
    
    classes = np.zeros(outcome.shape[0])
    classes[logical] = 1
    
    return classes

class CartTestCase(unittest.TestCase):
    def test_setup_cart(self):
        results = utilities.load_flu_data()
        
        cart_algorithm = cart.setup_cart(results, flu_classify, mass_min=0.05)
        
        # setup can be generalized --> cart and prim essentially the same
        # underlying code so move to sdutil
        
        # check include uncertainties
        
        # string type
        # --> classification vs. regresion
        
        # callable
        
        
    def test_boxes(self):
        pass
    def test_stats(self):
        pass
    def test_build_tree(self):
        pass
    
        
if __name__ == '__main__':
        unittest.main()

#     suite = unittest.TestSuite()
#     suite.addTest(PrimTestCase("test_write_boxes_to_stdout"))
#     unittest.TextTestRunner().run(suite)