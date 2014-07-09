'''
Created on Jul 9, 2014

@author: jhkwakkel@tudelft.net
'''
import unittest

import numpy as np

from expWorkbench import ema_logging
from analysis import feature_selection as fs

class FeatureSelectionTestCase(unittest.TestCase):
    def test_prepare_experiments(self):
        x = np.array([(0,1,2,1),
                      (2,5,6,1),
                      (3,2,1,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float),
                            ('c', np.float),
                            ('d', np.float)])
        x = fs._preperate_experiments(x)
        
        correct = np.array([[0,1,2,1],
                            [2,5,6,1],
                            [3,2,1,1]], dtype=np.float)
        
        self.assertTrue(np.all(x==correct))
        
        # heterogeneous without NAN
        dtype = [('a', np.float),('b', np.int), ('c', np.object)]
        x = np.empty((10, ), dtype=dtype)
        
        x['a'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        x['b'] = [0,1,2,3,4,5,6,7,8,9]
        x['c'] = ['a','b','a','b','a','a','b','a','b','a', ]
        x = fs._preperate_experiments(x)
       
        correct = np.array([[0.1, 0, 0],
                            [0.2, 1, 1],
                            [0.3, 2, 0],
                            [0.4, 3, 1],
                            [0.5, 4, 0],
                            [0.6, 5, 0],
                            [0.7, 6, 1],
                            [0.8, 7, 0],
                            [0.9, 8, 1],
                            [1.0, 9, 0]
                            ], dtype=np.float)

        self.assertTrue(np.all(x==correct))
        
    
if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)   
    unittest.main()