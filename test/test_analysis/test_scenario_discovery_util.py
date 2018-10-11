'''
Created on May 22, 2015

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import (absolute_import, print_function, division)
import unittest

import numpy as np
import pandas as pd

from ema_workbench.analysis import scenario_discovery_util as sdutil


class ScenarioDiscoveryUtilTestCase(unittest.TestCase):
    def test_get_sorted_box_lims(self):
        x = pd.DataFrame([(0,1,2),
                          (2,5,6),
                          (3,2,1)], 
                          columns=['a', 'b', 'c'])
        
        box_init = sdutil._make_box(x)
        
        box_lim = pd.DataFrame([(0,1,1),
                                (2,5,2)],
                                columns=['a', 'b', 'c'])
        
        
        _, uncs = sdutil._get_sorted_box_lims([box_lim], box_init)
        
        self.assertEqual(uncs, ['c','a'])
    
    def test_in_box(self):
        x = pd.DataFrame([(0,),
                          (1,),
                          (2,),
                          (3,),
                          (4,),
                          (5,),
                          (6,),
                          (7,),
                          (8,),
                          (9,)], 
                         columns=['a'])
        boxlim = pd.DataFrame([(1,),
                               (8,)], columns=['a'])
        correct_result = np.array([[1,2,3,4,5,6,7,8]], dtype=np.int).T
        logical = sdutil._in_box(x, boxlim)
        result = x.loc[logical]
        self.assertTrue(np.all(correct_result==result.values))

        x = pd.DataFrame([(0,0),
                          (1,1),
                          (2,2),
                          (3,3),
                          (4,4),
                          (5,5),
                          (6,6),
                          (7,7),
                          (8,8),
                          (9,9)], 
                         columns=['a', 'b'])
        boxlim = pd.DataFrame([(1,0),
                              (8,7)], columns=['a', 'b'])
        correct_result = np.array([[1,2,3,4,5,6,7]], dtype=np.int).T
        logical = sdutil._in_box(x, boxlim)
        result = x.loc[logical]
        self.assertTrue(np.all(correct_result==result))

        x = pd.DataFrame([(0.1, 0, 'a'),
                          (1.1, 1, 'a'),
                          (2.1, 2, 'b'),
                          (3.1, 3, 'b'),
                          (4.1, 4, 'c'),
                          (5.1, 5, 'c'),
                          (6.1, 6, 'd'),
                          (7.1, 7, 'd'),
                          (8.1, 8, 'e'),
                          (9.1, 9, 'e')], 
                          columns=['a', 'b', 'c'])
        boxlim = pd.DataFrame([(1.2, 0, set(['a','b'])),
                               (8.0, 7, set(['a','b']) )],
                               columns=['a', 'b', 'c'])
        correct_result = x.loc[[2,3], :]
        logical = sdutil._in_box(x, boxlim)
        result = x.loc[logical]
        self.assertTrue(np.all(correct_result==result))
        
        boxlim = pd.DataFrame([(0.1, 0, set(['a','b','c','d','e'])),
                               (9.1, 9, set(['a','b','c','d','e']) )], 
                               columns=['a', 'b', 'c'])
        correct_result = x.loc[[0,1,2,3,4,5,6,7,8,9], :]
        logical = sdutil._in_box(x, boxlim)
        result = x.loc[logical]
        self.assertTrue(np.all(correct_result==result))
    
    
    def test_make_box(self):
        x = pd.DataFrame([(0, 1, 2, 'a'),
                          (2, 5, 6, 'b'),
                          (3, 2, 1, 'c')], 
                         columns=['a', 'b', 'c', 'd'])
        
        box_lims = sdutil._make_box(x)
        
        # some test on the box
        self.assertEqual(np.min(box_lims['a']), 0, 'min a fails')
        self.assertEqual(np.max(box_lims['a']), 3, 'max a fails')
        
        self.assertEqual(np.min(box_lims['b']), 1, 'min b fails')
        self.assertEqual(np.max(box_lims['b']), 5, 'max c fails')
        
        self.assertEqual(np.min(box_lims['c']), 1, 'min c fails')
        self.assertEqual(np.max(box_lims['c']), 6, 'max c fails')

        self.assertEqual(box_lims.loc[0, 'd'],
                         {'a', 'b', 'c'}, 'min d fails')
        self.assertEqual(box_lims.loc[1, 'd'], 
                         {'a', 'b', 'c'}, 'max d fails')    
    
    def test_normalize(self):
        x = pd.DataFrame([(0,1,2, 'a'),
                          (2,5,6, 'b'),
                          (3,2,1, 'c')], 
                          columns=['a', 'b', 'c', 'd'])
            
        box_init = sdutil._make_box(x)
        
        box_lim = pd.DataFrame([(0,1,1, {'a', 'b'}),
                                (2,5,2, {'a', 'b'})],
                                columns=['a', 'b', 'c', 'd'])
        uncs = box_lim.columns.values.tolist()
        normalized = sdutil._normalize(box_lim, box_init, uncs)
        
        for i, lims in enumerate([(0, 2/3),(0, 1),(0,0.2), (0, 2/3)]):
            lower, upper = lims
            self.assertAlmostEqual(normalized[i, 0], lower, 
                                   msg='lower unequal for '+uncs[i])
            self.assertAlmostEqual(normalized[i, 1], upper, 
                                   msg='upper unequal for '+uncs[i])
            
    def test_determine_restricted_dims(self):
        x = np.random.rand(5, 2)
        x = pd.DataFrame(x, columns=['a', 'b'])

        
        # all dimensions the same
        box_init = sdutil._make_box(x)
        u = sdutil._determine_restricted_dims(box_init, box_init)
        
        self.assertEqual(list(u), [])
        
        # dimensions 1 different and dimension 2 the same
        b = pd.DataFrame([(1,1),
                          (0,1)], 
                          columns=['a', 'b'])
        u = sdutil._determine_restricted_dims(b, box_init)
        
        self.assertEqual(list(u), ['a', 'b'])
  
    def test_determine_nr_restricted_dims(self):
        x = np.random.rand(5, 2)
        x = pd.DataFrame(x, columns=['a', 'b'])

        
        # all dimensions the same
        box_init = sdutil._make_box(x)
        n = sdutil._determine_nr_restricted_dims(box_init, box_init)
        
        self.assertEqual(n, 0)
        
        # dimensions 1 different and dimension 2 the same
        b = pd.DataFrame([(1,1),
                          (0,1)], 
                          columns=['a', 'b'])
        n = sdutil._determine_nr_restricted_dims( b, box_init)
        self.assertEqual(n, 2)
    
    def test_compare(self):
        # all dimensions the same
        a = np.array([(0,1),
                      (0,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float)])
        b = np.array([(0,1),
                      (0,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float)])
        
        self.assertTrue(np.all(sdutil._compare(a,b)))
        
        # all dimensions different
        a = np.array([(0,1),
                      (0,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float)])
        b = np.array([(1,1),
                      (0,0)], 
                     dtype=[('a', np.float),
                            ('b', np.float)])
        test = sdutil._compare(a,b)==False
        self.assertTrue(np.all(test))
        
        # dimensions 1 different and dimension 2 the same
        a = np.array([(0,1),
                      (0,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float)])
        b = np.array([(1,1),
                      (0,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float)])
        test = sdutil._compare(a,b)
        test = (test[0]==False) & (test[1]==True)
        self.assertTrue(test)
    
    def test_boxes_to_dataframe(self):
        pass
    
    def test_stats_to_dataframe(self):
        pass
    
    def test_display_boxes(self):
        pass
    
if __name__ == '__main__':
        unittest.main()

#     suite = unittest.TestSuite()
#     suite.addTest(PrimTestCase("test_write_boxes_to_stdout"))
#     unittest.TextTestRunner().run(suite)