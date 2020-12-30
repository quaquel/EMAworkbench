'''
Created on May 22, 2015

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
import unittest

import matplotlib.pyplot as plt
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
        x['c'] = x['c'].astype('category')
        
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
        x = pd.DataFrame([(0,1,2),
                      (2,5,6),
                      (3,2,1)], 
                     columns=['a', 'b', 'c'])
        
        box_lims = sdutil._make_box(x)
        
        # some test on the box
        self.assertEqual(np.min(box_lims['a']), 0, 'min a fails')
        self.assertEqual(np.max(box_lims['a']), 3, 'max a fails')
        
        self.assertEqual(np.min(box_lims['b']), 1, 'min b fails')
        self.assertEqual(np.max(box_lims['b']), 5, 'max c fails')
        
        self.assertEqual(np.min(box_lims['c']), 1, 'min c fails')
        self.assertEqual(np.max(box_lims['c']), 6, 'max c fails')
    
    
    def test_normalize(self):
        x = pd.DataFrame([(0,1,2),
                          (2,5,6),
                          (3,2,1)], 
                          columns=['a', 'b', 'c'])
            
        box_init = sdutil._make_box(x)
        
        box_lim = pd.DataFrame([(0,1,1),
                                (2,5,2)],
                                columns=['a', 'b', 'c'])
        uncs = box_lim.columns.values.tolist()
        normalized = sdutil._normalize(box_lim, box_init, uncs)
        
        for i, lims in enumerate([(0, 2/3),(0, 1),(0,0.2)]):
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
        
    def test_plot_box(self):
        x = pd.DataFrame([[0.1, 0, 'a'],
                          [0.2, 1, 'b'],
                          [0.3, 2, 'a'],
                          [0.4, 3, 'b'],
                          [0.5, 4, 'a'],
                          [0.6, 5, 'a'],
                          [0.7, 6, 'b'],
                          [0.8, 7, 'a'],
                          [0.9, 8, 'b'],
                          [1.0, 9, 'a']], 
                          columns=['a', 'b', 'c'])
        y = np.zeros((x.shape[0],), dtype=np.int)         
        y[(x.a>0.5) & (x.c!='a')] = 1
        
        x['c'] = x['c'].astype('category')   
        
        box_init = sdutil._make_box(x)
        boxlim = box_init.copy()
        boxlim.a = [0.5, 1.0]
        boxlim.c = [set('b',),]*2
        restricted_dims = ['a', 'c']        
        
        qp_values = {'a': [0.05, 0.9], 
                     'c': [0.05, -1]}
        
        sdutil.plot_box(boxlim, qp_values, box_init, restricted_dims, 1, 1) 
        plt.draw()
        plt.close('all')       
    
    def test_plot_pairwise_scatter(self):
        x = pd.DataFrame([[0.1, 0, 'a'],
                          [0.2, 1, 'b'],
                          [0.3, 2, 'a'],
                          [0.4, 3, 'b'],
                          [0.5, 4, 'a'],
                          [0.6, 5, 'a'],
                          [0.7, 6, 'b'],
                          [0.8, 7, 'a'],
                          [0.9, 8, 'b'],
                          [1.0, 9, 'a']], 
                          columns=['a', 'b', 'c'])
        y = np.zeros((x.shape[0],), dtype=np.int)         
        y[(x.a>0.5) & (x.c!='a')] = 1
        
        x.loc[6, 'c'] = 'a'
        
        x['c'] = x['c'].astype('category')   
        
        box_init = sdutil._make_box(x)
        boxlim = box_init.copy()
        boxlim.a = [0.5, 1.0]
        boxlim.c = [set('b',),]*2
        restricted_dims = ['a', 'c']
    
        sdutil.plot_pair_wise_scatter(x, y, boxlim, box_init, restricted_dims)
        plt.draw()
        plt.close('all')
        
    def test_plot_boxes(self):
        x = pd.DataFrame([[0.1, 0, 'a'],
                          [0.2, 1, 'b'],
                          [0.3, 2, 'a'],
                          [0.4, 3, 'b'],
                          [0.5, 4, 'a'],
                          [0.6, 5, 'a'],
                          [0.7, 6, 'b'],
                          [0.8, 7, 'a'],
                          [0.9, 8, 'b'],
                          [1.0, 9, 'a']], 
                          columns=['a', 'b', 'c'])
        y = np.zeros((x.shape[0],), dtype=np.int)       
        logical = (x.a>0.5) & (x.c!='a')
        y[logical] = 1

        logical = (x.a<0.5) & (x.c!='b')
        y[logical] = 1

        x['c'] = x['c'].astype('category')   
        
        box_init = sdutil._make_box(x)
        boxlim1 = box_init.copy()
        boxlim1.a = [0.5, 1]
        boxlim1.c = [set('b',),]*2
        
        boxlim2 = box_init.copy()
        boxlim2.a = [0.1, 0.5]
        boxlim2.c = [set('a',),]*2

        sdutil.plot_boxes(x, [boxlim1, boxlim2], together=True)
        sdutil.plot_boxes(x, [boxlim1, boxlim2], together=False)
        plt.draw()
        plt.close('all')

    def test_OutputFormatterMixin(self):
        x = pd.DataFrame([[0.1, 0, 'a'],
                          [0.2, 1, 'b'],
                          [0.3, 2, 'a'],
                          [0.4, 3, 'b'],
                          [0.5, 4, 'a'],
                          [0.6, 5, 'a'],
                          [0.7, 6, 'b'],
                          [0.8, 7, 'a'],
                          [0.9, 8, 'b'],
                          [1.0, 9, 'a']], 
                          columns=['a', 'b', 'c'])
        y = np.zeros((x.shape[0],), dtype=np.int)       
        logical = (x.a>0.5) & (x.c!='a')
        y[logical] = 1

        logical = (x.a<0.5) & (x.c!='b')
        y[logical] = 1

        x['c'] = x['c'].astype('category')   
        
        box_init = sdutil._make_box(x)
        boxlim1 = box_init.copy()
        boxlim1.a = [0.5, 1]
        boxlim1.c = [set('b',),]*2
        
        boxlim2 = box_init.copy()
        boxlim2.a = [0.1, 0.5]
        boxlim2.c = [set('a',),]*2
        
        with self.assertRaises(AttributeError):
            class WrongTestFormatter(sdutil.OutputFormatterMixin):
                pass
            formatter = WrongTestFormatter()
            formatter.boxes = [boxlim1, boxlim2]
            formatter.stats = [{'coverage':0.5, 'density':1},
                               {'coverage':0.5, 'density':1}]        
        
        class TestFormatter(sdutil.OutputFormatterMixin):
            boxes = []
            stats = []
        
        formatter = TestFormatter()
        formatter.boxes = [boxlim1, boxlim2]
        formatter.stats = [{'coverage':0.5, 'density':1},
                           {'coverage':0.5, 'density':1}]
        formatter.x = x
        
        formatter.show_boxes()
        plt.draw()
        plt.close('all')
        
        boxes = formatter.boxes_to_dataframe()

        expected_boxes = pd.DataFrame([[{'b'}, {'b'}, {'a'}, {'a'}],
                                       [0.5, 1, 0.1, 0.5]], index=['c', 'a'],
                    columns=pd.MultiIndex(levels=[['box 1', 'box 2'],
                                                  ['max', 'min']],
                                          codes=[[0, 0, 1, 1], [1, 0, 1, 0]]))
        self.assertTrue(expected_boxes.equals(boxes))
        
        # check stats
        stats = formatter.stats_to_dataframe()
        expected_stats = pd.DataFrame([[0.5, 1], [0.5, 1]],
                                      index=['box 1', 'box 2'],
                                      columns=['coverage', 'density'])
        
        self.assertTrue(expected_stats.equals(stats))

    
if __name__ == '__main__':
        unittest.main()

#     suite = unittest.TestSuite()
#     suite.addTest(PrimTestCase("test_write_boxes_to_stdout"))
#     unittest.TextTestRunner().run(suite)