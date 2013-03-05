'''
Created on Mar 13, 2012

@author: jhkwakkel
'''
from __future__ import division
import unittest

import numpy as np
import matplotlib.pyplot as plt

from expWorkbench import ema_logging, load_results
from analysis import new_prim


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


class PrimTestCase(unittest.TestCase):
#    def setUp(self):
#        self.results = load_results(r'../data/scarcity 1000.bz2')
#        self.classify = scarcity_classify

    def test_prim_init(self):
        self.results = load_results(r'../data/1000 flu cases no policy.bz2')
        self.classify = flu_classify        
        
        experiments, outcomes = self.results
        
        # test initialization, including t_coi calculation in case of searching
        # for results equal to or higher than the threshold
        outcomes['death toll'] = outcomes['deceased population region 1'][:, -1]
        results = experiments, outcomes
        threshold = 10000
        prim = new_prim.Prim(results, classify='death toll', 
                             threshold_type=new_prim.ABOVE, threshold=threshold)
        
        value = np.ones((experiments.shape[0],))
        value = value[outcomes['death toll'] >= threshold].shape[0]
        self.assertTrue(prim.t_coi==value)
                
        # test initialization, including t_coi calculation in case of searching
        # for results equal to or lower  than the threshold
        threshold = 1000
        prim = new_prim.Prim(results, classify='death toll', 
                             threshold_type=new_prim.BELOW, 
                             threshold=threshold)
        
        value = np.ones((experiments.shape[0],))
        value = value[outcomes['death toll'] <= threshold].shape[0]
        self.assertTrue(prim.t_coi==value)
        
        new_prim.Prim(self.results, self.classify, threshold=new_prim.ABOVE)

    def test_box(self):
        x = np.array([(0,1,2),
                      (2,5,6),
                      (3,2,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float),
                            ('c', np.float)])
        y = {'y':np.array([0,1,2])}
        
        prim = new_prim.Prim((x,y), 'y', threshold=0.5)
        box_lims = prim.make_box(x)
        box = new_prim.PrimBox(prim, box_lims, [0,1,2])
        
        # some test on the box
        self.assertTrue(box.res_dim[0]==0)
        self.assertTrue(box.mass[0]==1)
        self.assertTrue(box.coverage[0]==1)
        self.assertTrue(box.density[0]==2/3)

    def test_restricted_dimension(self):
        x = np.random.rand(10, )
        x = np.asarray(x, dtype=[('a', np.float),
                                 ('b', np.float)])
        y = np.random.randint(0,2, (10,)).astype(int)
        y = {'y': np.random.randint(0,2, (10,)).astype(int)}
        
        prim = new_prim.Prim((x,y), 'y')
        
        # all dimensions the same
        b = prim.box_init
        u = prim.determine_restricted_dims(b)
        
        self.assertEqual(len(u), 0)
        
       
        # dimensions 1 different and dimension 2 the same
        b = np.array([(1,1),
                      (0,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float)])
        u = prim.determine_restricted_dims(b)
        
        self.assertEqual(len(u), 2)

    def test_compare(self):
        self.results = load_results(r'../data/scarcity 1000.bz2')
        self.classify = scarcity_classify
        
        prim = new_prim.Prim(self.results, self.classify)
        
        # all dimensions the same
        a = np.array([(0,1),
                      (0,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float)])
        b = np.array([(0,1),
                      (0,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float)])
        
        self.assertTrue(np.all(prim.compare(a,b)))
        
        # all dimensions different
        a = np.array([(0,1),
                      (0,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float)])
        b = np.array([(1,1),
                      (0,0)], 
                     dtype=[('a', np.float),
                            ('b', np.float)])
        test = prim.compare(a,b)==False
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
        test = prim.compare(a,b)
        test = (test[0]==False) & (test[1]==True)
        self.assertTrue(test)

    def test_in_box(self):
        results = load_results(r'../data/1000 flu cases no policy.bz2')
        prim = new_prim.Prim(results, flu_classify)
        
        box = prim.make_box(results[0])
        # I need an encompassing box
        # the shape[0] of the return should be equal to experiment.shape[0]
        # assuming that the box is an encompassing box
        self.assertEqual(prim.in_box(box).shape[0], results[0].shape[0])
    
    def test_prim_init_exception(self):
        results = load_results(r'../data/1000 flu cases no policy.bz2')
        self.assertRaises(new_prim.PrimException, 
                          new_prim.Prim,
                          results, 
                          'deceased population region 1')
        
        def faulty_classify(outcomes):
            return outcomes['deceased population region 1'][:, 0:10]
        self.assertRaises(new_prim.PrimException, new_prim.Prim, results, 
                          faulty_classify)

    def test_write_boxes_to_stdout(self):
        results = load_results(r'../data/1000 flu cases no policy.bz2')
        classify = flu_classify
        
        prim = new_prim.Prim(results, classify, 
                             threshold=0.8)
        box_1 = prim.find_box()
        prim.write_boxes_to_stdout()   
        

#    def test_write_boxes_to_stdout(self):
#        dtype = [('a', np.float),('b', np.object)]
#        box_lims1 = np.array([(0, set(['a','b'])),
#                        (1, set(['a','b']))], dtype=dtype )
#        box_lims2 = np.array([(0, set(['a',])),
#                                (1, set(['a',]))], dtype=dtype )
#        
#        new_prim._write_boxes_to_stdout([box_lims1,
#                                         box_lims2], ['b','a'])


    def test_select(self):
        results = load_results(r'../data/1000 flu cases no policy.bz2')
        classify = flu_classify
        
        prim = new_prim.Prim(results, classify, 
                             threshold=0.8)
        box = prim.find_box()
        sb = 27
        box.select(sb)
        
        self.assertEqual(len(box.mean), sb+1)

    
    def test_find_boxes(self):
        results = load_results(r'../data/1000 flu cases no policy.bz2')
        classify = flu_classify
        
        
        prim = new_prim.Prim(results, classify, 
                             threshold=0.8)
        box_1 = prim.find_box()
        prim._update_yi_remaining()
        
        after_find = box_1.yi.shape[0] + prim.yi_remaining.shape[0]
        self.assertEqual(after_find, prim.y.shape[0])
        
        box_2 = prim.find_box()
        prim._update_yi_remaining()
        
        after_find = box_1.yi.shape[0] +\
                     box_2.yi.shape[0] +\
                     prim.yi_remaining.shape[0]
        self.assertEqual(after_find, prim.y.shape[0])
        
#        box_1.write_ppt_stdout()
#        box_1.show_ppt()
#        plt.show()
#       try and perform a peel and then check if the indices of the box and
#       the yi_remaining in prim combined reproduce the data

        
    def test_categorical_peel(self):
        dtype = [('a', np.float),('b', np.object)]
        x = np.empty((10, ), dtype=dtype)
        
        x['a'] = np.random.rand(10,)
        x['b'] = ['a','b','a','b','a','a','b','a','b','a', ]
        y = np.random.randint(0,2, (10,))
        y = y.astype(np.int)
        y = {'y':y}
        results = x,y
        classify = 'y'
        
        prim  = new_prim.Prim(results, classify, threshold=0.8)
        box_lims = np.array([(0, set(['a','b'])),
                        (1, set(['a','b']))], dtype=dtype )
        box = new_prim.PrimBox(prim, box_lims, prim.yi)
        
        u = 'b'
        x = x
        peels = prim._categorical_peel(box, u, x)
        
        self.assertEquals(len(peels), 2)
        
        for peel in peels:
            pl  = peel[1][u]
            self.assertEquals(len(pl[0]), 1)
            self.assertEquals(len(pl[1]), 1)
        

    def test_categorical_paste(self):
            dtype = [('a', np.float),('b', np.object)]
            x = np.empty((10, ), dtype=dtype)
            
            x['a'] = np.random.rand(10,)
            x['b'] = ['a','b','a','b','a','a','b','a','b','a', ]
            y = np.random.randint(0,2, (10,))
            y = y.astype(np.int)
            y = {'y':y}
            results = x,y
            classify = 'y'
            
            prim  = new_prim.Prim(results, classify, threshold=0.8)
            box_lims = np.array([(0, set(['a',])),
                            (1, set(['a',]))], dtype=dtype )
            
            yi = np.where(x['b']=='a')
            
            box = new_prim.PrimBox(prim, box_lims, yi)
            
            u = 'b'
            pastes = prim._categorical_paste(box, u)
            
            self.assertEquals(len(pastes), 1)
            
            for paste in pastes:
                indices, box_lims = paste
                
                self.assertEquals(indices.shape[0], 10)
                self.assertEqual(box_lims[u][0], set(['a','b']))
                
        

        
def test_prim():
    results = load_results(r'../data/1000 flu cases no policy.bz2')
    from analysis import prim
    
    #perform prim on modified results tuple
    boxes = prim.perform_prim(results, flu_classify, threshold=0.8, 
                              threshold_type=1, pasting=True)


if __name__ == '__main__':
#    ema_logging.log_to_stderr(ema_logging.INFO)    
    unittest.main()