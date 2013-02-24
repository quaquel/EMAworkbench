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
    def setUp(self):
        self.results = load_results(r'../data/scarcity 1000.bz2')
        self.classify = scarcity_classify

#    def test_prim_init(self):
#        experiments, outcomes = self.results
#        
#        # test initialization, including t_coi calculation in case of searching
#        # for results equal to or higher than the threshold
#        outcomes['death toll'] = outcomes['deceased population region 1'][:, -1]
#        results = experiments, outcomes
#        threshold = 10000
#        prim = new_prim.Prim(results, classify='death toll', 
#                             threshold_type=new_prim.ABOVE, threshold=threshold)
#        
#        value = np.ones((experiments.shape[0],))
#        value = value[outcomes['death toll'] >= threshold].shape[0]
#        self.assertTrue(prim.t_coi==value)
#                
#        # test initialization, including t_coi calculation in case of searching
#        # for results equal to or lower  than the threshold
#        threshold = 1000
#        prim = new_prim.Prim(results, classify='death toll', 
#                             threshold_type=new_prim.BELOW, 
#                             threshold=threshold)
#        
#        value = np.ones((experiments.shape[0],))
#        value = value[outcomes['death toll'] <= threshold].shape[0]
#        self.assertTrue(prim.t_coi==value)
#        
#        new_prim.Prim(self.results, self.classify, threshold=new_prim.ABOVE)

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

    def test_compare(self):
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
        
    def test_find_boxes(self):
        prim = new_prim.Prim(self.results, self.classify, 
                             threshold=0.8)
        box = prim.find_box()
        
        after_find = box.yi.shape[0] + prim.yi_remaining.shape[0]
        self.assertEqual(after_find, prim.y.shape[0])
        
#        box.write_ppt_stdout()
#        box.show_ppt()
#        plt.show()
#       try and perform a peel and then check if the indices of the box and
#       the yi_remaining in prim combined reproduce the data

        
def test_prim():
    results = load_results(r'../data/1000 flu cases no policy.bz2')
    from analysis import prim
    
    #perform prim on modified results tuple
    boxes = prim.perform_prim(results, flu_classify, threshold=0.8, 
                              threshold_type=1, pasting=True)

#
#def test_write_to_std_out(results):
##    boxes = test_prim(results)
##    prim.write_prim_to_stdout(boxes, results[0])
#    pass
if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)    
#    test_prim()
    unittest.main()