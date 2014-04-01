'''
Created on Mar 13, 2012

@author: jhkwakkel
'''
from __future__ import division
import unittest

import numpy as np
import numpy.lib.recfunctions as recfunctions

import matplotlib.pyplot as plt

from expWorkbench import ema_logging
from analysis import prim

from test import util
from expWorkbench.util import load_results
from analysis.prim import PrimBox


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

class PrimBoxTestCase(unittest.TestCase):
    def test_init(self):
        x = np.array([(0,1,2),
                      (2,5,6),
                      (3,2,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float),
                            ('c', np.float)])
        y = {'y':np.array([0,1,2])}
        results = (x,y)
        
        prim_obj = prim.Prim(results, 'y', threshold=0.8)
        box = PrimBox(prim_obj, prim_obj.box_init, prim_obj.yi)

        self.assertTrue(box.peeling_trajectory.shape==(1,5))
    
    def test_select(self):
        x = np.array([(0,1,2),
                      (2,5,6),
                      (3,2,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float),
                            ('c', np.float)])
        y = {'y':np.array([1,1,0])}
        results = (x,y)
        
        prim_obj = prim.Prim(results, 'y', threshold=0.8)
        box = PrimBox(prim_obj, prim_obj.box_init, prim_obj.yi)

        new_box_lim = np.array([(0,1,1),
                                (2,5,6)], 
                                dtype=[('a', np.float),
                                        ('b', np.float),
                                        ('c', np.float)])
        indices = np.array([0,1], dtype=np.int)
        box.update(new_box_lim, indices)
        
        box.select(0)
        self.assertTrue(np.all(box.yi==prim_obj.yi))
    
    def test_inspect(self):
        x = np.array([(0,1,2),
                      (2,5,6),
                      (3,2,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float),
                            ('c', np.float)])
        y = {'y':np.array([1,1,0])}
        results = (x,y)
        
        prim_obj = prim.Prim(results, 'y', threshold=0.8)
        box = PrimBox(prim_obj, prim_obj.box_init, prim_obj.yi)

        new_box_lim = np.array([(0,1,1),
                                (2,5,6)], 
                                dtype=[('a', np.float),
                                        ('b', np.float),
                                        ('c', np.float)])
        indices = np.array([0,1], dtype=np.int)
        box.update(new_box_lim, indices)
        
        box.inspect(1)
    
    def test_update(self):
        x = np.array([(0,1,2),
                      (2,5,6),
                      (3,2,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float),
                            ('c', np.float)])
        y = {'y':np.array([1,1,0])}
        results = (x,y)
        
        prim_obj = prim.Prim(results, 'y', threshold=0.8)
        box = PrimBox(prim_obj, prim_obj.box_init, prim_obj.yi)

        new_box_lim = np.array([(0,1,1),
                                (2,5,6)], 
                                dtype=[('a', np.float),
                                        ('b', np.float),
                                        ('c', np.float)])
        indices = np.array([0,1], dtype=np.int)
        box.update(new_box_lim, indices)
        
        self.assertEqual(box.peeling_trajectory['mean'][1], 1)
        self.assertEqual(box.peeling_trajectory['coverage'][1], 1)
        self.assertEqual(box.peeling_trajectory['density'][1], 1)
        self.assertEqual(box.peeling_trajectory['res dim'][1], 1)
        self.assertEqual(box.peeling_trajectory['mass'][1], 2/3)
        
    
    def test_drop_restriction(self):
        x = np.array([(0,1,2),
                      (2,5,6),
                      (3,2,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float),
                            ('c', np.float)])
        y = {'y':np.array([1,1,0])}
        results = (x,y)
        
        prim_obj = prim.Prim(results, 'y', threshold=0.8)
        box = PrimBox(prim_obj, prim_obj.box_init, prim_obj.yi)

        new_box_lim = np.array([(0,1,1),
                                (2,2,6)], 
                                dtype=[('a', np.float),
                                        ('b', np.float),
                                        ('c', np.float)])
        indices = np.array([0,1], dtype=np.int)
        box.update(new_box_lim, indices)
        
        box.drop_restriction('b')
        
        correct_box_lims = np.array([(0,1,1),
                                     (2,5,6)], 
                                    dtype=[('a', np.float),
                                           ('b', np.float),
                                           ('c', np.float)])        
        box_lims = box.box_lims[-1]
        names = recfunctions.get_names(correct_box_lims.dtype)
        for entry in names:
            lim_correct = correct_box_lims[entry]
            lim_box = box_lims[entry]
            for i in range(len(lim_correct)):
                self.assertEqual(lim_correct[i], lim_box[i])
        
        self.assertEqual(box.peeling_trajectory['mean'][2], 1)
        self.assertEqual(box.peeling_trajectory['coverage'][2], 1)
        self.assertEqual(box.peeling_trajectory['density'][2], 1)
        self.assertEqual(box.peeling_trajectory['res dim'][2], 1)
        self.assertEqual(box.peeling_trajectory['mass'][2], 2/3)        

    
    def test_calculate_quasi_p(self):
        pass

class PrimTestCase(unittest.TestCase):
    def test_prim_init(self):
        self.results = util.load_flu_data()
        self.classify = flu_classify        
        
        experiments, outcomes = self.results
        
        # test initialization, including t_coi calculation in case of searching
        # for results equal to or higher than the threshold
        outcomes['death toll'] = outcomes['deceased population region 1'][:, -1]
        results = experiments, outcomes
        threshold = 10000
        prim_obj = prim.Prim(results, classify='death toll', 
                             threshold_type=prim.ABOVE, threshold=threshold)
        
        value = np.ones((experiments.shape[0],))
        value = value[outcomes['death toll'] >= threshold].shape[0]
        self.assertTrue(prim_obj.t_coi==value)
                
        # test initialization, including t_coi calculation in case of searching
        # for results equal to or lower  than the threshold
        threshold = 1000
        prim_obj = prim.Prim(results, classify='death toll', 
                             threshold_type=prim.BELOW, 
                             threshold=threshold)
        
        value = np.ones((experiments.shape[0],))
        value = value[outcomes['death toll'] <= threshold].shape[0]
        self.assertTrue(prim_obj.t_coi==value)
        
        prim.Prim(self.results, self.classify, threshold=prim.ABOVE)

    def test_prim_init_select(self):
        self.results = util.load_flu_data()
        self.classify = flu_classify        
        
        experiments, outcomes = self.results
        
        unc = recfunctions.get_names(experiments.dtype)
        
        # test initialization, including t_coi calculation in case of searching
        # for results equal to or higher than the threshold
        outcomes['death toll'] = outcomes['deceased population region 1'][:, -1]
        results = experiments, outcomes
        threshold = 10000
        prim_obj = prim.Prim(results, classify='death toll', 
                             threshold_type=prim.ABOVE, threshold=threshold,
                             incl_unc=unc)
        
        value = np.ones((experiments.shape[0],))
        value = value[outcomes['death toll'] >= threshold].shape[0]
        self.assertTrue(prim_obj.t_coi==value)
                
        # test initialization, including t_coi calculation in case of searching
        # for results equal to or lower  than the threshold
        threshold = 1000
        prim_obj = prim.Prim(results, classify='death toll', 
                             threshold_type=prim.BELOW, 
                             threshold=threshold)
        
        value = np.ones((experiments.shape[0],))
        value = value[outcomes['death toll'] <= threshold].shape[0]
        self.assertTrue(prim_obj.t_coi==value)
        
        prim.Prim(self.results, self.classify, threshold=prim.ABOVE)

    def test_quantile(self):
        data = np.ma.array([x for x in range(10)])
        self.assertTrue(prim.get_quantile(data, 0.9)==8.5)
        self.assertTrue(prim.get_quantile(data, 0.95)==8.5)
        self.assertTrue(prim.get_quantile(data, 0.1)==0.5)
        self.assertTrue(prim.get_quantile(data, 0.05)==0.5)
        
        data = np.ma.array(data = [1])
        self.assertTrue(prim.get_quantile(data, 0.9)==1)
        self.assertTrue(prim.get_quantile(data, 0.95)==1)
        self.assertTrue(prim.get_quantile(data, 0.1)==1)
        self.assertTrue(prim.get_quantile(data, 0.05)==1)
        
        data = np.ma.array([1,1,2,3,4,5,6,7,8,9,9])
        self.assertTrue(prim.get_quantile(data, 0.9)==8.5)
        self.assertTrue(prim.get_quantile(data, 0.95)==8.5)
        self.assertTrue(prim.get_quantile(data, 0.1)==1.5)
        self.assertTrue(prim.get_quantile(data, 0.05)==1.5)        
        
        data = np.ma.array([1,1,2,3,4,5,6,7,8,9,9, np.NAN], 
                           mask=[0,0,0,0,0,0,0,0,0,0,0,1])
        self.assertTrue(prim.get_quantile(data, 0.9)==8.5)
        self.assertTrue(prim.get_quantile(data, 0.95)==8.5)
        self.assertTrue(prim.get_quantile(data, 0.1)==1.5)
        self.assertTrue(prim.get_quantile(data, 0.05)==1.5)   
        

    def test_make_box(self):
        x = np.array([(0,1,2),
                      (2,5,6),
                      (3,2,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float),
                            ('c', np.float)])
        y = {'y':np.array([0,1,2])}
        
        prim_obj = prim.Prim((x,y), 'y', threshold=0.5)
        box_lims = prim_obj.make_box(x)
        box = prim.PrimBox(prim_obj, box_lims, [0,1,2])
        
        # some test on the box
        self.assertTrue(box.res_dim==0)
        self.assertTrue(box.mass==1)
        self.assertTrue(box.coverage==1)
        self.assertTrue(box.density==2/3)
        
    
    def test_box_init(self):
        # test init box without NANS
        x = np.array([(0,1,2),
                      (2,5,6),
                      (3,2,7)], 
                     dtype=[('a', np.float),
                            ('b', np.float),
                            ('c', np.float)])
        y = {'y':np.array([0,1,2])}
        
        prim_obj = prim.Prim((x,y), 'y', threshold=0.5)
        box_init = prim_obj.box_init
        
        # some test on the box
        self.assertTrue(box_init['a'][0]==0)
        self.assertTrue(box_init['a'][1]==3)
        self.assertTrue(box_init['b'][0]==1)
        self.assertTrue(box_init['b'][1]==5)
        self.assertTrue(box_init['c'][0]==2)
        self.assertTrue(box_init['c'][1]==7)  
 
        # test init box with NANS
        x = np.array([(0,1,2),
                      (2,5,np.NAN),
                      (3,2,7)], 
                     dtype=[('a', np.float),
                            ('b', np.float),
                            ('c', np.float)])
        y = {'y':np.array([0,1,2])}
        
        x = np.ma.array(x)
        x['a'] = np.ma.masked_invalid(x['a'])
        x['b'] = np.ma.masked_invalid(x['b'])
        x['c'] = np.ma.masked_invalid(x['c'])
         
        prim_obj = prim.Prim((x,y), 'y', threshold=0.5)
        box_init = prim_obj.box_init
         
        # some test on the box
        self.assertTrue(box_init['a'][0]==0)
        self.assertTrue(box_init['a'][1]==3)
        self.assertTrue(box_init['b'][0]==1)
        self.assertTrue(box_init['b'][1]==5)
        self.assertTrue(box_init['c'][0]==2)
        self.assertTrue(box_init['c'][1]==7)  
        
        # heterogenous without NAN
        dtype = [('a', np.float),('b', np.int), ('c', np.object)]
        x = np.empty((10, ), dtype=dtype)
        
        x['a'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.7, 0.8, 0.9, 1.0]
        x['b'] = [0,1,2,3,4,5,6,7,8,9]
        x['c'] = ['a','b','a','b','a','a','b','a','b','a', ]
        
        prim_obj = prim.Prim((x,y), 'y', threshold=0.5)
        box_init = prim_obj.box_init
         
        # some test on the box
        self.assertTrue(box_init['a'][0]==0.1)
        self.assertTrue(box_init['a'][1]==1.0)
        self.assertTrue(box_init['b'][0]==0)
        self.assertTrue(box_init['b'][1]==9)
        self.assertTrue(box_init['c'][0]==set(['a','b']))
        self.assertTrue(box_init['c'][1]==set(['a','b'])) 
 
        # heterogenous with NAN
        dtype = [('a', np.float),('b', np.int), ('c', np.object)]
        x = np.empty((10, ), dtype=dtype)
        
        x[:] = np.NAN
        x['a'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.7, 0.8, np.NAN, 1.0]
        x['b'] = [0,1,2,3,4,5,6,7,8,9]
        x['c'] = ['a','b','a','b',np.NAN,'a','b','a','b','a', ]
        
        
        x = np.ma.array(x)
        x['a'] = np.ma.masked_invalid(x['a'])
        x['b'] = np.ma.masked_invalid(x['b'])
        x['c'][4] = np.ma.masked
        
        prim_obj = prim.Prim((x,y), 'y', threshold=0.5)
        box_init = prim_obj.box_init
         
        # some test on the box
        self.assertTrue(box_init['a'][0]==0.1)
        self.assertTrue(box_init['a'][1]==1.0)
        self.assertTrue(box_init['b'][0]==0)
        self.assertTrue(box_init['b'][1]==9)
        self.assertTrue(box_init['c'][0]==set(['a','b']))
        self.assertTrue(box_init['c'][1]==set(['a','b'])) 
  
    def test_restricted_dimension(self):
        x = np.random.rand(10, )
        x = np.asarray(x, dtype=[('a', np.float),
                                 ('b', np.float)])
        y = {'y': np.random.randint(0,2, (10,)).astype(int)}
        
        prim_obj = prim.Prim((x,y), 'y', threshold=0.8)
        
        # all dimensions the same
        b = prim_obj.box_init
        u = prim_obj.determine_restricted_dims(b)
        
        self.assertEqual(len(u), 0)
        
       
        # dimensions 1 different and dimension 2 the same
        b = np.array([(1,1),
                      (0,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float)])
        u = prim_obj.determine_restricted_dims(b)
        
        self.assertEqual(len(u), 2)

    def test_compare(self):
        self.results = util.load_scarcity_data()
        self.classify = scarcity_classify
        
        prim_obj = prim.Prim(self.results, self.classify, threshold=0.8)
        
        # all dimensions the same
        a = np.array([(0,1),
                      (0,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float)])
        b = np.array([(0,1),
                      (0,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float)])
        
        self.assertTrue(np.all(prim_obj.compare(a,b)))
        
        # all dimensions different
        a = np.array([(0,1),
                      (0,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float)])
        b = np.array([(1,1),
                      (0,0)], 
                     dtype=[('a', np.float),
                            ('b', np.float)])
        test = prim_obj.compare(a,b)==False
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
        test = prim_obj.compare(a,b)
        test = (test[0]==False) & (test[1]==True)
        self.assertTrue(test)

    def test_in_box(self):
        dtype = [('a', np.int)]
        x = np.array([(0,),
                      (1,),
                      (2,),
                      (3,),
                      (4,),
                      (5,),
                      (6,),
                      (7,),
                      (8,),
                      (9,)], 
                     dtype=dtype)
        boxlim = np.array([(1,),
                           (8,)], dtype=dtype)
        correct_result = np.array([1,2,3,4,5,6,7,8], dtype=np.int)
        result = prim._in_box(x, boxlim)
        
        self.assertTrue(np.all(correct_result==result))


        dtype = [('a', np.int),
                 ('b', np.int)]
        x = np.array([(0,0),
                      (1,1),
                      (2,2),
                      (3,3),
                      (4,4),
                      (5,5),
                      (6,6),
                      (7,7),
                      (8,8),
                      (9,9)], 
                     dtype=dtype)
        boxlim = np.array([(1,0),
                           (8,7)], dtype=dtype)
        correct_result = np.array([1,2,3,4,5,6,7], dtype=np.int)
        result = prim._in_box(x, boxlim)
        
        self.assertTrue(np.all(correct_result==result))
    
        dtype = [('a', np.float),
                 ('b', np.int),
                 ('c', np.object)]
        x = np.array([(0.1, 0, 'a'),
                      (1.1, 1, 'a'),
                      (2.1, 2, 'b'),
                      (3.1, 3, 'b'),
                      (4.1, 4, 'c'),
                      (5.1, 5, 'c'),
                      (6.1, 6, 'd'),
                      (7.1, 7, 'd'),
                      (8.1, 8, 'e'),
                      (9.1, 9, 'e')], 
                     dtype=dtype)
        boxlim = np.array([(1.2,0, set(['a','b'])),
                           (8.0,7, set(['a','b']) )], dtype=dtype)
        correct_result = np.array([2,3], dtype=np.int)
        result = prim._in_box(x, boxlim)
        self.assertTrue(np.all(correct_result==result))
        
        boxlim = np.array([(0.1, 0, set(['a','b','c','d','e'])),
                           (9.1, 9, set(['a','b','c','d','e']) )], dtype=dtype)
        correct_result = np.array([0,1,2,3,4,5,6,7,8,9], dtype=np.int)
        result = prim._in_box(x, boxlim)
        self.assertTrue(np.all(correct_result==result))
    
    def test_prim_init_exception(self):
        results = util.load_flu_data()
        self.assertRaises(prim.PrimException, 
                          prim.Prim,
                          results, 
                          'deceased population region 1', 
                          threshold=0.8)
        
        def faulty_classify(outcomes):
            return outcomes['deceased population region 1'][:, 0:10]
        self.assertRaises(prim.PrimException, prim.Prim, results, 
                          faulty_classify, threshold=0.8)

#    def test_write_boxes_to_stdout(self):
#        results = load_results(r'../data/1000 flu cases no policy.bz2')
#        classify = flu_classify
#
##        results = load_results(r'../data/scarcity 1000.bz2')
##        classify = scarcity_classify
#                
#        prim = prim.Prim(results, classify, 
#                             threshold=0.7)
#        prim.find_box()
#        prim.find_box()
#        
#        print "\n"
#        prim.write_boxes_to_stdout()   

    def test_show_boxes(self):
#        results = load_results(r'../data/1000 flu cases no policy.bz2')
#        classify = flu_classify
                
        results = util.load_flu_data()
        classify = flu_classify
                
        prim_obj = prim.Prim(results, classify, 
                             threshold=0.7)
        box1 = prim_obj.find_box()
        box2 = prim_obj.find_box()
        
        prim_obj.write_boxes_to_stdout()
        
        prim_obj.show_boxes()   
#         plt.show()
  
    def test_find_boxes(self):
        results = util.load_flu_data()
        classify = flu_classify
        
        
        prim_obj = prim.Prim(results, classify, 
                             threshold=0.8)
        box_1 = prim_obj.find_box()
        prim_obj._update_yi_remaining()
        
        after_find = box_1.yi.shape[0] + prim_obj.yi_remaining.shape[0]
        self.assertEqual(after_find, prim_obj.y.shape[0])
        
        box_2 = prim_obj.find_box()
        prim_obj._update_yi_remaining()
        
        after_find = box_1.yi.shape[0] +\
                     box_2.yi.shape[0] +\
                     prim_obj.yi_remaining.shape[0]
        self.assertEqual(after_find, prim_obj.y.shape[0])
        
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
        
        prim_obj  = prim.Prim(results, classify, threshold=0.8)
        box_lims = np.array([(0, set(['a','b'])),
                        (1, set(['a','b']))], dtype=dtype )
        box = prim.PrimBox(prim_obj, box_lims, prim_obj.yi)
        
        u = 'b'
        x = x
        peels = prim_obj._categorical_peel(box, u, x)
        
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
            
            prim_obj  = prim.Prim(results, classify, threshold=0.8)
            box_lims = np.array([(0, set(['a',])),
                            (1, set(['a',]))], dtype=dtype )
            
            yi = np.where(x['b']=='a')
            
            box = prim.PrimBox(prim_obj, box_lims, yi)
            
            u = 'b'
            pastes = prim_obj._categorical_paste(box, u)
            
            self.assertEquals(len(pastes), 1)
            
            for paste in pastes:
                indices, box_lims = paste
                
                self.assertEquals(indices.shape[0], 10)
                self.assertEqual(box_lims[u][0], set(['a','b']))

if __name__ == '__main__':
#     ema_logging.log_to_stderr(ema_logging.INFO)    

#    unittest.main()

    suite = unittest.TestSuite()
    suite.addTest(PrimBoxTestCase("test_inspect"))
    unittest.TextTestRunner().run(suite)