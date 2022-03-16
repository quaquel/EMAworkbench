'''
Created on Mar 13, 2012

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
import unittest

import numpy as np
import pandas as pd

from ema_workbench.analysis import prim
from ema_workbench.analysis.prim import PrimBox
from test import utilities
from ema_workbench.analysis.scenario_discovery_util import RuleInductionType
from ema_workbench.em_framework.outcomes import ScalarOutcome



def flu_classify(data):
    # get the output for deceased population
    result = data['deceased population region 1']
    
    # make an empty array of length equal to number of cases
    classes = np.zeros(result.shape[0])
    
    # if deceased population is higher then 1.000.000 people, classify as 1
    classes[result[:, -1] > 1000000] = 1
    
    return classes


class PrimBoxTestCase(unittest.TestCase):
    def test_init(self):
        x = pd.DataFrame([(0, 1, 2),
                          (2, 5, 6),
                          (3, 2, 1)],
                          columns=['a', 'b', 'c'])
        y = {'y':np.array([0,1,2])}
        results = (x,y)
        
        prim_obj = prim.setup_prim(results, 'y', threshold=0.8)
        box = PrimBox(prim_obj, prim_obj.box_init, prim_obj.yi)

        self.assertEqual(box.peeling_trajectory.shape, (1,6))
    
    def test_select(self):
        x = pd.DataFrame([(0, 1, 2),
                          (2, 5, 6),
                          (3, 2, 1)],
                          columns=['a', 'b', 'c'])
        y = {'y':np.array([1,1,0])}
        results = (x,y)
        
        prim_obj = prim.setup_prim(results, 'y', threshold=0.8)
        box = PrimBox(prim_obj, prim_obj.box_init, prim_obj.yi)

        new_box_lim = pd.DataFrame([(0, 1, 1),
                                    (2, 5, 6)],
                                    columns=['a', 'b', 'c'])
        indices = np.array([0,1], dtype=int)
        box.update(new_box_lim, indices)
        
        box.select(0)
        self.assertTrue(np.all(box.yi==prim_obj.yi))
    
    def test_inspect(self):
        x = pd.DataFrame([(0, 1, 2),
                          (2, 5, 6),
                          (3, 2, 1)],
                          columns=['a', 'b', 'c'])
        y = np.array([1,1,0])
        
        prim_obj = prim.Prim(x, y, threshold=0.8)
        box = PrimBox(prim_obj, prim_obj.box_init, prim_obj.yi)

        new_box_lim = pd.DataFrame([(0, 1, 1),
                                    (2, 5, 6)],
                                    columns=['a', 'b', 'c'])
        indices = np.array([0, 1], dtype=int)
        box.update(new_box_lim, indices)
        
        box.inspect(1)
        box.inspect()
        box.inspect(style='graph')
        
        with self.assertRaises(ValueError):
            box.inspect(style='some unknown style')
    
    def test_show_ppt(self):
        x = pd.DataFrame([(0,1,2),
                          (2,5,6),
                          (3,2,1)], 
                          columns=['a', 'b', 'c'])
        y = np.array([1,1,0])
        
        prim_obj = prim.Prim(x, y, threshold=0.8)
        box = PrimBox(prim_obj, prim_obj.box_init, prim_obj.yi)
        
        
        cols = ['mean', 'mass', 'coverage', 'density', 'res_dim']
        data = np.zeros((100, 5))
        data[:, 0:4] = np.random.rand(100, 4)
        data[:, 4] = np.random.randint(0, 5, size=(100, ))
        box.peeling_trajectory = pd.DataFrame(data, columns=cols)
        
        box.show_ppt()
        
    def test_show_tradeoff(self):    
        x = pd.DataFrame([(0,1,2),
                          (2,5,6),
                          (3,2,1)], 
                          columns=['a', 'b', 'c'])
        y = np.array([1,1,0])
        
        prim_obj = prim.Prim(x, y, threshold=0.8)
        box = PrimBox(prim_obj, prim_obj.box_init, prim_obj.yi)
        
        
        cols = ['mean', 'mass', 'coverage', 'density', 'res_dim']
        data = np.zeros((100, 5))
        data[:, 0:4] = np.random.rand(100, 4)
        data[:, 4] = np.random.randint(0, 5, size=(100, ))
        box.peeling_trajectory = pd.DataFrame(data, columns=cols)
        
        box.show_tradeoff()    
    
    def test_update(self):
        x = pd.DataFrame([(0, 1, 2),
                          (2, 5, 6),
                          (3, 2, 1)],
                          columns=['a', 'b', 'c'])
        y = {'y':np.array([1, 1, 0])}
        results = (x, y)
        
        prim_obj = prim.setup_prim(results, 'y', threshold=0.8)
        box = PrimBox(prim_obj, prim_obj.box_init, prim_obj.yi)

        new_box_lim = pd.DataFrame([(0, 1, 1),
                                    (2, 5, 6)],
                                    columns=['a', 'b', 'c'])
        indices = np.array([0, 1], dtype=int)
        box.update(new_box_lim, indices)
        
        self.assertEqual(box.peeling_trajectory['mean'][1], 1)
        self.assertEqual(box.peeling_trajectory['coverage'][1], 1)
        self.assertEqual(box.peeling_trajectory['density'][1], 1)
        self.assertEqual(box.peeling_trajectory['res_dim'][1], 1)
        self.assertEqual(box.peeling_trajectory['mass'][1], 2/3)
    
    def test_drop_restriction(self):
        x = pd.DataFrame([(0, 1, 2),
                          (2, 5, 6),
                          (3, 2, 1)],
                          columns=['a', 'b', 'c'])
        y = {'y':np.array([1,1,0])}
        results = (x,y)
        
        prim_obj = prim.setup_prim(results, 'y', threshold=0.8)
        box = PrimBox(prim_obj, prim_obj.box_init, prim_obj.yi)

        new_box_lim = pd.DataFrame([(0, 1, 1),
                                    (2, 2, 6)],
                                    columns=['a', 'b', 'c'])
        indices = np.array([0,1], dtype=int)
        box.update(new_box_lim, indices)
        
        box.drop_restriction('b')
        
        correct_box_lims = pd.DataFrame([(0, 1, 1),
                                         (2, 5, 6)],
                                         columns=['a', 'b', 'c'])        
        box_lims = box.box_lims[-1]
        names = box_lims.columns
        for entry in names:
            lim_correct = correct_box_lims[entry]
            lim_box = box_lims[entry]
            for i in range(len(lim_correct)):
                self.assertEqual(lim_correct[i], lim_box[i])
        
        self.assertEqual(box.peeling_trajectory['mean'][2], 1)
        self.assertEqual(box.peeling_trajectory['coverage'][2], 1)
        self.assertEqual(box.peeling_trajectory['density'][2], 1)
        self.assertEqual(box.peeling_trajectory['res_dim'][2], 1)
        self.assertEqual(box.peeling_trajectory['mass'][2], 2/3)        

    
    def test_calculate_quasi_p(self):
        pass

class PrimTestCase(unittest.TestCase):

    def test_setup_prim(self):
        self.results = utilities.load_flu_data()
        self.classify = flu_classify        
        
        experiments, outcomes = self.results
        
        # test initialization, including t_coi calculation in case of searching
        # for results equal to or higher than the threshold
        outcomes['death toll'] = outcomes['deceased population region 1'][:, -1]
        results = experiments, outcomes
        threshold = 10000
        prim_obj = prim.setup_prim(results, classify='death toll', 
                             threshold_type=prim.ABOVE, threshold=threshold)
        
        value = np.ones((experiments.shape[0],))
        value = value[outcomes['death toll'] >= threshold].shape[0]
        self.assertTrue(prim_obj.t_coi==value)
                
        # test initialization, including t_coi calculation in case of searching
        # for results equal to or lower  than the threshold
        threshold = 1000
        prim_obj = prim.setup_prim(results, classify='death toll', 
                             threshold_type=prim.BELOW, 
                             threshold=threshold)
        
        value = np.ones((experiments.shape[0],))
        value = value[outcomes['death toll'] <= threshold].shape[0]
        self.assertTrue(prim_obj.t_coi==value)
        
        prim.setup_prim(self.results, self.classify, threshold=prim.ABOVE)
    
    def test_boxes(self):
        x = pd.DataFrame([(0,1,2),
                          (2,5,6),
                          (3,2,1)], 
                         columns=['a', 'b', 'c'])
        y = {'y':np.array([0,1,2])}
        results = (x,y)
        
        prim_obj = prim.setup_prim(results, 'y', threshold=0.8)
        boxes = prim_obj.boxes
        
        self.assertEqual(len(boxes), 1, 'box length not correct')
        
        # real data test case        
        prim_obj = prim.setup_prim(utilities.load_flu_data(), flu_classify,
                                   threshold=0.8)
        prim_obj.find_box()
        boxes = prim_obj.boxes
        self.assertEqual(len(boxes), 1, 'box length not correct')  

    def test_prim_init_select(self):
        self.results = utilities.load_flu_data()
        self.classify = flu_classify        
        
        experiments, outcomes = self.results
        
        unc = experiments.columns.values.tolist()
        
        # test initialization, including t_coi calculation in case of searching
        # for results equal to or higher than the threshold
        outcomes['death toll'] = outcomes['deceased population region 1'][:, -1]
        results = experiments, outcomes
        threshold = 10000
        prim_obj = prim.setup_prim(results, classify='death toll', 
                             threshold_type=prim.ABOVE, threshold=threshold,
                             incl_unc=unc)
        
        value = np.ones((experiments.shape[0],))
        value = value[outcomes['death toll'] >= threshold].shape[0]
        self.assertTrue(prim_obj.t_coi==value)
                
        # test initialization, including t_coi calculation in case of searching
        # for results equal to or lower  than the threshold
        threshold = 1000
        prim_obj = prim.setup_prim(results, classify='death toll', 
                             threshold_type=prim.BELOW, 
                             threshold=threshold)
        
        value = np.ones((experiments.shape[0],))
        value = value[outcomes['death toll'] <= threshold].shape[0]
        self.assertTrue(prim_obj.t_coi==value)
        
        prim.setup_prim(self.results, self.classify, threshold=prim.ABOVE)

    def test_quantile(self):
        data = pd.Series(np.arange(10))
        self.assertTrue(prim.get_quantile(data, 0.9)==8.5)
        self.assertTrue(prim.get_quantile(data, 0.95)==8.5)
        self.assertTrue(prim.get_quantile(data, 0.1)==0.5)
        self.assertTrue(prim.get_quantile(data, 0.05)==0.5)
        
        data = pd.Series(1)
        self.assertTrue(prim.get_quantile(data, 0.9)==1)
        self.assertTrue(prim.get_quantile(data, 0.95)==1)
        self.assertTrue(prim.get_quantile(data, 0.1)==1)
        self.assertTrue(prim.get_quantile(data, 0.05)==1)
        
        data = pd.Series([1,1,2,3,4,5,6,7,8,9,9])
        self.assertTrue(prim.get_quantile(data, 0.9)==8.5)
        self.assertTrue(prim.get_quantile(data, 0.95)==8.5)
        self.assertTrue(prim.get_quantile(data, 0.1)==1.5)
        self.assertTrue(prim.get_quantile(data, 0.05)==1.5)        
          
        

    def test_box_init(self):
        # test init box without NANS
        x = pd.DataFrame([(0,1,2),
                          (2,5,6),
                          (3,2,7)], 
                         columns=['a', 'b', 'c'])
        y = np.array([0,1,2])
        
        prim_obj = prim.Prim(x,y, threshold=0.5,
                             mode=RuleInductionType.REGRESSION)
        box_init = prim_obj.box_init
        
        # some test on the box
        self.assertTrue(box_init.loc[0, 'a']==0)
        self.assertTrue(box_init.loc[1, 'a']==3)
        self.assertTrue(box_init.loc[0, 'b']==1)
        self.assertTrue(box_init.loc[1, 'b']==5)
        self.assertTrue(box_init.loc[0, 'c']==2)
        self.assertTrue(box_init.loc[1, 'c']==7)  
 
        # heterogenous without NAN
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
        y = np.arange(0, x.shape[0])

        prim_obj = prim.Prim(x,y, threshold=0.5,
                             mode=RuleInductionType.REGRESSION)
        box_init = prim_obj.box_init
         
        # some test on the box
        self.assertTrue(box_init['a'][0]==0.1)
        self.assertTrue(box_init['a'][1]==1.0)
        self.assertTrue(box_init['b'][0]==0)
        self.assertTrue(box_init['b'][1]==9)
        self.assertTrue(box_init['c'][0]==set(['a','b']))
        self.assertTrue(box_init['c'][1]==set(['a','b'])) 

  
    def test_prim_exceptions(self):
        results = utilities.load_flu_data()
        x, outcomes = results
        y = outcomes['deceased population region 1']
        
        self.assertRaises(prim.PrimException, prim.Prim,
                          x, y, threshold=0.8,
                          mode=RuleInductionType.REGRESSION)

    def test_find_box(self):
        results = utilities.load_flu_data()
        classify = flu_classify
        
        prim_obj = prim.setup_prim(results, classify, 
                                   threshold=0.8)
        box_1 = prim_obj.find_box()
        prim_obj._update_yi_remaining(prim_obj)
        
        after_find = box_1.yi.shape[0] + prim_obj.yi_remaining.shape[0]
        self.assertEqual(after_find, prim_obj.y.shape[0])
        
        box_2 = prim_obj.find_box()
        prim_obj._update_yi_remaining(prim_obj)
        
        after_find = box_1.yi.shape[0] +\
                     box_2.yi.shape[0] +\
                     prim_obj.yi_remaining.shape[0]
        self.assertEqual(after_find, prim_obj.y.shape[0])
        
    def test_discrete_peel(self):
        x = pd.DataFrame(np.random.randint(0, 10, size=(100,), dtype=int),
                         columns=['a'])
        y  = np.zeros(100,)
        y[x.a > 5] = 1
        
        primalg = prim.Prim(x, y, threshold=0.8)
        boxlims = primalg.box_init
        box = prim.PrimBox(primalg, boxlims, primalg.yi)     
        
        peels = primalg._discrete_peel(box, 'a', 0, primalg.x_int)
        
        self.assertEqual(len(peels), 2)
        for peel in peels:
            self.assertEqual(len(peel), 2)
            
            indices, tempbox = peel
            
            self.assertTrue(isinstance(indices, np.ndarray))
            self.assertTrue(isinstance(tempbox, pd.DataFrame))
            
        # have modified boxlims as starting point
        primalg = prim.Prim(x, y, threshold=0.8)
        boxlims = primalg.box_init
        boxlims.a = [1,8]
        box = prim.PrimBox(primalg, boxlims, primalg.yi)     
        
        peels = primalg._discrete_peel(box, 'a', 0, primalg.x_int)
        
        self.assertEqual(len(peels), 2)
        for peel in peels:
            self.assertEqual(len(peel), 2)
            
            indices, tempbox = peel
            
            self.assertTrue(isinstance(indices, np.ndarray))
            self.assertTrue(isinstance(tempbox, pd.DataFrame))
            
        # have modified boxlims as starting point
        x.a[x.a>5] = 5
        primalg = prim.Prim(x, y, threshold=0.8)
        boxlims = primalg.box_init
        boxlims.a = [5,8]
        box = prim.PrimBox(primalg, boxlims, primalg.yi)     
        
        peels = primalg._discrete_peel(box, 'a', 0, primalg.x_int)
        self.assertEqual(len(peels), 2)

        x.a[x.a<5] = 5
        primalg = prim.Prim(x, y, threshold=0.8)
        boxlims = primalg.box_init
        boxlims.a = [5,8]
        box = prim.PrimBox(primalg, boxlims, primalg.yi)     
        
        peels = primalg._discrete_peel(box, 'a', 0, primalg.x_int)
        self.assertEqual(len(peels), 2)

                
    def test_categorical_peel(self):
        x = pd.DataFrame(list(zip(np.random.rand(10,),
                                  ['a','b','a','b','a','a','b','a','b','a', ])),
                          columns=['a', 'b'])
        
        y = np.random.randint(0,2, (10,))
        y = y.astype(int)
        y = {'y':y}
        results = x, y
        classify = 'y'
        
        prim_obj  = prim.setup_prim(results, classify, threshold=0.8)
        box_lims = pd.DataFrame([(0, set(['a','b'])),
                                 (1, set(['a','b']))],
                                 columns=['a', 'b'] )
        box = prim.PrimBox(prim_obj, box_lims, prim_obj.yi)
        
        u = 'b'
        x = x.select_dtypes(exclude=np.number).values
        j = 0
        peels = prim_obj._categorical_peel(box, u, j, x)
        
        self.assertEqual(len(peels), 2)
        
        for peel in peels:
            pl  = peel[1][u]
            self.assertEqual(len(pl[0]), 1)
            self.assertEqual(len(pl[1]), 1)
            
            
        a = ('a',)
        b = ('b',)
        x = pd.DataFrame(list(zip(np.random.rand(10,),
                                  [a, b, a, b, a,
                                   a, b, a, b, a])),
                         columns=['a', 'b'])
        
        y = np.random.randint(0,2, (10,))
        y = y.astype(int)
        y = {'y':y}
        results = x, y
        classify = 'y'
        
        prim_obj  = prim.setup_prim(results, classify, threshold=0.8)
        box_lims = prim_obj.box_init
        box = prim.PrimBox(prim_obj, box_lims, prim_obj.yi)
        
        u = 'b'
        x = x.select_dtypes(exclude=np.number).values
        j = 0
        peels = prim_obj._categorical_peel(box, u, j, x)
        
        self.assertEqual(len(peels), 2)
        
        for peel in peels:
            pl  = peel[1][u]
            self.assertEqual(len(pl[0]), 1)
            self.assertEqual(len(pl[1]), 1)
        

    def test_categorical_paste(self):
        a = np.random.rand(10,)
        b = ['a','b','a','b','a','a','b','a','b','a', ]
        x = pd.DataFrame(list(zip(a,b)), columns=['a', 'b'])
        x['b'] = x['b'].astype('category')
        
        y = np.random.randint(0,2, (10,))
        y = y.astype(int)
        y = {'y':y}
        results = x,y
        classify = 'y'
        
        prim_obj  = prim.setup_prim(results, classify, threshold=0.8)
        box_lims = pd.DataFrame([(0, set(['a',])),
                                 (1, set(['a',]))], columns=x.columns)
        
        yi = np.where(x.loc[:,'b']=='a')
        
        box = prim.PrimBox(prim_obj, box_lims, yi)
        
        u = 'b'
        pastes = prim_obj._categorical_paste(box, u, x, ['b'])
        
        self.assertEqual(len(pastes), 1)
        
        for paste in pastes:
            indices, box_lims = paste
            
            self.assertEqual(indices.shape[0], 10)
            self.assertEqual(box_lims[u][0], set(['a','b']))
            
    def test_constrained_prim(self):
        experiments, outcomes = utilities.load_flu_data()
        y = flu_classify(outcomes)
        
        box = prim.run_constrained_prim(experiments, y, issignificant=True)        
        

if __name__ == '__main__':
#     ema_logging.log_to_stderr(ema_logging.INFO)    

    unittest.main()

#     suite = unittest.TestSuite()
#     suite.addTest(PrimTestCase("test_write_boxes_to_stdout"))
#     unittest.TextTestRunner().run(suite)