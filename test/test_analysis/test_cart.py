'''
Created on May 22, 2015

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
import unittest

import numpy as np
import pandas as pd

from ema_workbench.analysis import cart
from ema_workbench.analysis.scenario_discovery_util import RuleInductionType

from test import utilities



def flu_classify(data):
    # get the output for deceased population
    result = data['deceased population region 1']
    
    # make an empty array of length equal to number of cases
    classes = np.zeros(result.shape[0])
    
    # if deceased population is higher then 1.000.000 people, classify as 1
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
        
        alg = cart.setup_cart(results, flu_classify,
                              mass_min=0.05)
        
        self.assertTrue(alg.mode==RuleInductionType.BINARY)

        x, outcomes = results
        y = {}
        
        for k, v in outcomes.items():
            y[k] = v[:, -1] 
            
        temp_results = (x, y)
        alg = cart.setup_cart(temp_results,
                              'deceased population region 1',
                              mass_min=0.05)
        self.assertTrue(alg.mode == RuleInductionType.REGRESSION)

        n_cols = 5
        unc = x.columns.values[0:n_cols]
        alg = cart.setup_cart(results,
                              flu_classify,
                              mass_min=0.05,
                              incl_unc=unc)
        self.assertTrue(alg.mode == RuleInductionType.BINARY)
        self.assertTrue(alg.x.shape[1] == n_cols)

        with self.assertRaises(TypeError):
            alg = cart.setup_cart(results, 10,
                                  mass_min=0.05)
        
        # setup can be generalized --> cart and prim essentially the same
        # underlying code so move to sdutil
        
        # check include uncertainties
        
        # string type
        # --> classification vs. regresion
        
        # callable

    def test_boxes(self):
        np.random.seed(42)
        x = pd.DataFrame(np.random.rand(1000,2), columns=['a', 'b'])
        y = (x.a>0.5) & (x.b <0.5)
        alg = cart.CART(x,y, mode=RuleInductionType.BINARY)
        alg.build_tree()
        
        boxes = alg.boxes
        
        self.assertEqual(len(boxes), 3)


    def test_stats(self):
        x = pd.DataFrame([(0,1,2),
                          (2,5,6),
                          (3,2,1)], 
                          columns=['a', 'b', 'c'])
        

        box = pd.DataFrame([(0,1,1),
                            (3,5,3)], 
                            columns=['a', 'b', 'c'])

        y = np.array([0,1,1])
        alg = cart.CART(x,y, mode=RuleInductionType.BINARY)
        alg._boxes = [box]
        alg.clf = "something"
        stats = alg.stats[0]
                
        self.assertEqual(stats['coverage'], 0.5)
        self.assertEqual(stats['density'], 0.5)
        self.assertEqual(stats['res dim'], 1)
        self.assertEqual(stats['mass'], 2/3)
        
        y = np.array([0,1,2])
        alg = cart.CART(x,y, mode=RuleInductionType.REGRESSION)
        alg._boxes = [box]
        alg.clf = "something"
        stats = alg.stats[0]

        self.assertEqual(stats['mean'], 1)
        self.assertEqual(stats['res dim'], 1)
        self.assertEqual(stats['mass'], 2/3)

        y = np.array([0,1,2])
        alg = cart.CART(x,y, mode=RuleInductionType.CLASSIFICATION)
        alg._boxes = [box]
        alg.clf = "something"
        stats = alg.stats[0]
        
        self.assertEqual(stats['gini'], 0.5)
        self.assertEqual(stats['box_composition'], [1, 0, 1])
        self.assertEqual(stats['res dim'], 1)
        self.assertEqual(stats['mass'], 2/3)
        
        self.assertEqual(stats, alg.stats[0])


    def test_stats_to_dataframe(self):
        x, outcomes = utilities.load_flu_data()

        y = flu_classify(outcomes)
        alg = cart.CART(x,y, mode=RuleInductionType.BINARY)
        alg.build_tree()
        stats = alg.stats_to_dataframe()
        
        
        y = outcomes['deceased population region 1'][:, -1]
        alg = cart.CART(x,y, mode=RuleInductionType.REGRESSION)
        alg.build_tree()
        stats = alg.stats_to_dataframe()
        

        y = np.random.randint(1, 5, y.shape[0])        
        alg = cart.CART(x,y, mode=RuleInductionType.CLASSIFICATION)
        alg.build_tree()
        stats = alg.stats_to_dataframe()
        print(stats)




    def test_build_tree(self):
        results = utilities.load_flu_data()
        
        alg = cart.setup_cart(results, flu_classify,
                              mass_min=0.05)
        alg.build_tree()
        
        self.assertTrue(isinstance(alg.clf,
                                   cart.tree.DecisionTreeClassifier))

        x, outcomes = results
        y = {}
        
        for k, v in outcomes.items():
            y[k] = v
        
        temp_results = (x, y)
        alg = cart.setup_cart(temp_results,
                              'deceased population region 1',
                              mass_min=0.05)
        alg.build_tree()
        self.assertTrue(isinstance(alg.clf,
                                   cart.tree.DecisionTreeRegressor))
        
#     def test_show_tree(self):
#         results = utilities.load_flu_data()
#         
#         alg = cart.setup_cart(results, flu_classify,
#                               mass_min=0.05)
#         alg.build_tree()
#         
#         fig = alg.show_tree(mplfig=True)
#         bytestream = alg.show_tree(mplfig=False)
#         
#         self.assertTrue(isinstance(fig, mpl.figure.Figure))
#         self.assertTrue(isinstance(bytestream, bytes))
        
        
if __name__ == '__main__':
        unittest.main()

#     suite = unittest.TestSuite()
#     suite.addTest(PrimTestCase("test_write_boxes_to_stdout"))
#     unittest.TextTestRunner().run(suite)