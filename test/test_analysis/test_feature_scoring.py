"""
Created on Jul 9, 2014

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
"""
import unittest

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesClassifier,
                              ExtraTreesRegressor)

from ema_workbench.analysis import feature_scoring as fs
from ema_workbench.analysis.feature_scoring import F_CLASSIFICATION, CHI2, F_REGRESSION
from ema_workbench.analysis.scenario_discovery_util import RuleInductionType
from ema_workbench.util import ema_logging

from ema_workbench import ScalarOutcome

from test import utilities


class FeatureScoringTestCase(unittest.TestCase):
    def test_prepare_experiments(self):
        x = pd.DataFrame([(0, 1, 2, 1),
                          (2, 5, 6, 1),
                          (3, 2, 1, 1)],
                     columns=['a', 'b', 'c', 'd'])
        x, _ = fs._prepare_experiments(x)
        
        correct = np.array([[0, 1, 2, 1],
                            [2, 5, 6, 1],
                            [3, 2, 1, 1]], dtype=np.float)
        
        self.assertTrue(np.all(x==correct))
        
        # heterogeneous without NAN
        
        x = pd.DataFrame({'a':[0.1, 0.2, 0.3, 0.4, 0.5,
                               0.6, 0.7, 0.8, 0.9, 1.0],
                          'b':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                          'c':['a','b','a','b','a','a',
                               'b','a','b','a', ]})
        x, _ = fs._prepare_experiments(x)
       
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
                            ], dtype=float)

        self.assertTrue(np.all(x==correct))


    def test_prepare_outcomes(self):
        experiments, outcomes = utilities.load_flu_data()
        
        # string type correct
        ooi = 'nr deaths'
        outcomes[ooi] = outcomes['deceased population region 1'][:, -1]
        y, categorical = fs._prepare_outcomes(outcomes, ooi)
        
        self.assertFalse(categorical)
        self.assertTrue(len(y.shape)==1)
        
        # string type not correct --> KeyError
        with self.assertRaises(KeyError):
            fs._prepare_outcomes(outcomes, "non existing key")
        
        # classify function correct
        def classify(data):
            result = data['deceased population region 1']
            classes = np.zeros(result.shape[0])
            classes[result[:, -1] > 1000000] = 1
            return classes
        
        y, categorical = fs._prepare_outcomes(outcomes, classify)
        
        self.assertTrue(categorical)
        self.assertTrue(len(y.shape)==1)
        
        # neither string nor classify function --> TypeError
        with self.assertRaises(TypeError):
            fs._prepare_outcomes(outcomes, 1)
   
   
    def test_get_univariate_feature_scores(self):
        x, outcomes = utilities.load_flu_data()
        
        def classify(data):
            #get the output for deceased population
            result = data['deceased population region 1']
            
            #make an empty array of length equal to number of cases 
            classes =  np.zeros(result.shape[0])
            
            #if deceased population is higher then 1.000.000 people, classify as 1 
            classes[result[:, -1] > 1000000] = 1
            
            return classes
        
        y = classify(outcomes)
        
        # f classify
        scores = fs.get_univariate_feature_scores(x,y, 
                                                  score_func=F_CLASSIFICATION)
        self.assertEqual(len(scores), len(x.columns)-3)

        # chi2
        scores = fs.get_univariate_feature_scores(x,y, score_func=CHI2)
        self.assertEqual(len(scores), len(x.columns)-3)
        
        # f regression
        y= outcomes['deceased population region 1'][:,-1]
        scores = fs.get_univariate_feature_scores(x,y, score_func=F_REGRESSION)
        self.assertEqual(len(scores), len(x.columns)-3)

    def test_get_rf_feature_scores(self):
        x, outcomes = utilities.load_flu_data()
                
        def classify(data):
            # get the output for deceased population
            result = data['deceased population region 1']
            
            # make an empty array of length equal to number of cases
            classes =  np.zeros(result.shape[0])
            
            # if deceased population is higher then 1.000.000 people,
            # classify as 1
            classes[result[:, -1] > 1000000] = 1
            
            return classes
        
        y = classify(outcomes)
                
        scores, forest = fs.get_rf_feature_scores(x, y,
                                                  mode=RuleInductionType.CLASSIFICATION,
                                                  random_state=10)
        
        self.assertEqual(len(scores), len(x.columns)-3)
        self.assertTrue(isinstance(forest, RandomForestClassifier))
        
        
        self.assertRaises(ValueError, fs.get_rf_feature_scores, x,y, 
                          mode='illegal argument')
        
        y = outcomes['deceased population region 1'][:,-1]
        scores, forest = fs.get_rf_feature_scores(x,y,
                                                  mode=RuleInductionType.REGRESSION, 
                                                  random_state=10)
        
        self.assertEqual(len(scores), len(x.columns)-3)
        self.assertTrue(isinstance(forest, RandomForestRegressor))
        
    def test_get_ex_feature_scores(self):
        x, outcomes = utilities.load_flu_data()
        y = outcomes['deceased population region 1'][:, -1] > 1000000
                
        scores, forest = fs.get_ex_feature_scores(x,y,
                                                  mode=RuleInductionType.CLASSIFICATION,
                                                  random_state=10)
        
        self.assertEqual(len(scores), len(x.columns)-3)
        self.assertTrue(isinstance(forest, ExtraTreesClassifier))
        
        
        self.assertRaises(ValueError, fs.get_ex_feature_scores, x,y, 
                          mode='illegal argument')
        
        y = outcomes['deceased population region 1'][:,-1]
        scores, forest = fs.get_ex_feature_scores(x,y,
                                                  mode=RuleInductionType.REGRESSION, 
                                                  random_state=10)
        
        self.assertEqual(len(scores), len(x.columns)-3)
        self.assertTrue(isinstance(forest, ExtraTreesRegressor))

    def test_get_feature_scores_all(self):
        x, outcomes = utilities.load_flu_data()

        # we have timeseries so we need scalars
        y = {'deceased population':outcomes['deceased population region 1'][:, -1],
             'max. infected fraction':np.max(outcomes['infected fraction R1'], axis=1)}
        scores = fs.get_feature_scores_all(x,y)
        
        self.assertEqual(len(scores), len(x.columns)-3)
        self.assertTrue(scores.ndim==2)

if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)   
    unittest.main()

#     suite = unittest.TestSuite()
#     suite.addTest(FeatureScoringTestCase("test_get_lasso_feature_scores"))
#     unittest.TextTestRunner().run(suite)