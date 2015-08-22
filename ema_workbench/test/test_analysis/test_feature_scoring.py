'''
Created on Jul 9, 2014

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
import unittest

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from util import ema_logging
from test import test_utilities
from analysis import feature_scoring as fs
from sklearn.ensemble.forest import RandomForestRegressor

class FeatureScoringTestCase(unittest.TestCase):
    def test_prepare_experiments(self):
        x = np.array([(0,1,2,1),
                      (2,5,6,1),
                      (3,2,1,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float),
                            ('c', np.float),
                            ('d', np.float)])
        x = fs._prepare_experiments(x)
        
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
        x = fs._prepare_experiments(x)
       
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


    def test_prepare_outcomes(self):
        results = test_utilities.load_flu_data()
        
        # string type correct
        ooi = 'nr deaths'
        results[1][ooi] = results[1]['deceased population region 1'][:,-1]
        y, categorical = fs._prepare_outcomes(results[1], ooi)
        
        self.assertFalse(categorical)
        self.assertTrue(len(y.shape)==1)
        
        # string type not correct --> KeyError
        with self.assertRaises(KeyError):
            fs._prepare_outcomes(results[1], "non existing key")
        
        # classify function correct
        def classify(data):
            result = data['deceased population region 1']
            classes =  np.zeros(result.shape[0])
            classes[result[:, -1] > 1000000] = 1
            return classes
        
        y, categorical = fs._prepare_outcomes(results[1], classify)
        
        self.assertTrue(categorical)
        self.assertTrue(len(y.shape)==1)
        
        # neither string nor classify function --> TypeError
        with self.assertRaises(TypeError):
            fs._prepare_outcomes(results[1], 1)
   
   
    def test_get_univariate_feature_scores(self):
        results = test_utilities.load_flu_data()
        
        def classify(data):
            #get the output for deceased population
            result = data['deceased population region 1']
            
            #make an empty array of length equal to number of cases 
            classes =  np.zeros(result.shape[0])
            
            #if deceased population is higher then 1.000.000 people, classify as 1 
            classes[result[:, -1] > 1000000] = 1
            
            return classes
        
        # f classify
        scores = fs.get_univariate_feature_scores(results, classify)
        self.assertEqual(len(scores), len(results[0].dtype.fields))

        # chi2
        scores = fs.get_univariate_feature_scores(results, classify, 
                                                  score_func='chi2')
        self.assertEqual(len(scores), len(results[0].dtype.fields))
        
        # f regression
        ooi = 'nr deaths'
        results[1][ooi] = results[1]['deceased population region 1'][:,-1]
        scores = fs.get_univariate_feature_scores(results, ooi)
        self.assertEqual(len(scores), len(results[0].dtype.fields))
        
   
    def test_get_rf_feature_scores(self):
        results = test_utilities.load_flu_data()
                
        def classify(data):
            #get the output for deceased population
            result = data['deceased population region 1']
            
            #make an empty array of length equal to number of cases 
            classes =  np.zeros(result.shape[0])
            
            #if deceased population is higher then 1.000.000 people, classify as 1 
            classes[result[:, -1] > 1000000] = 1
            
            return classes
        
        scores, forest = fs.get_rf_feature_scores(results, classify, 
                                                  random_state=10)
        
        self.assertEqual(len(scores), len(results[0].dtype.fields))
        self.assertTrue(isinstance(forest, RandomForestClassifier))
        
        ooi = 'nr deaths'
        results[1][ooi] = results[1]['deceased population region 1'][:,-1]
        scores, forest = fs.get_rf_feature_scores(results, ooi, 
                                                  random_state=10)
        
        self.assertEqual(len(scores), len(results[0].dtype.fields))
        self.assertTrue(isinstance(forest, RandomForestRegressor))
        
    def test_get_lasso_feature_scores(self):
        results = test_utilities.load_flu_data()
                
        def classify(data):
            #get the output for deceased population
            result = data['deceased population region 1']
            
            #make an empty array of length equal to number of cases 
            classes =  np.zeros(result.shape[0])
            
            #if deceased population is higher then 1.000.000 people, classify as 1 
            classes[result[:, -1] > 1000000] = 1
            
            return classes
        
        # classification based
        scores = fs.get_lasso_feature_scores(results, classify, random_state=42)
        self.assertEqual(len(scores), len(results[0].dtype.fields))
                
        #regression based
        ooi = 'nr deaths'
        results[1][ooi] = results[1]['deceased population region 1'][:,-1]
        scores = fs.get_lasso_feature_scores(results, ooi,random_state=42)
        self.assertEqual(len(scores), len(results[0].dtype.fields))
        
if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)   
    unittest.main()

#     suite = unittest.TestSuite()
#     suite.addTest(FeatureScoringTestCase("test_get_lasso_feature_scores"))
#     unittest.TextTestRunner().run(suite)