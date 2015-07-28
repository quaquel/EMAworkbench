'''
Created on 22 Jan 2013

@author: jhkwakkel
'''
import numpy as np
import random
import numpy.lib.recfunctions as rf
import unittest

from expWorkbench.callbacks import DefaultCallback
from expWorkbench import ParameterUncertainty, Outcome



class TestDefaultCallback(unittest.TestCase):
    def test_init(self):
        # let's add some uncertainties to this
        uncs = [ParameterUncertainty((0,1), "a"),
               ParameterUncertainty((0,1), "b")]
        outcomes = [Outcome("test", time=True)]
        callback = DefaultCallback(uncs, outcomes, nr_experiments=100)
        
        self.assertEqual(callback.i, 0)
        self.assertEqual(callback.nr_experiments, 100)
        self.assertEqual(callback.cases.shape[0], 100)
        self.assertEqual(callback.outcomes, outcomes)
        
        names = rf.get_names(callback.cases.dtype)
        names = set(names)
        self.assertEqual(names, {'a', 'b', 'policy', 'model'})
        self.assertEqual(callback.results, {})

    def test_store_results(self):
        nr_experiments = 3
        uncs = [ParameterUncertainty((0,1), "a"),
               ParameterUncertainty((0,1), "b")]
        outcomes = [Outcome("test", time=True)]
        case = {unc.name:random.random() for unc in uncs}
        policy = {'name':'none'}
        name = "test"
     
        # case 1 scalar shape = (1)
        callback = DefaultCallback(uncs, 
                                   [outcome.name for outcome in outcomes], 
                                   nr_experiments=nr_experiments)
        result = {outcomes[0].name: 1}
        callback(0, case, policy, name, result)
         
        _, out = callback.get_results()
        self.assertIn(outcomes[0].name, out.keys())
        self.assertEqual(out[outcomes[0].name].shape, (3,))
     
        # case 2 time series shape = (1, nr_time_steps)
        callback = DefaultCallback(uncs, 
                                   [outcome.name for outcome in outcomes], 
                                   nr_experiments=nr_experiments)
        result = {outcomes[0].name: np.random.rand(10)}
        callback(0, case, policy, name, result)
          
        _, out = callback.get_results()
        self.assertIn(outcomes[0].name, out.keys())
        self.assertEqual(out[outcomes[0].name].shape, (3,10))

        # case 3 maps etc. shape = (x,y)
        callback = DefaultCallback(uncs, 
                                   [outcome.name for outcome in outcomes], 
                                   nr_experiments=nr_experiments)
        result = {outcomes[0].name: np.random.rand(2,2)}
        callback(0, case, policy, name, result)
          
        _, out = callback.get_results()
        self.assertIn(outcomes[0].name, out.keys())
        self.assertEqual(out[outcomes[0].name].shape, (3,2,2))

    
    def test_store_cases(self):
        nr_experiments = 3
        uncs = [ParameterUncertainty((0,1), "a"),
               ParameterUncertainty((0,1), "b")]
        outcomes = [Outcome("test", time=True)]
        case = {unc.name:random.random() for unc in uncs}
        policy = {'name':'none'}
        name = "test"
     
        callback = DefaultCallback(uncs, 
                                   [outcome.name for outcome in outcomes], 
                                   nr_experiments=nr_experiments)
        result = {outcomes[0].name: 1}
        callback(0, case, policy, name, result)
         
        experiments, _ = callback.get_results()
        design = case
        design['policy'] = policy['name']
        design['model'] = name
        
        names = rf.get_names(experiments.dtype)
        for name in names:
            self.assertEqual(experiments[name][0], design[name])
        
 

if __name__ == "__main__":
    unittest.main()
    