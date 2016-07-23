'''
Created on 22 Jan 2013

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
import random
import unittest

import mock
import numpy as np
import numpy.lib.recfunctions as rf

from ema_workbench.em_framework.callbacks import DefaultCallback
from ema_workbench.em_framework.uncertainties import (CategoricalParameter,
                                                      RealParameter, 
                                                      IntegerParameter)
from ema_workbench.em_framework.parameters import Policy                                           ,\
    Experiment
from ema_workbench.util import EMAError
from ema_workbench.em_framework.outcomes import TimeSeriesOutcome 
from ema_workbench.em_framework.util import NamedObject

class TestDefaultCallback(unittest.TestCase):
    def test_init(self):
        # let's add some uncertainties to this
        uncs = [RealParameter("a", 0, 1),
               RealParameter("b", 0, 1)]
        outcomes = [TimeSeriesOutcome("test")]
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
        uncs = [RealParameter("a", 0, 1),
               RealParameter("b", 0, 1)]
        outcomes = [TimeSeriesOutcome("test")]
        model = NamedObject('test')

        experiment = Experiment(0, model, 'test', 0, a=1)
     
        # case 1 scalar shape = (1)
        callback = DefaultCallback(uncs, 
                                   [outcome.name for outcome in outcomes], 
                                   nr_experiments=nr_experiments)
        result = {outcomes[0].name: 1}
        callback(experiment, result)
         
        _, out = callback.get_results()
        self.assertIn(outcomes[0].name, out.keys())
        self.assertEqual(out[outcomes[0].name].shape, (3,))
     
        # case 2 time series shape = (1, nr_time_steps)
        callback = DefaultCallback(uncs, 
                                   [outcome.name for outcome in outcomes], 
                                   nr_experiments=nr_experiments)
        result = {outcomes[0].name: np.random.rand(10)}
        callback(experiment, result)
          
        _, out = callback.get_results()
        self.assertIn(outcomes[0].name, out.keys())
        self.assertEqual(out[outcomes[0].name].shape, (3,10))

        # case 3 maps etc. shape = (x,y)
        callback = DefaultCallback(uncs, 
                                   [outcome.name for outcome in outcomes], 
                                   nr_experiments=nr_experiments)
        result = {outcomes[0].name: np.random.rand(2,2)}
        callback(experiment,result)
          
        _, out = callback.get_results()
        self.assertIn(outcomes[0].name, out.keys())
        self.assertEqual(out[outcomes[0].name].shape, (3,2,2))

        # case 4 assert raises EMAError
        callback = DefaultCallback(uncs, 
                                   [outcome.name for outcome in outcomes], 
                                   nr_experiments=nr_experiments)
        result = {outcomes[0].name: np.random.rand(2,2,2)}
        self.assertRaises(EMAError, callback, experiment, result)
        
        # KeyError
        with mock.patch('ema_workbench.util.ema_logging.debug') as mocked_logging:
            callback = DefaultCallback(uncs, 
                           [outcome.name for outcome in outcomes], 
                           nr_experiments=nr_experiments)
            result = {'incorrect': np.random.rand(2,)}
            callback(experiment, result)
            
            for outcome in outcomes:
                mocked_logging.assert_called_with("%s not specified as outcome in msi" % outcome.name)
              
    def test_store_cases(self):
        nr_experiments = 3
        uncs = [RealParameter("a", 0, 1),
                RealParameter("b", 0, 1),
                CategoricalParameter('c', [0, 1, 2]),
                IntegerParameter("d", 0, 1)]
        outcomes = [TimeSeriesOutcome("test")]
        case = {unc.name:random.random() for unc in uncs}
        case["c"] = int(round(case["c"]*2))
        case["d"] = int(round(case["d"]))
        
        model = NamedObject('test')
        experiment = Experiment(0, model, 'test', 0, **case)
     
        callback = DefaultCallback(uncs, 
                                   [outcome.name for outcome in outcomes], 
                                   nr_experiments=nr_experiments,
                                   reporting_interval=1)
        result = {outcomes[0].name: 1}
        callback(experiment, result)
         
        experiments, _ = callback.get_results()
        design = case
        design['policy'] = 'test'
        design['model'] = 'test'
        
        names = rf.get_names(experiments.dtype)
        for name in names:
            self.assertEqual(experiments[name][0], design[name])
        
 

if __name__ == "__main__":
    unittest.main()
    