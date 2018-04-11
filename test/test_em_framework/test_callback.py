'''
Created on 22 Jan 2013

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
import random
import unittest

import mock
import numpy as np
import numpy.lib.recfunctions as rf

from ema_workbench.em_framework.callbacks import DefaultCallback
from ema_workbench.em_framework.parameters import (CategoricalParameter,
                                                      RealParameter, 
                                                      IntegerParameter)
from ema_workbench.em_framework.parameters import Policy, Scenario, Case
from ema_workbench.util import EMAError
from ema_workbench.em_framework.outcomes import TimeSeriesOutcome 
from ema_workbench.em_framework.util import NamedObject

class TestDefaultCallback(unittest.TestCase):
    def test_init(self):
        # let's add some uncertainties to this
        uncs = [RealParameter("a", 0, 1),
               RealParameter("b", 0, 1)]
        outcomes = [TimeSeriesOutcome("test")]
        constraints = []
        callback = DefaultCallback(uncs, [], outcomes, constraints,
                                   nr_experiments=100)
        
        self.assertEqual(callback.i, 0)
        self.assertEqual(callback.nr_experiments, 100)
        self.assertEqual(callback.cases.shape[0], 100)
        self.assertEqual(callback.outcomes, [o.name for o in outcomes])
        
        names = rf.get_names(callback.cases.dtype)
        names = set(names)
        self.assertEqual(names, {'a', 'b', 'policy', 'model', 'scenario_id'})
        self.assertEqual(callback.results, {})
        
        # with levers
        levers = [RealParameter('c', 0, 10)]
        
        callback = DefaultCallback(uncs, levers, outcomes, constraints, 
                                   nr_experiments=100)
        
        self.assertEqual(callback.i, 0)
        self.assertEqual(callback.nr_experiments, 100)
        self.assertEqual(callback.cases.shape[0], 100)
        self.assertEqual(callback.outcomes, [o.name for o in outcomes])
        
        names = rf.get_names(callback.cases.dtype)
        names = set(names)
        self.assertEqual(names, {'a', 'b', 'c','policy', 'model', 'scenario_id'})
        self.assertEqual(callback.results, {})

    def test_store_results(self):
        nr_experiments = 3
        uncs = [RealParameter("a", 0, 1),
               RealParameter("b", 0, 1)]
        outcomes = [TimeSeriesOutcome("test")]
        constraints = []
        model = NamedObject('test')

        experiment = Case(0, model, Policy('policy'), 
                                Scenario(a=1, b=0), 0)
     
        # case 1 scalar shape = (1)
        callback = DefaultCallback(uncs, [], outcomes, constraints,
                                   nr_experiments=nr_experiments)
        model_outcomes = {outcomes[0].name: 1}
        model_constraints = {}
        callback(experiment, model_outcomes, model_constraints)
         
        _, out = callback.get_results()
        
        self.assertIn(outcomes[0].name, out.keys())
        self.assertEqual(out[outcomes[0].name].shape, (3,))
     
        # case 2 time series shape = (1, nr_time_steps)
        callback = DefaultCallback(uncs, [], outcomes, constraints, 
                                   nr_experiments=nr_experiments)
        model_outcomes = {outcomes[0].name: np.random.rand(10)}
        model_constraints = {}
        callback(experiment, model_outcomes, model_constraints)
          
        _, out = callback.get_results()
        self.assertIn(outcomes[0].name, out.keys())
        self.assertEqual(out[outcomes[0].name].shape, (3,10))

        # case 3 maps etc. shape = (x,y)
        callback = DefaultCallback(uncs, [], outcomes, constraints,
                                   nr_experiments=nr_experiments)
        model_outcomes = {outcomes[0].name: np.random.rand(2,2)}
        model_constraints = {}
        callback(experiment, model_outcomes, model_constraints)
          
        _, out = callback.get_results()
        self.assertIn(outcomes[0].name, out.keys())
        self.assertEqual(out[outcomes[0].name].shape, (3,2,2))

        # case 4 assert raises EMAError
        callback = DefaultCallback(uncs, [], outcomes, constraints,
                                   nr_experiments=nr_experiments)
        model_outcomes = {outcomes[0].name: np.random.rand(2,2,2)}
        model_constraints = {}
        self.assertRaises(EMAError, callback, experiment, model_outcomes, 
                          model_constraints)
        
        # KeyError
        with mock.patch('ema_workbench.util.ema_logging.debug') as mocked_logging:
            callback = DefaultCallback(uncs, [], outcomes, constraints,
                                       nr_experiments=nr_experiments)
            model_outcomes = {'incorrect': np.random.rand(2,)}
            model_constraints = {}
            callback(experiment, model_outcomes, model_constraints)
            
            for outcome in outcomes:
                mocked_logging.assert_called_with("%s not specified as outcome in msi" % outcome.name)
              
    def test_store_cases(self):
        nr_experiments = 3
        uncs = [RealParameter("a", 0, 1),
                RealParameter("b", 0, 1),
                CategoricalParameter('c', [0, 1, 2]),
                IntegerParameter("d", 0, 1)]
        outcomes = [TimeSeriesOutcome("test")]
        constraints = []
        case = {unc.name:random.random() for unc in uncs}
        case["c"] = int(round(case["c"]*2))
        case["d"] = int(round(case["d"]))
        
        model = NamedObject('test')
        policy  = Policy('policy')
        scenario = Scenario(**case)
        experiment = Case(0, model.name, policy, scenario, 0)
     
        callback = DefaultCallback(uncs, [],outcomes, constraints,
                                   nr_experiments=nr_experiments,
                                   reporting_interval=1)
        model_outcomes = {outcomes[0].name: 1}
        model_constraints = {}
        callback(experiment, model_outcomes, model_constraints)
         
        experiments, _ = callback.get_results()
        design = case
        design['policy'] = policy.name
        design['model'] = model.name
        design['scenario_id'] = scenario.name
        
        names = rf.get_names(experiments.dtype)
        for name in names:
            entry_a = experiments[name][0]
            entry_b = design[name]
            
            self.assertEqual(entry_a, entry_b, "failed for "+name)
             
        # with levers
        nr_experiments = 3
        uncs = [RealParameter("a", 0, 1),
                RealParameter("b", 0, 1)]
        levers = [RealParameter("c", 0, 1),
                  RealParameter("d", 0, 1)]
        outcomes = [TimeSeriesOutcome("test")]
        case = {unc.name:random.random() for unc in uncs}
        
        model = NamedObject('test')
        policy  = Policy('policy', c=1, d=1)
        scenario = Scenario(**case)
        experiment = Case(0, model.name, policy, scenario, 0)
     
        callback = DefaultCallback(uncs, levers,outcomes,constraints, 
                                   nr_experiments=nr_experiments,
                                   reporting_interval=1)
        model_outcomes = {outcomes[0].name: 1}
        model_constraints = {}
        callback(experiment, model_outcomes, model_constraints)
         
        experiments, _ = callback.get_results()
        design = case
        design['c'] = 1
        design['d'] = 1
        design['policy'] = policy.name
        design['model'] = model.name
        design['scenario_id'] = scenario.name
        
        names = rf.get_names(experiments.dtype)
        
        for name in names:
            self.assertEqual(experiments[name][0], design[name])
        
 

if __name__ == "__main__":
    unittest.main()
    