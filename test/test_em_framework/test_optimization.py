'''


'''
from __future__ import (unicode_literals, print_function, absolute_import,
                                        division)


import unittest

try:
    import unittest.mock as mock
except ImportError:
    import mock

from ema_workbench.em_framework.model import Model
from ema_workbench.em_framework.parameters import RealParameter,\
    IntegerParameter, CategoricalParameter
from ema_workbench.em_framework.outcomes import ScalarOutcome
from ema_workbench.em_framework.optimization import (Problem, RobustProblem, 
             to_dataframe, to_platypus_types, to_problem, to_robust_problem,
             process_levers, process_uncertainties, process_robust)

# Created on 6 Jun 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

class MockProblem(object):
    def __init__(self, ndv, nobjs, nconstr=0):
        pass
    
class TestProblem(unittest.TestCase):

   
    def test_problem(self):
        # evil way to mock super
        searchover = 'levers'
        parameters = [mock.Mock(), mock.Mock()]
        parameter_names = ['a', 'b']
        outcome_names = ['x', 'y']
        
        problem = Problem(searchover, parameters, parameter_names, 
                          outcome_names)

        self.assertEqual(searchover, problem.searchover)
        self.assertEqual(parameters, problem.parameters)
        self.assertEqual(parameter_names, problem.parameter_names)
        self.assertEqual(outcome_names, problem.outcome_names)
        
        searchover = 'uncertainties'
        problem = Problem(searchover, parameters, parameter_names, 
                          outcome_names)

        self.assertEqual(searchover, problem.searchover)
        self.assertEqual(parameters, problem.parameters)
        self.assertEqual(parameter_names, problem.parameter_names)
        self.assertEqual(outcome_names, problem.outcome_names)
    
    def test_robust_problem(self):
        parameters = [mock.Mock(), mock.Mock()]
        parameter_names = ['a', 'b']
        outcome_names = ['x', 'y']
        scenarios = 10
        robustness_functions = [mock.Mock(), mock.Mock()]
        
        problem = RobustProblem(parameters, parameter_names, outcome_names, 
                                scenarios, robustness_functions)
        
        self.assertEqual('robust', problem.searchover)
        self.assertEqual(parameters, problem.parameters)
        self.assertEqual(parameter_names, problem.parameter_names)
        self.assertEqual(outcome_names, problem.outcome_names)      

    
class TestOptimization(unittest.TestCase):
    @mock.patch('ema_workbench.em_framework.optimization.platypus')
    def test_to_dataframe(self, mocked_platypus):
        result1 = mock.Mock()
        result1.variables = [0,0]
        result1.objectives = [1,1]
        result2 = mock.Mock()
        result2.variables = [1,1]
        result2.objectives = [0,0]

        data = [result1, result2]
        
        mocked_platypus.unique.return_value = data 
        optimizer = mock.Mock()
        optimizer.results = data
        
        dvnames = ['a', 'b']
        outcome_names = ['x', 'y']
        
        df = to_dataframe(optimizer, dvnames, outcome_names) 
        self.assertListEqual(list(df.columns.values),['a', 'b', 'x', 'y'] )
        
        for i, entry in enumerate(data):
            self.assertListEqual(list((df.loc[i, dvnames].values)), 
                                      entry.variables)
            self.assertListEqual(list((df.loc[i, outcome_names].values)), 
                                      entry.objectives)
     

    @mock.patch('ema_workbench.em_framework.optimization.platypus')
    def test_to_platypus_types(self, mocked_platypus):
        dv = [RealParameter("real", 0, 1),
              IntegerParameter("integer", 0, 10),
              CategoricalParameter("categorical", ["a", "b"])]
        
        types = to_platypus_types(dv)
        self.assertTrue(str(types[0]).find("platypus.Real") != -1)
        self.assertTrue(str(types[1]).find("platypus.Integer") != -1)
        self.assertTrue(str(types[2]).find("platypus.Permutation") != -1)

    @mock.patch('ema_workbench.em_framework.optimization.platypus')
    def test_to_problem(self, mocked_platypus):
        mocked_model = Model('test', function=mock.Mock())
        mocked_model.levers = [RealParameter('a', 0, 1),
                               RealParameter('b', 0, 1)]
        mocked_model.uncertainties = [RealParameter('c', 0, 1),
                                      RealParameter('d', 0, 1)]
        mocked_model.outcomes = [ScalarOutcome('x'), ScalarOutcome('y')]
        
        searchover = 'levers'
        problem = to_problem(mocked_model, searchover)
        self.assertEqual(searchover, problem.searchover)
        
        for entry in problem.parameters:
            self.assertIn(entry.name, mocked_model.levers.keys())
            self.assertIn(entry, list(mocked_model.levers))
        for entry in problem.outcome_names:
            self.assertIn(entry.name, mocked_model.outcomes.keys())
 
        searchover = 'uncertainties'        
        problem = to_problem(mocked_model, searchover)
         
        self.assertEqual(searchover, problem.searchover)
        for entry in problem.parameters:
            self.assertIn(entry.name, mocked_model.uncertainties.keys())
            self.assertIn(entry, list(mocked_model.uncertainties))
        for entry in problem.outcome_names:
            self.assertIn(entry.name, mocked_model.outcomes.keys())
            
    def test_process_levers(self):
        pass
    
    def test_process_uncertainties(self):
        pass

class TestRobustOptimization(unittest.TestCase):
    
    @mock.patch('ema_workbench.em_framework.optimization.platypus')
    def test_to_robust_problem(self, mocked_platypus):
        mocked_model = Model('test', function=mock.Mock())
        mocked_model.levers = [RealParameter('a', 0, 1),
                               RealParameter('b', 0, 1)]
        mocked_model.uncertainties = [RealParameter('c', 0, 1),
                                      RealParameter('d', 0, 1)]
        mocked_model.outcomes = [ScalarOutcome('x'), ScalarOutcome('y')]
        
        scenarios = 5
        robustness_functions = [ScalarOutcome('mean x', variable_name='x',
                                              function=mock.Mock()), 
                                ScalarOutcome('mean y', variable_name='y', 
                                              function=mock.Mock())]
        
        problem = to_robust_problem(mocked_model, scenarios, robustness_functions)
        
        self.assertEqual('robust', problem.searchover)
        for entry in problem.parameters:
            self.assertIn(entry.name, mocked_model.levers.keys())
        self.assertEqual(['a', 'b'], problem.parameter_names)
        self.assertEqual(['mean x', 'mean y'], problem.outcome_names)   
        
    def test_process_robust(self):
        pass

if __name__ == '__main__':
    unittest.main()