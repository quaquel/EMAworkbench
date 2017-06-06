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
from ema_workbench.em_framework.parameters import RealParameter
from ema_workbench.em_framework.outcomes import ScalarOutcome
from ema_workbench.em_framework.optimization import (Problem, RobustProblem, 
             to_dataframe, to_platypus_types, to_problem, to_robust_problem)

# Created on 6 Jun 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

class TestProblem(unittest.TestCase):

    def test_problem(self):
        with mock.patch('ema_workbench.em_framework.optimization.PlatypusProblem'):
            
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
        with mock.patch('ema_workbench.em_framework.optimization.PlatypusProblem'):
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
    pass

#     def test_to_dataframe(self):
#         df = to_dataframe(optimizer, dvnames, outcome_names) 
#     
#     def test_to_platypus_types(self):
#         types = to_platypus_types(decision_variables)
#     
#     def test_to_problem(self):
#         searchover = 'levers'
#         problem = to_problem(model, searchover)
#         self.assertEqual(searchover, problem.searchover)
#         self.assertEqual(parameters, problem.parameters)
#         self.assertEqual(parameter_names, problem.parameter_names)
#         self.assertEqual(outcome_names, problem.outcome_names)
# 
#         searchover = 'uncertainties'        
#         problem = to_problem(model, searchover)
#         
#         self.assertEqual(searchover, problem.searchover)
#         self.assertEqual(parameters, problem.parameters)
#         self.assertEqual(parameter_names, problem.parameter_names)
#         self.assertEqual(outcome_names, problem.outcome_names)

class TestRobustOptimization(unittest.TestCase):
    def test_to_robust_problem(self):
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

if __name__ == '__main__':
    unittest.main()