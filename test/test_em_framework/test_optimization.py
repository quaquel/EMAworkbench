'''


'''
from __future__ import (unicode_literals, print_function, absolute_import,
                                        division)


import unittest

import unittest.mock as mock


from ema_workbench.em_framework.model import Model
from ema_workbench.em_framework.parameters import (RealParameter,
                                       IntegerParameter, CategoricalParameter)
from ema_workbench.em_framework.outcomes import ScalarOutcome, Constraint
from ema_workbench.em_framework.optimization import (Problem, RobustProblem, 
             to_dataframe, to_platypus_types, to_problem, to_robust_problem)

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
        outcome_names = ['x', 'y']
        
        constraints = [Constraint('n', function= lambda x:x)] 
        
        problem = Problem(searchover, parameters, outcome_names,constraints)

        self.assertEqual(searchover, problem.searchover)
        self.assertEqual(parameters, problem.parameters)
        self.assertEqual(outcome_names, problem.outcome_names)
        self.assertEqual(constraints, problem.ema_constraints)
        self.assertEqual([c.name for c in constraints], 
                         problem.constraint_names)
        
        searchover = 'uncertainties'
        problem = Problem(searchover, parameters, outcome_names, constraints)

        self.assertEqual(searchover, problem.searchover)
        self.assertEqual(parameters, problem.parameters)
        self.assertEqual(outcome_names, problem.outcome_names)
        self.assertEqual(constraints, problem.ema_constraints)
        self.assertEqual([c.name for c in constraints], 
                         problem.constraint_names)
    
    def test_robust_problem(self):
        parameters = [mock.Mock(), mock.Mock()]
        outcome_names = ['x', 'y']
        scenarios = 10
        robustness_functions = [mock.Mock(), mock.Mock()]
        constraints = [Constraint('n', function= lambda x:x)] 
        
        problem = RobustProblem(parameters, outcome_names, 
                                scenarios, robustness_functions, 
                                constraints)
        
        self.assertEqual('robust', problem.searchover)
        self.assertEqual(parameters, problem.parameters)
        self.assertEqual(outcome_names, problem.outcome_names)
        self.assertEqual(constraints, problem.ema_constraints)
        self.assertEqual([c.name for c in constraints], 
                         problem.constraint_names)
    
class TestOptimization(unittest.TestCase):
    @mock.patch('ema_workbench.em_framework.optimization.platypus')
    def test_to_dataframe(self, mocked_platypus):
        problem = mock.Mock()
        type = mocked_platypus.Real  # @ReservedAssignment
        type.decode.return_value = 0
        problem.types = [type, type]
        
        result1 = mock.Mock()
        result1.variables = [0,0]
        result1.objectives = [1,1]
        result1.problem = problem
        
        result2 = mock.Mock()
        result2.variables = [0,0]
        result2.objectives = [0,0]
        result2.problem = problem

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
        self.assertTrue(str(types[2]).find("platypus.Subset") != -1)

    @mock.patch('ema_workbench.em_framework.optimization.platypus')
    def test_to_problem(self, mocked_platypus):
        mocked_model = Model('test', function=mock.Mock())
        mocked_model.levers = [RealParameter('a', 0, 1),
                               RealParameter('b', 0, 1)]
        mocked_model.uncertainties = [RealParameter('c', 0, 1),
                                      RealParameter('d', 0, 1)]
        mocked_model.outcomes = [ScalarOutcome('x', kind=1),
                                 ScalarOutcome('y', kind=1)]
        
        searchover = 'levers'
        problem = to_problem(mocked_model, searchover)
        self.assertEqual(searchover, problem.searchover)
        
        for entry in problem.parameters:
            self.assertIn(entry.name, mocked_model.levers.keys())
            self.assertIn(entry, list(mocked_model.levers))
        for entry in problem.outcome_names:
            self.assertIn(entry, mocked_model.outcomes.keys())
 
        searchover = 'uncertainties'        
        problem = to_problem(mocked_model, searchover)
         
        self.assertEqual(searchover, problem.searchover)
        for entry in problem.parameters:
            self.assertIn(entry.name, mocked_model.uncertainties.keys())
            self.assertIn(entry, list(mocked_model.uncertainties))
        for entry in problem.outcome_names:
            self.assertIn(entry, mocked_model.outcomes.keys())
            
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
                                              function=mock.Mock(),
                                              kind='maximize'), 
                                ScalarOutcome('mean y', variable_name='y', 
                                              function=mock.Mock(),
                                              kind='maximize')]
        
        problem = to_robust_problem(mocked_model, scenarios, 
                                    robustness_functions)
        
        self.assertEqual('robust', problem.searchover)
        for entry in problem.parameters:
            self.assertIn(entry.name, mocked_model.levers.keys())
        self.assertEqual(['mean x', 'mean y'], problem.outcome_names)   
        
    def test_process_robust(self):
        pass

if __name__ == '__main__':
    unittest.main()