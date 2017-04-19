'''
Created on Jul 28, 2015

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import unittest
import mock

from ema_workbench.em_framework.outcomes import ScalarOutcome,\
    TimeSeriesOutcome

class TestScalarOutcome(unittest.TestCase):
    outcome_class = ScalarOutcome
    outcome_klass = "ScalarOutcome"
    
    def test_outcome(self):
        name = 'test'
        outcome = self.outcome_class(name)
        
        self.assertEqual(outcome.name, name)
        self.assertEqual(outcome.variable_name, [name])
        self.assertIsNone(outcome.function)
        self.assertEqual(repr(outcome), self.outcome_klass+'(\'test\')')

        name = 'test'
        var_name = 'something else'
        outcome = self.outcome_class(name, variable_name=var_name)
        
        self.assertEqual(outcome.name, name)
        self.assertEqual(outcome.variable_name, [var_name])
        self.assertIsNone(outcome.function)

        name = 'test'
        var_name = 'something else'
        function = mock.Mock()
        outcome = self.outcome_class(name, variable_name=var_name, 
                                function=function)
        
        self.assertEqual(outcome.name, name)
        self.assertEqual(outcome.variable_name, [var_name])
        self.assertIsNotNone(outcome.function)
        
        with self.assertRaises(ValueError):
            name = 'test'
            var_name = 'something else'
            function = 'not a function'
            outcome = self.outcome_class(name, variable_name=var_name, 
                                    function=function)
        
        with self.assertRaises(ValueError):
            name = 'test'
            var_name = 1
            outcome = self.outcome_class(name, variable_name=var_name, 
                                    function=function)
        
        with self.assertRaises(ValueError):
            name = 'test'
            var_name = ['a variable', 1]
            outcome = self.outcome_class(name, variable_name=var_name, 
                                    function=function)
        
        name = 'test'
        var_name = 'something else'
        function = lambda x: x
        outcome1 = self.outcome_class(name, variable_name=var_name, 
                                function=function)

        outcome2 = self.outcome_class(name, variable_name=var_name, 
                                function=function)
        
        self.assertEqual(outcome1, outcome2)


    def test_process(self):
        name = 'test'
        outcome = self.outcome_class(name)
        
        outputs = [1]
        self.assertEqual(outcome.process(outputs), outputs[0])

        name = 'test'
        function = mock.Mock()
        function.return_value = 2
        outcome = self.outcome_class(name, function=function)
        
        outputs = [1]
        self.assertEqual(outcome.process(outputs), 2)
        function.assert_called_once()


        name = 'test'
        function = mock.Mock()
        function.return_value = 2
        variable_name = ['a', 'b']
        
        outcome = self.outcome_class(name, function=function, 
                                variable_name=variable_name)
        
        outputs = [1, 2]
        self.assertEqual(outcome.process(outputs), 2)
        function.assert_called_once()
        function.assert_called_with(1, 2)

        with self.assertRaises(ValueError):
            name = 'test'
            function = mock.Mock()
            function.return_value = 2
            variable_name = ['a', 'b']
            
            outcome = self.outcome_class(name, function=function, 
                                    variable_name=variable_name)
            
            outcome.process([1])

class TestTimeSeriesOutcome(TestScalarOutcome):
    outcome_class = TimeSeriesOutcome
    outcome_klass = "TimeSeriesOutcome"

if __name__ == "__main__":
    unittest.main()