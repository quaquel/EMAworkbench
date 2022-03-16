"""
Created on Jul 28, 2015

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
"""
import unittest
import unittest.mock as mock

from ema_workbench.em_framework.outcomes import (ScalarOutcome,
                                                 create_outcomes,
                                                 TimeSeriesOutcome, register)

class TestScalarOutcome(unittest.TestCase):
    outcome_class = ScalarOutcome
    outcome_klass = "ScalarOutcome"
    
    def test_outcome(self):
        register.outcomes = {}
        name = 'a'
        outcome = self.outcome_class(name)
        
        self.assertEqual(outcome.name, name)
        self.assertEqual(outcome.variable_name, [name])
        self.assertIsNone(outcome.function)
        self.assertEqual(repr(outcome), self.outcome_klass+'(\'a\')')

        name = 'b'
        var_name = 'something else'
        outcome = self.outcome_class(name, variable_name=var_name)
        
        self.assertEqual(outcome.name, name)
        self.assertEqual(outcome.variable_name, (var_name,))
        self.assertIsNone(outcome.function)

        name = 'c'
        var_name = 'something else'
        function = mock.Mock()
        outcome = self.outcome_class(name, variable_name=var_name, 
                                function=function)
        
        self.assertEqual(outcome.name, name)
        self.assertEqual(outcome.variable_name, (var_name,))
        self.assertIsNotNone(outcome.function)
        
        with self.assertRaises(ValueError):
            name = 'd'
            var_name = 'something else'
            function = 'not a function'
            outcome = self.outcome_class(name, variable_name=var_name, 
                                    function=function)
        
        with self.assertRaises(ValueError):
            name = 'e'
            var_name = 1
            outcome = self.outcome_class(name, variable_name=var_name, 
                                    function=function)
        
        with self.assertRaises(ValueError):
            name = 'f'
            var_name = ['a variable', 1]
            outcome = self.outcome_class(name, variable_name=var_name, 
                                    function=function)
        
        name = 'g'
        var_name = 'something else'
        function = lambda x: x
        outcome1 = self.outcome_class(name, variable_name=var_name, 
                                function=function)

        outcome2 = self.outcome_class(name, variable_name=var_name, 
                                function=function)
        
        self.assertEqual(outcome1, outcome2)


    def test_process(self):
        register.outcomes = {}
        name = 'q'
        outcome = self.outcome_class(name)
        
        outputs = [1]
        self.assertEqual(outcome.process(outputs), outputs[0])

        name = 'r'
        function = mock.Mock()
        function.return_value = 2
        outcome = self.outcome_class(name, function=function)
        
        outputs = [1]
        self.assertEqual(outcome.process(outputs), 2)
        function.assert_called_once()


        name = 's'
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
            name = 't'
            function = mock.Mock()
            function.return_value = 2
            variable_name = ['a', 'b']
            
            outcome = self.outcome_class(name, function=function, 
                                    variable_name=variable_name)
            
            outcome.process([1])

class TestTimeSeriesOutcome(TestScalarOutcome):
    outcome_class = TimeSeriesOutcome
    outcome_klass = "TimeSeriesOutcome"

    def test_process(self):
        register.outcomes = {}
        name = 'a'
        outcome = self.outcome_class(name)

        outputs = [[1]]
        self.assertEqual(outcome.process(outputs), outputs[0])

        name = 'b'
        function = mock.Mock()
        function.return_value = [2]
        outcome = self.outcome_class(name, function=function)

        outputs = [1]
        self.assertEqual(outcome.process(outputs), [2])
        function.assert_called_once()


        name = 'c'
        function = mock.Mock()
        function.return_value = [2]
        variable_name = ['a', 'b']

        outcome = self.outcome_class(name, function=function,
                                variable_name=variable_name)

        outputs = [1, 2]
        self.assertEqual(outcome.process(outputs), [2])
        function.assert_called_once()
        function.assert_called_with(1, 2)

        with self.assertRaises(ValueError):
            name = 'd'
            function = mock.Mock()
            function.return_value = [2]
            variable_name = ['a', 'b']

            outcome = self.outcome_class(name, function=function,
                                    variable_name=variable_name)

            outcome.process([1])


class CreateOutcomesTestCase(unittest.TestCase):
    def test_create_outcomes(self):
        register.outcomes = {}
        outcome_list = [dict(type='scalar', name='a'),
                        dict(type='timeseries', name='b')]

        outcomes = create_outcomes(outcome_list)

        for x, y in zip(outcome_list, outcomes):
            self.assertEqual(x['name'], y.name)

        with self.assertRaises(ValueError):
            outcome_list = [dict(type='unknown', name='a')]
            outcomes = create_outcomes(outcome_list)

        with self.assertRaises(ValueError):
            outcome_list = [dict(kind='unknown', name='a')]
            outcomes = create_outcomes(outcome_list)


if __name__ == "__main__":
    unittest.main()