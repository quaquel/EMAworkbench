'''
Created on Jul 28, 2015

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import unittest
import mock

from ema_workbench.em_framework.outcomes import ScalarOutcome

class TestScalarOutcome(unittest.TestCase):
    def test(self):
        name = 'test'
        outcome = ScalarOutcome(name)
        
        self.assertEqual(outcome.name, name)
        self.assertEqual(outcome.variable_name, name)
        self.assertIsNone(outcome.function)

        name = 'test'
        var_name = 'something else'
        outcome = ScalarOutcome(name, variable_name=var_name)
        
        self.assertEqual(outcome.name, name)
        self.assertEqual(outcome.variable_name, var_name)
        self.assertIsNone(outcome.function)

        name = 'test'
        var_name = 'something else'
        function = mock.Mock()
        outcome = ScalarOutcome(name, variable_name=var_name, 
                                function=function)
        
        self.assertEqual(outcome.name, name)
        self.assertEqual(outcome.variable_name, var_name)
        self.assertIsNotNone(outcome.function)
        
        with self.assertRaises(ValueError):
            name = 'test'
            var_name = 'something else'
            function = 'not a function'
            outcome = ScalarOutcome(name, variable_name=var_name, 
                                    function=function)
        
        name = 'test'
        var_name = 'something else'
        function = lambda x: x
        outcome1 = ScalarOutcome(name, variable_name=var_name, 
                                function=function)

        outcome2 = ScalarOutcome(name, variable_name=var_name, 
                                function=function)
        
        self.assertEqual(outcome1, outcome2)

if __name__ == "__main__":
    unittest.main()