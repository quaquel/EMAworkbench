'''
Created on Jul 28, 2015

.. codeauthor:: jhkwakkel
'''
import unittest
from expWorkbench.ema_exceptions import EMAError, CaseError


class TestEMAError(unittest.TestCase):
    def test(self):
        error = EMAError('a message')

        self.assertEqual(str(error), 'a message')
        
        error = EMAError('a message', 'another message')

        self.assertEqual(str(error), "('a message', 'another message')")


class TestCaseError(unittest.TestCase):
    def test(self):
        error = CaseError('a message', {'a':1, 'b':2})
        self.assertEqual(str(error), "a message case: {a:1, b:2, policy:not specified}")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()