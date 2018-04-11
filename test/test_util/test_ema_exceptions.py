'''
Created on Jul 28, 2015

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import unittest
import sys


from ema_workbench.util.ema_exceptions import EMAError, CaseError


class TestEMAError(unittest.TestCase):
    def test_emaerror(self):
        error = EMAError('a message')

        self.assertEqual(str(error), 'a message')
        
        error = EMAError('a message', 'another message')

        if sys.version_info[0] < 3:
            self.assertEqual(str(error), str("(u'a message', u'another message')"))
        else: 
            self.assertEqual(str(error), str("('a message', 'another message')"))

class TestCaseError(unittest.TestCase):
    def test_caseerror(self):
        error = CaseError('a message', {'a':1, 'b':2})

        self.assertEqual(str(error), "a message case: {a:1, b:2, policy:None}")
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()