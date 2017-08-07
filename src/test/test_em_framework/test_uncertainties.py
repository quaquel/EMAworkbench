'''


'''
from __future__ import (unicode_literals, print_function, absolute_import,
                                        division)

import unittest

from ema_workbench.em_framework.uncertainties import (ParameterUncertainty, 
                                                      CategoricalUncertainty)
from ema_workbench.em_framework import (RealParameter, IntegerParameter, 
                                        CategoricalParameter)


class Test(unittest.TestCase):
    def test(self):
        values = (0, 2)
        name = 'a'
        
        a = ParameterUncertainty(values, name)
        self.assertTrue(isinstance(a, RealParameter))
        
        b = ParameterUncertainty(values, name, integer=True)
        self.assertTrue(isinstance(b, IntegerParameter))
        
        c = CategoricalUncertainty(values, name)
        self.assertTrue(isinstance(c, CategoricalParameter))     
        
if __name__ == '__main__':
    unittest.main()   