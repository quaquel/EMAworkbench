'''
Created on 18 jan. 2013

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''

import unittest
from em_framework import ParameterUncertainty, CategoricalUncertainty
from em_framework.uncertainties import INTEGER, UNIFORM

class UncertaintyTestCase(unittest.TestCase):

    def test_init(self):
        
        values = (0, 1)
        name = "test"
        integer = False
        unc = ParameterUncertainty(values, name, integer=integer)
        self.assertEqual(values, unc.values)
        self.assertEqual(name, unc.name)
        self.assertEqual(name, str(unc))
        self.assertEqual(UNIFORM, unc.dist)
        
        values = (0, 1)
        name = "test"
        integer = True
        unc = ParameterUncertainty(values, name, integer=integer)
        self.assertEqual(values, unc.values)
        self.assertEqual(name, unc.name)
        self.assertEqual(INTEGER, unc.dist)
        
        
        values = ('a', 'b',  'c')
        name = "test"
        unc = CategoricalUncertainty(values, name)

        self.assertEqual(values, unc.categories)
        self.assertEqual((0, len(values)-1), unc.values)
        self.assertEqual(name, unc.name)
        self.assertEqual(INTEGER, unc.dist)

    def test_parameter_uncertainty(self):
        values = (1, 0)
        name = "test"
        self.assertRaises(ValueError, ParameterUncertainty, values, name)
        
        values = (0,1,2)
        name = "test"
        self.assertRaises(ValueError, ParameterUncertainty, values, name)

    def test_categorical_unc(self):
        values = ('a', 'b',  'c')
        name = "test"
        unc = CategoricalUncertainty(values, name)
        
        self.assertEqual('a', unc.transform(0))
        self.assertEqual(0, unc.invert('a'))
        
        self.assertRaises(IndexError, unc.transform, 3)
        self.assertRaises(ValueError, unc.invert, 'd')
        

    def test_uncertainty_identity(self):
        # what are the test cases
        
        # uncertainties are the same
        # let's add some uncertainties to this
        shared_ab_1 = ParameterUncertainty((0,10), "shared ab 1")
        shared_ab_2 = ParameterUncertainty((0,10), "shared ab 1")
    
        self.assertTrue(shared_ab_1 == shared_ab_2)
        self.assertTrue(shared_ab_2 == shared_ab_1)
        
        # uncertainties are not the same
        shared_ab_1 = ParameterUncertainty((0,10), "shared ab 1")
        shared_ab_2 = ParameterUncertainty((0,10), "shared ab 1", integer=True)
    
        self.assertFalse(shared_ab_1 == shared_ab_2)
        self.assertFalse(shared_ab_2 == shared_ab_1)
        
        # uncertainties are of different classes
        # what should happen then?
        # in principle it should return false, but what if the classes are
        # different but the __dicts__ are the same? This would be lousy coding, 
        # but should it concern us here?
        shared_ab_1 = ParameterUncertainty((0,10), "shared ab 1")
        shared_ab_2 = CategoricalUncertainty([x for x in range(11)], "shared ab 1")
        
        self.assertFalse(shared_ab_1 == shared_ab_2)
        self.assertFalse(shared_ab_2 == shared_ab_1)

if __name__ == "__main__":
    unittest.main()
    
