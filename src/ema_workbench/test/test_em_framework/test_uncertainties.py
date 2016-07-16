'''


'''
from __future__ import (unicode_literals, print_function, absolute_import,
                                        division)

from ema_workbench.em_framework import uncertainties

import unittest
    
class RealUncertaintyTestCase(unittest.TestCase):
    def test_instantiation(self):
        name = 'test'
        resolution = [0, 1, 2]
        lower_bound = 0
        upper_bound = 2.1
        par = uncertainties.RealUncertainty(name, lower_bound, upper_bound, resolution)
        
        self.assertEqual(par.name, name)
        self.assertEqual(par.resolution, resolution)
        self.assertEqual(par.lower_bound, lower_bound)
        self.assertEqual(par.upper_bound, upper_bound)
        
        name = 'test'
        resolution = [0, 1,2]
        lower_bound = 2.1
        upper_bound = 0
        
        with self.assertRaises(ValueError):
            par = uncertainties.RealParameter(name, lower_bound, upper_bound, 
                                           resolution)
            
        with self.assertRaises(ValueError):
            resolution = [-1, 0]
            par = uncertainties.RealParameter(name, lower_bound, upper_bound, 
                                                       resolution)
            
            resolution = [0, 1, 3]
            par = uncertainties.RealParameter(name, lower_bound, upper_bound, 
                                                       resolution)

    def test_comparison(self):
        name = 'test'
        resolution = [0, 1,2]
        lower_bound = 0
        upper_bound = 2.0
        par1 = uncertainties.RealUncertainty(name, lower_bound, upper_bound, resolution)
        par2 = uncertainties.RealUncertainty(name, lower_bound, upper_bound, resolution)
        
        self.assertTrue(par1==par2)
        
        name = 'test'
        par1 = uncertainties.RealUncertainty(name, lower_bound, upper_bound, resolution)
        
        name = 'what?'
        par2 = uncertainties.RealUncertainty(name, lower_bound, upper_bound, resolution)
        self.assertFalse(par1==par2)
    
    def test_params(self):
        name = 'test'
        resolution = [0, 1, 2]
        lower_bound = 0
        upper_bound = 2.1
        par = uncertainties.RealUncertainty(name, lower_bound, upper_bound, resolution)
        
        self.assertEqual(par.params, (0, 2.1))
    
    
class IntegerUncertaintyTestCase(unittest.TestCase):
    
    def test_instantiation(self):
        name = 'test'
        resolution = [0, 1, 2]
        lower_bound = 0
        upper_bound = 2
        par = uncertainties.IntegerUncertainty(name, lower_bound, upper_bound, resolution)
        
        self.assertEqual(par.name, name)
        self.assertEqual(par.resolution, resolution)
        self.assertEqual(par.lower_bound, lower_bound)
        self.assertEqual(par.upper_bound, upper_bound)
        
        name = 'test'
        resolution = [0, 1,2]
        lower_bound = 2
        upper_bound = 0
        
        with self.assertRaises(ValueError):
            par = uncertainties.IntegerUncertainty(name, lower_bound, upper_bound, 
                                           resolution)
            
        with self.assertRaises(ValueError):
            resolution = [-1, 0]
            par = uncertainties.IntegerUncertainty(name, lower_bound, upper_bound, 
                                                       resolution)
            
            resolution = [0, 1, 3]
            par = uncertainties.IntegerUncertainty(name, lower_bound, upper_bound, 
                                                       resolution)
            
        with self.assertRaises(ValueError):
            par = uncertainties.IntegerUncertainty(name, lower_bound, 2.1, 
                                           resolution)
            
            par = uncertainties.IntegerUncertainty(name, 0.0, 2, 
                                           resolution)
            
        with self.assertRaises(ValueError):
            par = uncertainties.IntegerUncertainty(name, lower_bound, upper_bound, 
                                             [0, 1.5, 2])
    
    def test_params(self):
        name = 'test'
        resolution = [0, 1, 2]
        lower_bound = 0
        upper_bound = 2
        par = uncertainties.IntegerUncertainty(name, lower_bound, upper_bound, 
                                          resolution)
        
        self.assertEqual(par.params, (0, 2))
    
    
class CategoricalUncertaintyTestCase(unittest.TestCase):
    def test_instantiation(self):
        name = 'test'
        values = ('a', 'b')
        par = uncertainties.CategoricalUncertainty(name, values)
        
        self.assertEqual(par.name, name)
        self.assertEqual(par.resolution, list(values))
        self.assertEqual(par.lower_bound, 0)
        self.assertEqual(par.upper_bound, 2)
    
    def test_comparison(self):
        name = 'test'
        values = ('a', 'b')
        par1 = uncertainties.CategoricalUncertainty(name, values)
        par2 = uncertainties.CategoricalUncertainty(name, values)
        
        self.assertEqual(par1, par2)

        name = 'what'
        par2 = uncertainties.CategoricalUncertainty(name, values)
        self.assertNotEqual(par1, par2)
    
    def test_index_for_cat(self):
        name = 'test'
        values = ('a', 'b')
        par1 = uncertainties.CategoricalUncertainty(name, values)
       
        self.assertEqual(par1.index_for_cat('a'), 0)
        self.assertEqual(par1.index_for_cat('b'), 1)
        
        with self.assertRaises(ValueError):
            par1.index_for_cat('c')
    
    def test_cat_for_indext(self):
        name = 'test'
        values = ('a', 'b')
        par1 = uncertainties.CategoricalUncertainty(name, values)
       
        self.assertEqual(par1.cat_for_index(0), 'a')
        self.assertEqual(par1.cat_for_index(1), 'b')
        
        with self.assertRaises(IndexError):
            par1.cat_for_index(3)

    
if __name__ == "__main__":
    unittest.main()