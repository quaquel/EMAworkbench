"""


"""

import unittest
import unittest.mock as mock
import scipy as sp

from ema_workbench.em_framework import parameters
from ema_workbench.em_framework.points import Scenario, Policy
from ema_workbench.em_framework.samplers import LHSSampler
from ema_workbench.em_framework.model import Model
from ema_workbench.em_framework.util import NamedObject


    
class RealParameterTestCase(unittest.TestCase):
   
    def test_instantiation(self):
        name = 'test'
        resolution = [0, 1, 2]
        lower_bound = 0
        upper_bound = 2.1
        par = parameters.RealParameter(name, lower_bound, upper_bound, 
                                       resolution)
        
        self.assertEqual(par.name, name)
        self.assertEqual(par.resolution, resolution)
        self.assertEqual(par.lower_bound, lower_bound)
        self.assertEqual(par.upper_bound, upper_bound)
        
        name = 'test'
        resolution = [0, 1,2]
        lower_bound = 2.1
        upper_bound = 0
        
        with self.assertRaises(ValueError):
            par = parameters.RealParameter(name, lower_bound, upper_bound, 
                                           resolution)
            
        with self.assertRaises(ValueError):
            resolution = [-1, 0]
            par = parameters.RealParameter(name, lower_bound, upper_bound, 
                                                       resolution)
            
            resolution = [0, 1, 3]
            par = parameters.RealParameter(name, lower_bound, upper_bound, 
                                                       resolution)

    def test_comparison(self):
        name = 'test'
        resolution = [0, 1,2]
        lower_bound = 0
        upper_bound = 2.0
        par1 = parameters.RealParameter(name, lower_bound, upper_bound, resolution)
        par2 = parameters.RealParameter(name, lower_bound, upper_bound, resolution)
        
        self.assertTrue(par1==par2)
        
        name = 'test'
        par1 = parameters.RealParameter(name, lower_bound, upper_bound, resolution)
        
        name = 'what?'
        par2 = parameters.RealParameter(name, lower_bound, upper_bound, resolution)
        self.assertFalse(par1==par2)
    
    def test_dist(self):
        name = 'test'
        resolution = [0, 1, 2]
        lower_bound = 0
        upper_bound = 2.1
        par = parameters.RealParameter(name, lower_bound, upper_bound, resolution)
        
        self.assertEqual(par.dist.dist.name, "uniform")
        self.assertEqual(par.lower_bound, lower_bound)
        self.assertEqual(par.upper_bound, upper_bound)
        
    def test_from_dist(self):
        par = parameters.RealParameter.from_dist("test", sp.stats.uniform(0, 1), # @UndefinedVariable
                                                 resolution=[0,1])  
        
        self.assertEqual(par.name, "test")
        self.assertEqual(par.dist.dist.name, "uniform")
        self.assertEqual(par.lower_bound, 0)
        self.assertEqual(par.upper_bound, 1)    
        self.assertEqual(par.resolution, [0,1])        
        
        with self.assertRaises(ValueError):
            parameters.RealParameter.from_dist("test", sp.stats.randint(0, 1))  # @UndefinedVariable
            parameters.RealParameter.from_dist("test", sp.stats.uniform(0, 1), # @UndefinedVariable
                                                 blaat=[0,1])
    
class IntegerParameterTestCase(unittest.TestCase):
    
    def test_instantiation(self):
        name = 'test'
        resolution = [0, 1, 2]
        lower_bound = 0
        upper_bound = 2
        par = parameters.IntegerParameter(name, lower_bound, upper_bound, resolution)
        
        self.assertEqual(par.name, name)
        self.assertEqual(par.resolution, resolution)
        self.assertEqual(par.lower_bound, lower_bound)
        self.assertEqual(par.upper_bound, upper_bound)
        
        name = 'test'
        resolution = [0, 1,2]
        lower_bound = 2
        upper_bound = 0
        
        with self.assertRaises(ValueError):
            par = parameters.IntegerParameter(name, lower_bound, upper_bound, 
                                           resolution)
            
        with self.assertRaises(ValueError):
            resolution = [-1, 0]
            par = parameters.IntegerParameter(name, lower_bound, upper_bound, 
                                                       resolution)
            
            resolution = [0, 1, 3]
            par = parameters.IntegerParameter(name, lower_bound, upper_bound, 
                                                       resolution)
            
        with self.assertRaises(ValueError):
            par = parameters.IntegerParameter(name, lower_bound, 2.1, 
                                           resolution)
            
            par = parameters.IntegerParameter(name, 0.0, 2, 
                                           resolution)
            
        with self.assertRaises(ValueError):
            par = parameters.IntegerParameter(name, lower_bound, upper_bound, 
                                             [0, 1.5, 2])
    
    def test_dist(self):
        name = 'test'
        resolution = [0, 1, 2]
        lower_bound = 0
        upper_bound = 2
        par = parameters.IntegerParameter(name, lower_bound, upper_bound, 
                                          resolution)
        
        self.assertEqual(par.dist.dist.name, "randint")
        self.assertEqual(par.lower_bound, lower_bound)
        self.assertEqual(par.upper_bound, upper_bound)
        
    def test_from_dist(self):
        par = parameters.IntegerParameter.from_dist("test",
                                                    sp.stats.randint(0, 10), # @UndefinedVariable
                                                    resolution=[0,9])  
        
        self.assertEqual(par.name, "test")
        self.assertEqual(par.dist.dist.name, "randint")
        self.assertEqual(par.lower_bound, 0)
        self.assertEqual(par.upper_bound, 9)    
        self.assertEqual(par.resolution, [0,9])        
        
        with self.assertRaises(ValueError):
            parameters.IntegerParameter.from_dist("test",
                                                  sp.stats.uniform(0, 1))  # @UndefinedVariable
            parameters.IntegerParameter.from_dist("test",
                                                  sp.stats.randint(0, 1), # @UndefinedVariable
                                                  blaat=[0,1])
            parameters.IntegerParameter.from_dist("test",
                                                  sp.stats.randint(0, 10), # @UndefinedVariable
                                                  resolution=[0,9])  
        
    
    
class CategoricalParameterTestCase(unittest.TestCase):
    def test_instantiation(self):
        name = 'test'
        values = ('a', 'b')
        par = parameters.CategoricalParameter(name, values)
        
        self.assertEqual(par.name, name)
        self.assertEqual(par.resolution, [0,1])
        self.assertEqual(par.lower_bound, 0)
        self.assertEqual(par.upper_bound, 1)
    
    def test_index_for_cat(self):
        name = 'test'
        values = ('a', 'b')
        par1 = parameters.CategoricalParameter(name, values)
       
        self.assertEqual(par1.index_for_cat('a'), 0)
        self.assertEqual(par1.index_for_cat('b'), 1)
        
        with self.assertRaises(ValueError):
            par1.index_for_cat('c')
    
    def test_cat_for_indext(self):
        name = 'test'
        values = ('a', 'b')
        par1 = parameters.CategoricalParameter(name, values)
       
        self.assertEqual(par1.cat_for_index(0).name, 'a')
        self.assertEqual(par1.cat_for_index(1).name, 'b')
        
        with self.assertRaises(KeyError):
            par1.cat_for_index(3)

    def test_from_dist(self):
        pass


class ParametersTestCase(unittest.TestCase):
    @mock.patch('ema_workbench.em_framework.parameters.pd')
    def test_to_csv(self, mock_pandas):
        params = [parameters.RealParameter('a', 0.1, 1.5),
                  parameters.IntegerParameter('b', 0, 10),
                  parameters.CategoricalParameter('c', ['a', 'b'])]
        
        parameters.parameters_to_csv(params, 'a.csv')
        
        # TODO:: add assertions
        mock_pandas.DataFrame.from_dict.assert_called()


if __name__ == "__main__":
    unittest.main()