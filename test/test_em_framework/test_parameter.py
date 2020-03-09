'''


'''
from __future__ import (unicode_literals, print_function, absolute_import,
                                        division)


import unittest
import unittest.mock as mock
import scipy as sp

from ema_workbench.em_framework import parameters
from ema_workbench.em_framework.samplers import LHSSampler
from ema_workbench.em_framework.model import Model
from ema_workbench.em_framework.outcomes import create_outcomes
from ema_workbench.em_framework.util import NamedObject


    
class RealParameterTestCase(unittest.TestCase):
   
    def test_experiment_generator(self):
        sampler = LHSSampler()
        
        shared_abc_1 = parameters.RealParameter("shared ab 1", 0, 1)
        shared_abc_2 = parameters.RealParameter("shared ab 2", 0, 1)
        unique_a = parameters.RealParameter("unique a ", 0, 1)
        unique_b = parameters.RealParameter("unique b ", 0, 1)
        uncertainties = [shared_abc_1, shared_abc_2, unique_a, unique_b]
        designs  = sampler.generate_designs(uncertainties, 10)
        designs.kind = parameters.Scenario
        
        # everything shared
        model_a = Model("A", mock.Mock())
        model_b = Model("B", mock.Mock())
        
        model_a.uncertainties = [shared_abc_1, shared_abc_2, unique_a]
        model_b.uncertainties = [shared_abc_1, shared_abc_2, unique_b]
        model_structures = [model_a, model_b]
        
        policies = [parameters.Policy('policy 1'),
                    parameters.Policy('policy 2'),
                    parameters.Policy('policy 3')]
        
        gen = parameters.experiment_generator(designs, model_structures, 
                                              policies)
        
        experiments = []
        for entry in gen:
            experiments.append(entry)
        self.assertEqual(len(experiments), 2*3*10)


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

class CreateOutcomesTestCase(unittest.TestCase):
    def test_create_outcomes(self):
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

class ParametersTestCase(unittest.TestCase):
    @mock.patch('ema_workbench.em_framework.parameters.pd')
    def test_to_csv(self, mock_pandas):
        params = [parameters.RealParameter('a', 0.1, 1.5),
                  parameters.IntegerParameter('b', 0, 10),
                  parameters.CategoricalParameter('c', ['a', 'b'])]
        
        parameters.parameters_to_csv(params, 'a.csv')
        
        # TODO:: add assertions
        mock_pandas.DataFrame.from_dict.assert_called()

    def test_experiment_gemerator(self):
        scenarios = [NamedObject("scen_1"), NamedObject("scen_2")]
        model_structures = [NamedObject("model")]
        policies = [NamedObject("1"), NamedObject("2"), NamedObject("3")]

        
        experiments = parameters.experiment_generator(scenarios,
                              model_structures, policies, combine='factorial')
        experiments = [e for e in experiments]
        self.assertEqual(len(experiments), 6, ("wrong number of experiments "
                                               "for factorial"))

        
        experiments = parameters.experiment_generator(scenarios,
                                model_structures, policies, combine='zipover')
        experiments = [e for e in experiments]
        self.assertEqual(len(experiments), 3, ("wrong number of experiments "
                                               "for zipover"))


if __name__ == "__main__":
    unittest.main()