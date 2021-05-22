"""
Created on 21 jan. 2013

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
"""
import unittest.mock as mock
import unittest

from ema_workbench.em_framework.samplers import (LHSSampler, MonteCarloSampler, 
                                FullFactorialSampler, determine_parameters)
from ema_workbench.em_framework.parameters import (RealParameter,
                                       IntegerParameter, CategoricalParameter)
from ema_workbench.em_framework.points import Scenario
from ema_workbench.em_framework import Model


class SamplerTestCase(unittest.TestCase):
    uncertainties = [RealParameter("1", 0, 10),
                     IntegerParameter("2", 0, 10),
                     CategoricalParameter('3', ['a','b', 'c'])]

    def _test_generate_designs(self, sampler):
        designs = sampler.generate_designs(self.uncertainties, 10)
        designs.kind = Scenario
        msg = 'tested for {}'.format(type(sampler))
        
        actual_nr_designs = 0
        for design in designs:
            actual_nr_designs +=1
            
        self.assertIn('1', design, msg)
        self.assertIn('2', design, msg)
        self.assertIn('3', design, msg)
        self.assertEqual(designs.n, actual_nr_designs, msg) 
    
    def test_lhs_sampler(self):
        sampler = LHSSampler()
        self._test_generate_designs(sampler)
     
    def test_mc_sampler(self):
        sampler = MonteCarloSampler()
        self._test_generate_designs(sampler)
    
    def test_ff_sampler(self):
        sampler = FullFactorialSampler()
        self._test_generate_designs(sampler)
        
    # def test_pf_sampler(self):
    #     uncs = [RealParameter('a', 0, 5, resolution=(0, 2.5,5), pff=True),
    #             RealParameter('b', 0, 1, resolution=(0,1), pff=True),
    #             RealParameter('c', 0, 1),
    #             RealParameter('d', 1, 2),
    #             ]
    #
    #     sampler = PartialFactorialSampler()
    #     designs = sampler.generate_designs(uncs, 10)
    #     designs.kind = Scenario
    #
    #     expected = 60
    #     self.assertEqual(expected, designs.n)
    #
    #     self.assertEqual(expected, len([design for design in designs]))
    #
    #     ff, other = sampler._sort_parameters(uncs)
    #
    #     received = {u.name for u in ff}
    #     expected = {'a', 'b'}
    #     self.assertEqual(received, expected)
    #
    #     received = {u.name for u in other}
    #     expected = {'c', 'd'}
    #     self.assertEqual(received, expected)
 
    def test_determine_parameters(self):
        function = mock.Mock()
        model_a = Model("A", function)
        model_a.uncertainties = [RealParameter('a', 0, 1),
                                 RealParameter('b', 0, 1),]
        function = mock.Mock()
        model_b = Model("B", function)
        model_b.uncertainties = [RealParameter('b', 0, 1),
                                 RealParameter('c', 0, 1),]
        
        models = [model_a, model_b]
        
        parameters = determine_parameters(models, 'uncertainties', union=True)
        for model in models:
            for unc in model.uncertainties:
                self.assertIn(unc.name, parameters.keys())
        
        parameters = determine_parameters(models, 'uncertainties', union=False)
        self.assertIn('b', parameters.keys())
        self.assertNotIn('c', parameters.keys())
        self.assertNotIn('a', parameters.keys())  
        
    # def test_sample_jointly(self):
    #     function = mock.Mock()
    #     model = Model("A", function)
    #     model.uncertainties = [RealParameter('a', 0, 1),
    #                            RealParameter('c', 0, 1),]
    #     model.levers = [RealParameter('b', 0, 1),
    #                     RealParameter('d', 0, 1),]
    #
    #     designs = sample_jointly(model, 10)
    #     self.assertEqual(designs.n, 10)
        

if __name__ == "__main__":
    unittest.main()