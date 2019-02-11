'''
Created on 21 jan. 2013

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import (absolute_import, unicode_literals, division, 
                        print_function)

import unittest.mock as mock
import unittest

from ema_workbench.em_framework.samplers import (LHSSampler, MonteCarloSampler,
                                FullFactorialSampler, PartialFactorialSampler,
                                determine_parameters, UniformLHSSampler)
from ema_workbench.em_framework.parameters import (RealParameter, 
                                                      IntegerParameter, 
                                                      CategoricalParameter,
                                                   BooleanParameter)
from ema_workbench.em_framework.parameters import Scenario
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

    def test_lhs_sampler_with_distribution(self):
        sampler = LHSSampler()

        uncertainties = [RealParameter("1", 0, 10),
                         IntegerParameter("2", 0, 10),
                         CategoricalParameter('3', ['a', 'b', 'c']),
                         RealParameter("4", 0, 10, dist=RealParameter.TRIANGLE, dist_params=[0.95]),
                         RealParameter("5", 0, 10, dist=RealParameter.PERT, dist_params=[0.05, 4.0]),
                         BooleanParameter("6"),
                         BooleanParameter("7", dist=BooleanParameter.BERNOULLI, dist_params=[0.1, ]),
                         ]

        designs = sampler.generate_designs(uncertainties, 100)
        designs.kind = Scenario
        msg = 'tested for {}'.format(type(sampler))

        checksum4 = sum(d['4'] for d in designs)
        checksum5 = sum(d['5'] for d in designs)
        checksum6 = sum(d['6'] for d in designs)
        checksum7 = sum(d['7'] for d in designs)
        self.assertAlmostEqual(checksum4, 650, delta=5)
        self.assertAlmostEqual(checksum5, 200, delta=5)
        self.assertAlmostEqual(checksum6, 50, delta=5)
        self.assertAlmostEqual(checksum7, 10, delta=5)

        actual_nr_designs = 0
        for design in designs:
            actual_nr_designs += 1

        self.assertIn('1', design, msg)
        self.assertIn('2', design, msg)
        self.assertIn('3', design, msg)
        self.assertIn('4', design, msg)
        self.assertIn('5', design, msg)
        self.assertIn('6', design, msg)
        self.assertIn('7', design, msg)
        self.assertEqual(designs.n, actual_nr_designs, msg)

    def test_ulhs_sampler(self):
        sampler = UniformLHSSampler()

        uncertainties = [RealParameter("1", 0, 10),
                         IntegerParameter("2", 0, 10),
                         CategoricalParameter('3', ['a', 'b', 'c']),
                         RealParameter("4", 0, 10, dist=RealParameter.TRIANGLE, dist_params=[0.95]),
                         RealParameter("5", 0, 10, dist=RealParameter.PERT, dist_params=[0.05, 4.0]),
                         BooleanParameter("6"),
                         BooleanParameter("7", dist=BooleanParameter.BERNOULLI, dist_params=[0.1, ]),
                         ]

        designs = sampler.generate_designs(uncertainties, 100)
        designs.kind = Scenario
        msg = 'tested for {}'.format(type(sampler))

        checksum4 = sum(d['4'] for d in designs)
        checksum5 = sum(d['5'] for d in designs)
        checksum6 = sum(d['6'] for d in designs)
        checksum7 = sum(d['7'] for d in designs)
        self.assertAlmostEqual(checksum4, 500, delta=5)
        self.assertAlmostEqual(checksum5, 500, delta=5)
        self.assertAlmostEqual(checksum6, 50, delta=5)
        self.assertAlmostEqual(checksum7, 50, delta=5)

        actual_nr_designs = 0
        for design in designs:
            actual_nr_designs += 1

        self.assertIn('1', design, msg)
        self.assertIn('2', design, msg)
        self.assertIn('3', design, msg)
        self.assertIn('4', design, msg)
        self.assertIn('5', design, msg)
        self.assertIn('6', design, msg)
        self.assertIn('7', design, msg)
        self.assertEqual(designs.n, actual_nr_designs, msg)

    def test_mc_sampler(self):
        sampler = MonteCarloSampler()
        self._test_generate_designs(sampler)
    
    def test_ff_sampler(self):
        sampler = FullFactorialSampler()
        self._test_generate_designs(sampler)
        
    def test_pf_sampler(self):
        uncs = [RealParameter('a', 0, 5, resolution=(0, 2.5,5), pff=True),
                RealParameter('b', 0, 1, resolution=(0,1), pff=True),
                RealParameter('c', 0, 1),
                RealParameter('d', 1, 2),
                ]

        sampler = PartialFactorialSampler()
        designs = sampler.generate_designs(uncs, 10)
        designs.kind = Scenario
        
        expected = 60
        self.assertEqual(expected, designs.n)
        
        self.assertEqual(expected, len([design for design in designs]))
        
        ff, other = sampler._sort_parameters(uncs)
        
        received = {u.name for u in ff}
        expected = {'a', 'b'}
        self.assertEqual(received, expected)
        
        received = {u.name for u in other}
        expected = {'c', 'd'}
        self.assertEqual(received, expected)
 
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


if __name__ == "__main__":
    unittest.main()