'''
Created on 21 jan. 2013

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
import unittest

from ...em_framework.samplers import (LHSSampler, MonteCarloSampler, 
                                FullFactorialSampler, PartialFactorialSampler)
from ...em_framework.uncertainties import (RealParameter, IntegerParameter, 
                                           CategoricalParameter)


class SamplerTestCase(unittest.TestCase):
    uncertainties = [RealParameter("1", 0, 10),
                     IntegerParameter("2", 0, 10),
                     CategoricalParameter('3', ['a','b', 'c'])]

    def _test_generate_designs(self, sampler):
        designs, nr_designs = sampler.generate_designs(self.uncertainties, 10)
        msg = 'tested for {}'.format(type(sampler))
        
        actual_nr_designs = 0
        for design in designs:
            actual_nr_designs +=1
            
        self.assertIn('1', design, msg)
        self.assertIn('2', design, msg)
        self.assertIn('3', design, msg)
        self.assertEqual(nr_designs, actual_nr_designs, msg) 
    
    def test_lhs_sampler(self):
        sampler = LHSSampler()
        self._test_generate_designs(sampler)
     
    def test_mc_sampler(self):
        sampler = MonteCarloSampler()
        self._test_generate_designs(sampler)
    
    def test_ff_sampler(self):
        sampler = FullFactorialSampler()
        self._test_generate_designs(sampler)
        
    def test_pf_sampler(self):
        uncs = [RealParameter('a', 0, 5, resolution=(0, 2.5,5)),
                RealParameter('b', 0, 1, resolution=(0,1)),
                RealParameter('c', 0, 1),
                RealParameter('d', 1, 2),
                ]

        sampler = PartialFactorialSampler()
        designs, nr_designs = sampler.generate_designs(uncs, 10, ['a', 'b'])
        
        expected = 60
        self.assertEqual(expected, nr_designs)
        
        self.assertEqual(expected, len([design for design in designs]))
        
        ff, other = sampler._sort_parameters(uncs, ['a', 'b'])
        
        received = {u.name for u in ff}
        expected = {'a', 'b'}
        self.assertEqual(received, expected)
        
        received = {u.name for u in other}
        expected = {'c', 'd'}
        self.assertEqual(received, expected)
        

if __name__ == "__main__":
    unittest.main()