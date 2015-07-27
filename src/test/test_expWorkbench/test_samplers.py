'''
Created on 21 jan. 2013

@author: localadmin
'''

import unittest


from expWorkbench import LHSSampler, MonteCarloSampler, FullFactorialSampler
from expWorkbench import ParameterUncertainty, CategoricalUncertainty

class SamplerTestCase(unittest.TestCase):
    uncertainties = [ParameterUncertainty((0,10), "1"),
                     ParameterUncertainty((0,10), "2", integer=True),
                     CategoricalUncertainty(['a','b', 'c'], "3")]

    def _test_generate_designs(self, sampler):
        designs, nr_designs = sampler.generate_designs(self.uncertainties, 10)
        msg = 'tested for {}'.format(type(sampler))
        
        actual_nr_designs = 0
        for design in designs:
            actual_nr_designs +=1
            
        self.assertIn('1', design, msg)
        self.assertIn('2', design, msg)
        self.assertIn('3', design, msg)
        self.assertEqual(nr_designs, actual_nr_designs, ) 
    
    def test_lhs_sampler(self):
        sampler = LHSSampler()
        self._test_generate_designs(sampler)
     
    def test_mc_sampler(self):
        sampler = MonteCarloSampler()
        self._test_generate_designs(sampler)
    
    def test_ff_sampler(self):
        sampler = FullFactorialSampler()
        self._test_generate_designs(sampler)

if __name__ == "__main__":
    unittest.main()