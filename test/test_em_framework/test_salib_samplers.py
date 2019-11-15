'''
embedding SALib sampling within the workbench

'''

import unittest
from ema_workbench import (RealParameter)
from ema_workbench.em_framework.salib_samplers import (get_SALib_problem,
                               SobolSampler, MorrisSampler, FASTSampler)
from ema_workbench.em_framework.parameters import IntegerParameter


# Created on 14 Mar 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = []


class SamplerTestCase(unittest.TestCase):
    
    def test_sobol(self):
        parameters = [RealParameter("a", 0, 10),
                      RealParameter("b", 0, 5)]        
        
        sampler = SobolSampler()
        samples = sampler.generate_samples(parameters, 100)
        
        N = 100 * (2*2 + 2)
        for key in ['a', 'b']:
            self.assertIn(key, samples.keys())
            self.assertEqual(samples[key].shape[0], N)

        sampler = SobolSampler(second_order=False)
        samples = sampler.generate_samples(parameters, 100)
        
        N = 100 * (2 + 2)
        for key in ['a', 'b']:
            self.assertIn(key, samples.keys())
            self.assertEqual(samples[key].shape[0], N)
            
        parameters = [RealParameter("a", 0, 10),
                      RealParameter("b", 0, 5),
                      IntegerParameter("c", 0, 2)]        
        
        sampler = SobolSampler()
        samples = sampler.generate_samples(parameters, 100)
        
        N = 100 * (2*3 + 2)
        for key in ['a', 'b']:
            self.assertIn(key, samples.keys())
            self.assertEqual(samples[key].shape[0], N)
            
        designs = sampler.generate_designs(parameters, 100)
        
        self.assertEqual(designs.parameters, parameters)
        self.assertEqual(designs.params, ['a', 'b', 'c'])
        self.assertEqual(designs.n, N)
    
    def test_morris(self):
        parameters = [RealParameter("a", 0, 10),
                      RealParameter("b", 0, 5)]        
        
        sampler = MorrisSampler()
        samples = sampler.generate_samples(parameters, 100)
        
        G = 4
        D = len(parameters)
        N = 100
        
        N = (G/D+1)*N
        for key in ['a', 'b']:
            self.assertIn(key, samples.keys())
            self.assertEqual(samples[key].shape[0], N)


    def test_FAST(self):
        parameters = [RealParameter("a", 0, 10),
                      RealParameter("b", 0, 5)]        
        
        sampler = FASTSampler()
        samples = sampler.generate_samples(parameters, 100)
        
        N = 100 * 2
        for key in ['a', 'b']:
            self.assertIn(key, samples.keys())
            self.assertEqual(samples[key].shape[0], N)
    
    def test_get_salib_problem(self):
        uncertainties = [RealParameter("a", 0, 10),
                         RealParameter("b", 0, 5),
                         IntegerParameter("c", 0, 4)]
        
        problem = get_SALib_problem(uncertainties)
        self.assertEqual(len(uncertainties), problem['num_vars'])
        self.assertEqual([u.name for u in uncertainties], problem['names'])
        
#         for i, (name, bounds) in enumerate(zip(problem['names'], problem['bounds'])):
#             self.assertEqual()
        
        self.assertEqual((0, 10), problem['bounds'][0])
        self.assertEqual((0, 5), problem['bounds'][1])
        self.assertEqual((0, 4+1), problem['bounds'][2]) # because flooring
    

if __name__ == '__main__':
    unittest.main()