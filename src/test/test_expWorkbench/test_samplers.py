'''
Created on 21 jan. 2013

@author: localadmin
'''
from expWorkbench import LHSSampler, MonteCarloSampler, FullFactorialSampler
from expWorkbench import ParameterUncertainty, CategoricalUncertainty

def test_generate_samples(sampler):
    unc_1 = ParameterUncertainty((0,10), "1")
    unc_2 = ParameterUncertainty((0,10), "2", integer=True)
    unc_3 = CategoricalUncertainty(['a','b', 'c'], "3")
    uncertainties = [unc_1, unc_2, unc_3]
    
    return sampler.generate_samples(uncertainties, 10)

def test_generate_designs(sampler):
    unc = test_generate_samples(sampler)
    unc_samples = [unc[key] for key in sorted(unc.keys())]
    return sampler.generate_designs(unc_samples) 

def test_sampler(sampler):
    samples = test_generate_samples(sampler)

    for key in sorted(samples.keys()):
        print key, samples[key]
        
    designs = test_generate_designs(sampler)
    temp_designs = set()
    for entry in designs:
        print entry
        temp_designs.add(entry)
    print sampler.deterimine_nr_of_designs(samples)
    print len(temp_designs)

def test_lhs_samper():
    sampler = LHSSampler()
    test_sampler(sampler)

def test_mc_sampler():
    sampler = MonteCarloSampler()
    test_sampler(sampler)

def test_ff_sampler():
    sampler = FullFactorialSampler()
    test_sampler(sampler)

if __name__ == "__main__":
#     test_lhs_samper()
#     test_mc_sampler()
    test_ff_sampler()