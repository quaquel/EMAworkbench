'''
Samplers for working with SALib

'''
from __future__ import (unicode_literals, print_function, absolute_import,
                        division)



# Created on 12 Jan 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ["SobolSampler", "MorrisSampler", "FASTSampler", 'get_SALib_problem']

from SALib.sample import saltelli, morris, fast_sampler

import operator

from .samplers import DefaultDesigns
from .parameters import CategoricalParameter, IntegerParameter


def get_SALib_problem(uncertainties):
    '''returns a dict with a problem specificatin as required by SALib'''
    
    _warning = False
    uncertainties = sorted(uncertainties, key=operator.attrgetter('name'))
    bounds = []
    
    for uncertainty in uncertainties:
        values = uncertainty.lower_bound, uncertainty.upper_bound
        bounds.append(values)

    problem = {'num_vars': len(uncertainties),
               'names': [unc.name for unc in uncertainties],
               'bounds': bounds}
    return problem

class SALibSampler(object):

    def generate_samples(self, uncertainties, size):
        '''
        The main method of :class: `~sampler.Sampler` and its 
        children. This will call the sample method for each of the 
        uncertainties and return the resulting designs. 
        
        Parameters
        ----------
        uncertainties : collection
                        a collection of :class:`~uncertainties.ParameterUncertainty` 
                        and :class:`~uncertainties.CategoricalUncertainty`
                        instances.
        size : int
               the number of samples to generate.
        
        
        Returns
        -------
        dict
            dict with the uncertainty.name as key, and the sample as value
        
        '''
        
        problem = get_SALib_problem(uncertainties)
        samples = self.sample(problem, size)
        samples = {unc.name:samples[:,i] for i, unc in enumerate(uncertainties)}
        
        # handle integer and categorical uncertainties
        for uncertainty in uncertainties:
            sample = samples[uncertainty.name]
            
            if isinstance(uncertainty, CategoricalParameter):
                sample = [uncertainty.cat_for_index(int(entry)) for entry in 
                          sample]
            elif isinstance(uncertainty, IntegerParameter):
                sample = (int(entry) for entry in sample)

            samples[uncertainty.name] = sample
        
        return samples
    
    def generate_designs(self,  parameters, nr_samples):
        '''external interface to sampler. Returns the computational experiments
        over the specified parameters, for the given number of samples for each
        parameter.
        
        Parameters
        ----------
        parameters : list 
                        a list of parameters for which to generate the
                        experimental designs
        nr_samples : int
                     the number of samples to draw for each parameter
        
        
        Returns
        -------
        generator
            a generator object that yields the designs resulting from
            combining the parameters
        int
            the number of experimental designs
        
        '''
        parameters = sorted(parameters, key=operator.attrgetter('name'))
        sampled_parameters = self.generate_samples(parameters, nr_samples)
        
        nr_designs = next(iter(sampled_parameters.values())).shape[0]

        params = sorted(sampled_parameters.keys())
        designs = zip(*[sampled_parameters[u] for u in params]) 
        designs = DefaultDesigns(designs, parameters, nr_designs)
        
        return designs

class SobolSampler(SALibSampler):
    '''Sampler generating a Sobol design using SALib
    
    Parameters
    ----------
    second_order : bool, optional
                   indicates whether second order effects should be included
    
    '''
    
    
    def __init__(self, second_order=True):
        self.second_order = second_order
        self._warning = False
        
        super(SobolSampler, self).__init__()
    
    def sample(self, problem, size):
        return saltelli.sample(problem, size, 
                                  calc_second_order=self.second_order)
    
    

class MorrisSampler(SALibSampler):
    '''Sampler generating a morris design using SALib
    
    Parameters
    ----------
    num_levels : int
        The number of grid levels
    grid_jump : int
        The grid jump size
    optimal_trajectories : int, optional
        The number of optimal trajectories to sample (between 2 and N)
    local_optimization : bool, optional
        Flag whether to use local optimization according to Ruano et al. (2012) 
        Speeds up the process tremendously for bigger N and num_levels.
        Stating this variable to be true causes the function to ignore gurobi.
    '''

    
    def __init__(self, num_levels, grid_jump, optimal_trajectories=None, 
                 local_optimization=False):
        super(MorrisSampler, self).__init__()
        self.num_levels = num_levels
        self.grid_jump = grid_jump
        self.optimal_trajectories = optimal_trajectories
        self.local_optimization = local_optimization
        

    def sample(self, problem, size):
        return morris.sample(problem, size, self.num_levels, self.grid_jump, 
                         self.optimal_trajectories, self.local_optimization)   
        
class FASTSampler(SALibSampler):
    '''Sampler generating a Fourier Amplitude Sensitivity Test (FAST) using 
    SALib
    
    Parameters
    ----------
    n : int
        The number of samples to generate
    m : int (default: 4)
        The interference parameter, i.e., the number of harmonics to sum in the
        Fourier series decomposition 
    '''

    
    def __init__(self, n, m=4):
        super(MorrisSampler, self).__init__()
        self.n = n
        self.m = m

    def sample(self, problem, size):
        return fast_sampler(self.n, self.m)  