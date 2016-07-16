'''

This module contains various classes that can be used for specifying different
types of samplers. These different samplers implement basic sampling 
techniques including Full Factorial sampling, Latin Hypercube sampling, and
Monte Carlo sampling.

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
from ema_workbench.em_framework.parameters import CategoricalParameter,\
    IntegerParameter

try:
    from future_builtins import zip
except ImportError:
    try:
        from itertools import izip as zip # < 2.5 or 3.x
    except ImportError:
        pass

import abc
import itertools
import numpy as np 
import scipy.stats as stats

# from .uncertainties import CategoricalUncertainty

# Created on 16 aug. 2011
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['AbstractSampler',
           'LHSSampler',
           'MonteCarloSampler',
           'FullFactorialSampler',
           'PartialFactorialSampler']

class AbstractSampler(object):
    '''
    Abstract base class from which different samplers can be derived. 
    
    In the simplest cases, only the sample method needs to be overwritten. 
    generate_designs` is the only method called by the ensemble class. The 
    other methods are used internally to generate the designs. 
    
    
    '''
    __metaaclass__ = abc.ABCMeta
    
    # types of distributions known by the sampler.
    # by default it knows the `uniform continuous <http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html>`_
    # distribution for sampling floats, and the `uniform discrete <http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.randint.html#scipy.stats.randint>`_
    # distribution for sampling integers. 
    distributions = {"uniform" : stats.uniform,
                     "integer" : stats.randint
                              }
    
    def __init__(self):
        super(AbstractSampler, self).__init__()

    @abc.abstractmethod
    def sample(self, distribution, params, size):
        '''
        method for sampling a number of samples from a particular distribution.
        The various samplers differ with respect to their implementation
        of this method. 
        
        Parameters
        ----------
        distribution : {'uniform', 'integer'} 
                       the distribution to sample from
        params : tuple
                 the parameters specifying the distribution
        size : int 
               the number of samples to generate
        
        Returns
        -------
        numpy array
            the samples for the distribution and specified parameters
        
        
        '''
    
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
        
        samples = {}

        for uncertainty in uncertainties:
            #range in uncertainty gives lower and upper bound
            sample = self.sample(uncertainty.dist, uncertainty.params, size) 
            
            if type(uncertainty) == CategoricalParameter:
                # TODO look into numpy ufunc
                sample = [uncertainty.transform(int(entry)) for entry in sample]
            elif uncertainty.dist=='integer':
                sample = [int(entry) for entry in sample]
            
            samples[uncertainty.name] = sample
        
        return samples

    def generate_designs(self,  uncertainties, nr_samples):
        '''external interface to sampler. Returns the computational experiments
        over the specified uncertainties, for the given number of
        samples for each uncertainty.
        
        Parameters
        ----------
        uncertainties : list 
                        a list of uncertainties for which to generate the
                        experimental designs
        nr_samples : int
                     the number of samples to draw for each uncertain factor
        
        
        Returns
        -------
        generator
            a generator object that yields the designs resulting from
            combining the uncertainties
        int
            the number of experimental designs
        
        '''
        
        sampled_uncertainties = self.generate_samples(uncertainties, nr_samples)
        uncs = sorted(sampled_uncertainties.keys())
        designs = DefaultDesigns(sampled_uncertainties, uncs)
        return designs, self.determine_nr_of_designs(sampled_uncertainties)

    def determine_nr_of_designs(self, sampled_uncertainties):
        '''
        Helper function for determining the number of experiments that will
        be generated given the sampled uncertainties.
        
        Parameters
        ----------
        sampled_uncertainties : list 
                        a list of sampled uncertainties, Typically,
                        this will be the values of the dict returned by
                        :meth:`generate_samples`. 
        
        Returns
        -------
        int
            the total number of experimental design
        
        '''
        
        return len(next(iter(sampled_uncertainties.values())))


class LHSSampler(AbstractSampler):
    """
    generates a Latin Hypercube sample for each of the uncertainties
    in case of categorical uncertainties, it handles the transform as well
    """
    
    def __init__(self):
        super(LHSSampler, self).__init__()
    
    def sample(self, distribution, params, size):
        '''
        generate a Latin Hypercube Sample.
        
        Parameters
        ----------
        distribution : scipy distribution
                       the distribution to sample from
        params : tuple
                 the parameters specifying the distribution
        size : int
               the number of samples to generate
    
        Returns
        -------
        dict 
            with the uncertainty.name as key, and the sample as value
    
        '''
        
        return self._lhs(self.distributions[distribution], params, size)

    def _lhs(self, dist, parms, siz):
        '''
        Latin Hypercube sampling of any distribution.
    
        Parameters
        ----------
        dist : random number generator from `scipy.stats <http://docs.scipy.org/doc/scipy/reference/stats.html>`_
        parms : tuple
                tuple of parameters as required for dist.
        siz : int 
              number of samples
    
        '''
        perc = np.arange(0,1.,1./siz)
        np.random.shuffle(perc)
        smp = stats.uniform(perc,1./siz).rvs() 
        v = dist(*parms).ppf(smp)
        return v
    
        
class MonteCarloSampler(AbstractSampler):
    """
    generates a Monte Carlo sample for each of the uncertainties. In case of a 
    Categorical Uncertainty it also handles the transform
    """
    
    def __init__(self):
        super(MonteCarloSampler, self).__init__()
        
    def sample(self, distribution, params, size):
        '''
        generate a Monte Carlo Sample.
        
        Parameters
        ----------
        distribution : scipy distribution
                       the distribution to sample from
        params : ti[;e
                 the parameters specifying the distribution
        size : int
               the number of samples to generate
        
        Returns
        -------
        dict 
            with the uncertainty.name as key, and the sample as value
               
        '''
        
        return self.distributions[distribution](*params).rvs(size)


class FullFactorialSampler(AbstractSampler):     
    '''
    generates a full factorial sample.
    
    If the uncertainty is non categorical, the resolution is set the 
    number of samples. If the uncertainty is categorical, the specified value 
    for samples will be ignored and each category will be used instead. 
    
    '''
    
    def __init__(self):
        super(FullFactorialSampler, self).__init__()
       
    def generate_samples(self,  uncertainties, size):
        '''
        The main method of :class: `~sampler.Sampler` and its 
        children. This will call the sample method for each of the 
        uncertainties and return the resulting samples 
        
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
            with the uncertainty.name as key, and the sample as value
        '''
        samples = {}
        for uncertainty in uncertainties:
            cats = uncertainty.resolution
            if not cats:
                cats = np.linspace(uncertainty.lower_bound, 
                                   uncertainty.upper_bound, 
                                   size)
                if isinstance(uncertainty, IntegerParameter):
                    cats = np.round(cats, 0)
                    cats = set(cats)
                    cats = [int(entry) for entry in cats]
                    cats = sorted(cats)
            samples[uncertainty.name] = cats
        
        return samples
       
    def generate_designs(self,  uncertainties, nr_samples):
        '''
        This method provides an alternative implementation to the default 
        implementation provided by :class:`~sampler.Sampler`. This
        version returns a full factorial design across the uncertainties. 
        
        Parameters
        ----------
        uncertainties : list 
                        a list of uncertainties for which to generate the
                        experimental designs
        nr_samples : int
                     the number of intervals to use on each
                     ParameterUncertainty. Categorical uncertainties always
                     return all their categories
        
        
        Returns
        -------
        generator
            a generator object that yields the designs resulting from
            combining the uncertainties
        int
            the number of experimental designs
        
        '''
        sampled_uncertainties = self.generate_samples(uncertainties, nr_samples)
        uncs = sorted(sampled_uncertainties.keys())
        designs = FullFactorialDesigns(sampled_uncertainties, uncs)
        return designs, self.determine_nr_of_designs(sampled_uncertainties)

    def determine_nr_of_designs(self, sampled_uncertainties):
        '''
        Helper function for determining the number of experiments that will
        be generated given the sampled uncertainties.
        
        Parameters
        ----------
        sampled_uncertainties : list 
                        a list of sampled uncertainties, as 
                        the values return by generate_samples
        
        Returns
        -------
        int
            the total number of experimental design
        '''
        nr_designs = 1
        for value in sampled_uncertainties.values():
            nr_designs *= len(value)
        return nr_designs


class PartialFactorialSampler(AbstractSampler):
    """
    generates a partial factorial design over the uncertainties. Any
    uncertainty where factorial is true will be included in a factorial design, 
    while the remainder will be sampled using LHS or MC sampling
    
    Parameters
    ----------
    sampling: {'LHS', 'MC'}, optional
              the desired sampling for the non factorial uncertainties.
              
    
    """
    
    def __init__(self, sampling='LHS'):
        super(PartialFactorialSampler, self).__init__()
        
        if sampling=='LHS':
            self.sampler = LHSSampler()
        elif sampling == 'MC':
            self.sampler = MonteCarloSampler()
        else:
            raise ValueError('invalid value for sampling type, should be LHS or MC')
        self.ff = FullFactorialSampler()
    
    def _sort_uncertainties(self, uncertainties):
        '''sort uncertainties into full factorial and other
        
        Parameters
        ----------
        uncertainties : list of uncertainties
        
        
        '''
        
        ff_uncs = []
        other_uncs = []
        for unc in uncertainties:
            if unc.factorial == True:
                ff_uncs.append(unc)
            else:
                other_uncs.append(unc)
                
        return ff_uncs, other_uncs
        
    
    def generate_designs(self,  uncertainties, nr_samples):
        '''external interface to sampler. Returns the computational experiments
        over the specified uncertainties, for the given number of
        samples for each uncertainty.
        
        Parameters
        ----------
        uncertainties : list 
                        a list of uncertainties for which to generate the
                        experimental designs
        nr_samples : int
                     the number of samples to draw for each uncertain factor
        
        
        Returns
        -------
        generator
            a generator object that yields the designs resulting from
            combining the uncertainties
        int
            the number of experimental designs
        
        '''
        
        ff_uncs, other_uncs = self._sort_uncertainties(uncertainties)
        
        # generate a design over the factorials
        # TODO update ff to use resolution if present
        ff_designs, n_ff = self.ff.generate_designs(ff_uncs, nr_samples)
        
        # generate a design over the remainder
        # for each factorial, run the MC design
        other_designs, n_other = self.sampler.generate_designs(other_uncs, 
                                                              nr_samples)
        
        nr_designs = n_other * n_ff
        
        designs = PartialFactorialDesigns(ff_designs, other_designs)
        
        return designs, nr_designs
   
   
class AbstractDesignsIterable(object):
    '''iterable for the experimental designs'''
    
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, sampled_uncs, uncertainties):
        self.sampled_uncs = sampled_uncs
        self.uncs = uncertainties
    
    @abc.abstractmethod 
    def __iter__(self):
        '''should return iterator'''


class DefaultDesigns(AbstractDesignsIterable):
    def __iter__(self):
        designs = zip(*[self.sampled_uncs[u] for u in self.uncs]) 
        return design_generator(designs, self.uncs)


class FullFactorialDesigns(AbstractDesignsIterable):
    def __iter__(self):
        designs = itertools.product(*[self.sampled_uncs[u] for u in self.uncs])
        return design_generator(designs, self.uncs)


class PartialFactorialDesigns(object):
    def __init__(self, ff_designs, other_designs):
        self.ff_designs = ff_designs
        self.other_designs = other_designs
    
    def __iter__(self):
        designs =  itertools.product(self.ff_designs, self.other_designs)
        return partial_designs_generator(designs)

def partial_designs_generator(designs):
    '''generator which combines the full factorial part of the desing with
    the non full factorial part into a single dict
    
    Parameters
    ----------
    designs: iterable of tuples
    
    Yields
    ------
    dict
        experimental design dict
    
    
    '''

    for design in designs:
        ff_part, other_part = design
        
        design = ff_part.copy()
        design.update(other_part)
        
        yield design

def design_generator(designs, uncs):
    '''generator that combines the sampled uncertainties with their correct 
    name in order to return dicts.
    
    Parameters
    ----------
    designs : iterable of tuples
    uncs : iterable of str
    
    Yields
    ------
    dict
        experimental design dictionary
    
    '''
    
    for design in designs:
        design = {unc:design[i] for i, unc in enumerate(uncs)}
        yield design
        