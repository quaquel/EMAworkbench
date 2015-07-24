'''

Created on 16 aug. 2011

This module contains various classes that can be used for specifying different
types of samplers. These different samplers implement basic sampling 
techniques including Full Factorial sampling, Latin Hypercube sampling, and
Monte Carlo sampling.

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

'''
import abc
import itertools
import numpy as np 
import scipy.stats as stats

from uncertainties import CategoricalUncertainty
from abc import abstractmethod

__all__ = ['LHSSampler',
           'MonteCarloSampler',
           'FullFactorialSampler',
           'AbstractSampler']

class AbstractSampler(object):
    '''
    base class from which different samplers can be derived
    '''
    __metaaclass__ = abc.ABCMeta
    
    
    #: types of distributions known by the sampler.
    #: by default it knows the `uniform continuous <http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html>`_
    #: distribution for sampling floats, and the `uniform discrete <http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.randint.html#scipy.stats.randint>`_
    #: distribution for sampling integers. 
    distributions = {"uniform" : stats.uniform,
                     "integer" : stats.randint
                              }
    
    def __init__(self):
        super(AbstractSampler, self).__init__()

    @abstractmethod
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
            
            if type(uncertainty) == CategoricalUncertainty:
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
        a generator object that yields the designs resulting from
                combining the uncertainties
        int
            the number of experimental designs
        
        '''
        
        sampled_uncertainties = self.generate_samples(uncertainties, nr_samples)
        a = zip(*sampled_uncertainties.values())
        designs = itertools.izip(*sampled_uncertainties) 
        return designs, self.deterimine_nr_of_designs(sampled_uncertainties)

    def deterimine_nr_of_designs(self, sampled_uncertainties):
        '''
        Helper function for determining the number of experiments that will
        be generated given the sampled uncertainties.
        
        Parameter
        ---------
        sampled_uncertainties : list 
                        a list of sampled uncertainties, as 
                        the values return by generate_samples
        
        Returns
        -------
        int
            the total number of experimental design
        
        '''
        
        return len(sampled_uncertainties.values()[0])

class LHSSampler(AbstractSampler):
    """
    generates a Latin Hypercube sample for each of the uncertainties
    in case of categorical uncertainties, it handles the transform as well
    """
    
    def __init__(self):
        super(LHSSampler, self).__init__()
    
    def sample(self, distribution, params, size):
        '''
        generate a Latin Hypercupe Sample.
        
        :param distribution: the distribution to sample from
        :param params: the parameters specifying the distribution
        :param size: the number of samples to generate
    
        '''
        
        return self._lhs(self.distributions[distribution], params, size)

    def _lhs(self, dist, parms, siz=100):
        '''
        Latin Hypercube sampling of any distribution.

        modified from code found `online <http://code.google.com/p/bayesian-inference/source/browse/trunk/BIP/Bayes/lhs.py?r=3cfbbaa5806f2b8cc9e2457d967b0a58a3ce459c>`_.
    
        :param dist: random number generator from `scipy.stats <http://docs.scipy.org/doc/scipy/reference/stats.html>`_
        :param parms: tuple of parameters as required for dist.
        :param siz: number or shape tuple for the output sample
    
        '''
        if not isinstance(dist, (stats.rv_discrete,stats.rv_continuous)):
            raise TypeError('dist is not a scipy.stats distribution object')
        #number of samples
        n=siz
        if isinstance(siz,(tuple,list)):
            n= np.product(siz)
        
        perc = np.arange(0,1.,1./n)
        np.random.shuffle(perc)
        smp = stats.uniform(perc,1./n).rvs() 
        v = dist(*parms).ppf(smp)
        
        if isinstance(siz,(tuple,list)):
            v.shape = siz
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
        
        :param distribution: the distribution to sample from
        :param params: the parameters specifying the distribution
        :param size: the number of samples to generate
        '''
        
        return self.distributions[distribution](*params).rvs(size)

class FullFactorialSampler(AbstractSampler):     
    '''
    generates a full factorial sample.
    If the uncertainty is non categorical, resolution is used to set the 
    samples. If the uncertainty is an integer, their wont be duplicates in 
    the sample. So, samples is equal to or smaller then the specified 
    resolution
    
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
        :param uncertainties: a collection of 
                               :class:`~uncertainties.ParameterUncertainty` 
                               and :class:`~uncertainties.CategoricalUncertainty`
                               instances.
        :param size: the number of samples to generate.
        :rtype: dict with the uncertainty.name as key, and the sample as value
        
        Returns
        -------
        
        '''
                
        
        samples = {}
        for uncertainty in uncertainties:
            if type(uncertainty) == CategoricalUncertainty:
                category = uncertainty.categories
            else:
                category = np.linspace(uncertainty.values[0], 
                                       uncertainty.values[1], 
                                       size)
                if uncertainty.dist == 'integer':
                    category = np.round(category, 0)
                    category = set(category)
                    category = [int(entry) for entry in category]
                    category = sorted(category)
            samples[uncertainty.name] = category
        
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
        a generator object that yields the designs resulting from
                combining the uncertainties
        int
            the number of experimental designs
        
        '''
        sampled_uncertainties = self.generate_samples(uncertainties, nr_samples)
        designs = itertools.product(*sampled_uncertainties)
        return designs, self.deterimine_nr_of_designs(sampled_uncertainties)

    def deterimine_nr_of_designs(self, sampled_uncertainties):
        '''
        Helper function for determining the number of experiments that will
        be generated given the sampled uncertainties.
        
        Parameter
        ---------
        sampled_uncertainties : list 
                        a list of sampled uncertainties, as 
                        the values return by generate_samples
        
        Returns
        -------
        int
            the total number of experimental design
        '''
        nr_designs = 1
        for value in sampled_uncertainties.itervalues():
            nr_designs *= len(value)
        return nr_designs
        
          
