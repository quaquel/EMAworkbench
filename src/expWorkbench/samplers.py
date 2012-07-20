'''

Created on 16 aug. 2011

This module contains various classes that can be used for specifying different
types of samplers. These different samplers implement basic sampling 
techniques including Full Factorial sampling, Latin Hypercube sampling, and
Monte Carlo sampling.

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

'''

import itertools
import scipy.stats as stats
import numpy as np 

from EMAlogging import info, warning
from uncertainties import CategoricalUncertainty

__all__ = ['LHSSampler',
           'MonteCarloSampler',
           'FullFactorialSampler',
           'Sampler']

#==============================================================================
# sampler classes
#==============================================================================
   
class Sampler(object):
    '''
    base class from which different samplers can be derived
    '''
    
    #: types of distributions known by the sampler.
    #: by default it knows the `uniform continuous <http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html>`_
    #: distribution for sampling floats, and the `uniform discrete <http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.randint.html#scipy.stats.randint>`_
    #: distribution for sampling integers. 
    distributions = {"uniform" : stats.uniform,
                     "integer" : stats.randint
                              }
    
    def __init__(self):
        super(Sampler, self).__init__()

    def sample(self, distribution, params, size):
        '''
        method for sampling a number of samples from a particular distribution.
        The various samplers differ with respect to their implementation
        of this method. 
        
        :param distribution: the distribution to sample from
        :param params: the parameters specifying the distribution
        :param size: the number of samples to generate
        
        '''
        raise NotImplementedError
    
    def generate_design(self, uncertainties, size):
        '''
        The main method of :class: `~sampler.Sampler` and its 
        children. This will call the sample method for each of the 
        uncertainties and return the resulting designs. 
        
        :param uncertainties: a collection of 
                               :class:`~uncertainties.ParameterUncertainty` 
                               and :class:`~uncertainties.CategoricalUncertainty`
                               instances.
        :param size: the number of samples to generate.
        :rtype: tuple with the designs as the first entry and the names
                of the uncertainties as the second entry.
        '''
        
        designs = []

        for uncertainty in uncertainties:
            #range in uncertainty gives lower and upper bound
            cases = self.sample(uncertainty.dist, uncertainty.params, size) 
            
            if type(uncertainty) == CategoricalUncertainty:
                cases = [uncertainty.transform(int(case)) for case in cases]
                cases = np.asarray(cases)
            
            designs.append(cases)
        
        designs = zip(*designs)
        return (designs, [uncertainty.name for uncertainty in uncertainties])

class LHSSampler(Sampler):
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
        smp = [stats.uniform(i,1./n).rvs() for i in perc]
        v = dist(*parms).ppf(smp)
        
        if isinstance(siz,(tuple,list)):
            v.shape = siz
        return v

    
        
class MonteCarloSampler(Sampler):
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

class FullFactorialSampler(Sampler):     
    '''
    generates a full factorial sample.
    If the uncertainty is non categorical, resolution is used to set the 
    samples. If the uncertainty is an integer, their wont be duplicates in 
    the sample. So, samples is equal to or smaller then the specified 
    resolution
    
    '''
    
    #: max number of designs that is allowed (Default 50.000).
    max_designs = 50000
    
    def __init__(self):
        super(FullFactorialSampler, self).__init__()
       
    def generate_design(self,  uncertainties, size):
        '''
        This method provides an alternative implementation to the default 
        implementation provided by :class:`~sampler.Sampler`. This
        version returns a full factorial design across the uncertainties. 
        
        :param uncertainties: a collection of 
                               :class:`~uncertainties.ParameterUncertainty` 
                               and :class:`~uncertainties.CategoricalUncertainty`
                               instances.
        :param size: the resolution to use for :class:`~uncertainties.ParameterUncertainty`
                     instances. For instances of :class:`~uncertainties.CategoricalUncertainty`,
                     the categories are used.
        :rtype: tuple with the designs as the first entry and the names
                of the uncertainties as the second entry.
        
        .. note:: The current implementation has a hard coded limit to the 
                  number of designs possible. This is set to 50.000 designs. 
                  If one want to go beyond this, set `self.max_designs` to
                  a higher value.
        
        '''
        
        def get_combos(branches):
            return itertools.product(*branches)
        
        categories = []
        totalDesigns = 1
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
            totalDesigns *= len(category)
            categories.append(category)
        
        if totalDesigns > self.max_designs:
            warning("full factorial design results in %s designs"\
                    % totalDesigns)
            raise Exception('too many designs')
        else:
            info("full factorial design results in %s designs"\
                 % totalDesigns)    
        
        designs = itertools.product(*categories)
        return (designs, [uncertainty.name for uncertainty in uncertainties])
  
