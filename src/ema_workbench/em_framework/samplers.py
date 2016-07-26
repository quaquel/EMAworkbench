'''

This module contains various classes that can be used for specifying different
types of samplers. These different samplers implement basic sampling 
techniques including Full Factorial sampling, Latin Hypercube sampling, and
Monte Carlo sampling.

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
from .parameters import (CategoricalParameter, IntegerParameter)

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
        The various samplers differ with respect to their implementation of 
        this method. 
        
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
    
    def generate_samples(self, parameters, size):
        '''
        The main method of :class: `~sampler.Sampler` and its 
        children. This will call the sample method for each of the 
        parameters and return the resulting designs. 
        
        Parameters
        ----------
        parameters : collection
                     a collection of :class:`~parameters.Parameterparamertainty` 
                     and :class:`~parameters.Categoricalparamertainty`
                     instances.
        size : int
               the number of samples to generate.
        
        
        Returns
        -------
        dict
            dict with the paramertainty.name as key, and the sample as value
        
        '''
        
        samples = {}

        for param in parameters:
            #range in parameter gives lower and upper bound
            sample = self.sample(param.dist, param.params, size) 
            
            if isinstance(param, CategoricalParameter):
                # TODO look into numpy ufunc
                sample = [param.cat_for_index(int(entry)) for entry in sample]
            elif isinstance(param, IntegerParameter):
                sample = (int(entry) for entry in sample)
            
            samples[param.name] = sample
        
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
        
        sampled_parameters = self.generate_samples(parameters, nr_samples)
        params = sorted(sampled_parameters.keys())
        designs = DefaultDesigns(sampled_parameters, params)
        
        return designs, nr_samples

#     def determine_nr_of_designs(self, sampled_parameters):
#         '''
#         Helper function for determining the number of experiments that will
#         be generated given the sampled parameters.
#         
#         Parameters
#         ----------
#         sampled_parameters : list 
#                         a list of sampled parameters, Typically,
#                         this will be the values of the dict returned by
#                         :meth:`generate_samples`. 
#         
#         Returns
#         -------
#         int
#             the total number of experimental design
#         
#         '''
#         
#         return len(next(iter(sampled_parameters.values())))


class LHSSampler(AbstractSampler):
    """
    generates a Latin Hypercube sample for each of the parameters
    in case of categorical parameters, it handles the transform as well
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
            with the paramertainty.name as key, and the sample as value
    
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
    generates a Monte Carlo sample for each of the parameters. 
    
    
    In case of a CategoricalParameter it also handles the transformation from 
    integers back to categories
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
        params : 2-tuple of floats
                 the parameters specifying the distribution
        size : int
               the number of samples to generate
        
        Returns
        -------
        dict 
            with the paramertainty.name as key, and the sample as value
               
        '''
        
        return self.distributions[distribution](*params).rvs(size)


class FullFactorialSampler(AbstractSampler):     
    '''
    generates a full factorial sample.
    
    If the parameter is non categorical, the resolution is set the 
    number of samples. If the parameter is categorical, the specified value 
    for samples will be ignored and each category will be used instead. 
    
    '''
    
    def __init__(self):
        super(FullFactorialSampler, self).__init__()
       
    def generate_samples(self,  parameters, size):
        '''
        The main method of :class: `~sampler.Sampler` and its 
        children. This will call the sample method for each of the 
        parameters and return the resulting samples 
        
        Parameters
        ----------
        parameters : collection
                        a collection of :class:`~parameters.Parameter`
                        instances
        size : int
                the number of samples to generate.
        
        Returns
        -------
        dict 
            with the paramertainty.name as key, and the sample as value
        '''
        samples = {}
        for param in parameters:
            cats = param.resolution
            if not cats:
                cats = np.linspace(param.lower_bound, 
                                   param.upper_bound, 
                                   size)
                if isinstance(param, IntegerParameter):
                    cats = np.round(cats, 0)
                    cats = set(cats)
                    cats = (int(entry) for entry in cats)
                    cats = sorted(cats)
            samples[param.name] = cats
        
        return samples
       
    def generate_designs(self,  parameters, nr_samples):
        '''
        This method provides an alternative implementation to the default 
        implementation provided by :class:`~sampler.Sampler`. This
        version returns a full factorial design across the parameters. 
        
        Parameters
        ----------
        parameters : list 
                        a list of parameters for which to generate the
                        experimental designs
        nr_samples : int
                     the number of intervals to use on each
                     Parameter. Categorical parameters always
                     return all their categories
        
        Returns
        -------
        generator
            a generator object that yields the designs resulting from
            combining the parameters
        int
            the number of experimental designs
        
        '''
        sampled_parameters = self.generate_samples(parameters, nr_samples)
        params = sorted(sampled_parameters.keys())
        designs = FullFactorialDesigns(sampled_parameters, params)
        return designs, self.determine_nr_of_designs(sampled_parameters)

    def determine_nr_of_designs(self, sampled_parameters):
        '''
        Helper function for determining the number of experiments that will
        be generated given the sampled parameters.
        
        Parameters
        ----------
        sampled_parameters : list 
                        a list of sampled parameters, as 
                        the values return by generate_samples
        
        Returns
        -------
        int
            the total number of experimental design
        '''
        nr_designs = 1
        for value in sampled_parameters.values():
            nr_designs *= len(value)
        return nr_designs


class PartialFactorialSampler(AbstractSampler):
    """
    generates a partial factorial design over the parameters. Any parameter 
    where factorial is true will be included in a factorial design, while the 
    remainder will be sampled using LHS or MC sampling.
    
    Parameters
    ----------
    sampling: {PartialFactorialSampler.LHS, PartialFactorialSampler.MC}, optional
              the desired sampling for the non factorial parameters.
    
    Raises
    ------
    ValueError
        if sampling is not either LHS or MC
    
    """
    
    LHS = 'LHS'
    MC = 'MC'
    
    def __init__(self, sampling='LHS'):
        super(PartialFactorialSampler, self).__init__()
        
        if sampling==PartialFactorialSampler.LHS:
            self.sampler = LHSSampler()
        elif sampling==PartialFactorialSampler.MC:
            self.sampler = MonteCarloSampler()
        else:
            raise ValueError(('invalid value for sampling type, should be LHS ' 
                              'or MC'))
        self.ff = FullFactorialSampler()
    
    def _sort_parameters(self, parameters, ff_params):
        '''sort parameters into full factorial and other
        
        Parameters
        ----------
        parameters : list of parameters
        ff_params : list of str
                    names of the parameters over which the full factorial 
                    design should be generated
        
        '''
        ff_paramnames = set(ff_params)
        ff_params = []
        other_params = []
        for param in parameters:
            if param.name in ff_paramnames:
                ff_params.append(param)
            else:
                other_params.append(param)
                
        return ff_params, other_params
        
    
    def generate_designs(self, parameters, nr_samples, ff_params=[]):
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
        ff_params : list of str
                    names of the parameters over which the full factorial 
                    design should be generated
        
        Returns
        -------
        generator
            a generator object that yields the designs resulting from
            combining the parameters
        int
            the number of experimental designs
        
        '''
        
        ff_params, other_params = self._sort_parameters(parameters, ff_params)
        
        # generate a design over the factorials
        # TODO update ff to use resolution if present
        ff_designs, n_ff = self.ff.generate_designs(ff_params, nr_samples)
        
        # generate a design over the remainder
        # for each factorial, run the MC design
        other_designs, n_other = self.sampler.generate_designs(other_params, 
                                                              nr_samples)
        
        nr_designs = n_other * n_ff
        
        designs = PartialFactorialDesigns(ff_designs, other_designs)
        
        return designs, nr_designs

    
def determine_parameters(models, attribute, union=True):
    '''determine the parameters over which to sample
    
    Parameters
    ----------
    models : a collection of AbstractModel instances
    attribute : {'uncertainties', 'levers'}
    union : bool, optional
            in case of multiple models, sample over the union of
            levers, or over the intersection of the levers
    sampler : Sampler instance, optional
    
    Returns
    -------
    collection of Parameter instances
    
    '''
    parameters = getattr(models[0], attribute).copy()
    intersection = set(parameters.keys())
    
    # gather parameters across all models
    # TODO:: need to make slice work on NamedObjectMap
    for model in models[1::]:
        model_params = getattr(model, attribute)
        
        # relies on name based identity, do we want that?
        parameters.extend(model_params)

        intersection = intersection.intersection(model_params.keys())
    
    # in case not union, remove all parameters not in intersection
    if not union:
        params_to_remove = set(parameters.keys()) - intersection
        for key in params_to_remove:
            del parameters[key]
    return parameters
            
def sample_levers(models, n_samples, union=True, sampler=LHSSampler):
    '''generate policies by sampling over the levers
    
    Parameters
    ----------
    models : a collection of AbstractModel instances
    n_samples : int
    union : bool, optional
            in case of multiple models, sample over the union of
            levers, or over the intersection of the levers
    sampler : Sampler instance, optional
    
    Returns
    -------
    generator yielding Policy instances
    
    '''
    levers = determine_parameters(models, 'levers', union=union)
    samples, n = sampler.generate_designs(levers, n_samples)
    
    # wrap samples in Policy
    raise Exception()
    

def sample_uncertainties(models, n_samples, union=True, sampler=LHSSampler):
    '''generate scenarios by sampling over the uncertainties
    
    Parameters
    ----------
    models : a collection of AbstractModel instances
    n_samples : int
    union : bool, optional
            in case of multiple models, sample over the union of
            uncertainties, or over the intersection of the uncertianties
    sampler : Sampler instance, optional
    
    Returns
    -------
    generator 
        yielding Scenario instances
    collection
        the collection of parameters over which to sample
    n_samples
        the number of scenarios (!= n_samples in case off FF sampling)
    
    
    '''
    uncertainties = determine_parameters(models, 'uncertainties', union=union)
    samples, n = sampler.generate_designs(uncertainties, n_samples)
    
    return samples, uncertainties, n
    # wrap samples in Scenario





class AbstractDesignsIterable(object):
    '''iterable for the experimental designs'''
    
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, sampled_params, parameters):
        self.sampled_params = sampled_params
        self.params = parameters
    
    @abc.abstractmethod 
    def __iter__(self):
        '''should return iterator'''


class DefaultDesigns(AbstractDesignsIterable):
    def __iter__(self):
        designs = zip(*[self.sampled_params[u] for u in self.params]) 
        return design_generator(designs, self.params)


class FullFactorialDesigns(AbstractDesignsIterable):
    def __iter__(self):
        designs = itertools.product(*[self.sampled_params[u] for u in self.params])
        return design_generator(designs, self.params)


class PartialFactorialDesigns(object):
    def __init__(self, ff_designs, other_designs):
        self.ff_designs = ff_designs
        self.other_designs = other_designs
    
    def __iter__(self):
        designs =  itertools.product(self.ff_designs, self.other_designs)
        return partial_designs_generator(designs)

def partial_designs_generator(designs):
    '''generator which combines the full factorial part of the design with
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

def design_generator(designs, params):
    '''generator that combines the sampled parameters with their correct 
    name in order to return dicts.
    
    Parameters
    ----------
    designs : iterable of tuples
    params : iterable of str
    
    Yields
    ------
    dict
        experimental design dictionary
    
    '''
    
    for design in designs:
        design = {param:design[i] for i, param in enumerate(params)}
        yield design
        