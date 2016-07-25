'''


'''
from __future__ import (unicode_literals, print_function, absolute_import,
                        division)

import abc
import numbers
import warnings

from .util import NamedObject
from ema_workbench.em_framework.util import NamedDict
# from ema_workbench.em_framework.model import AbstractModel

# Created on Jul 14, 2016
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['Parameter','RealParameter', 'IntegerParameter',
           'CategoricalParameter']


class Constant(NamedObject):
    '''Constant class, 
    
    can be used for any parameters that have to be set to a fixed value
    
    '''
    
    def __init__(self, name, value):
        super(Constant, self).__init__(name)
        self.value = value


class Parameter(NamedObject):
    ''' Base class for any model input parameter
    
    Parameters
    ----------
    name : str
    lower_bound : int or float
    upper_bound : int or float
    resolution : collection
    
    Raises
    ------
    ValueError 
        if lower bound is larger than upper bound
    ValueError 
        if entries in resolution are outside range of lower_bound and
        upper_bound
    
    '''
    
    __metaclass__ = abc.ABCMeta
        
    INTEGER = 'integer'
    UNIFORM = 'uniform'
    
    def __init__(self, name, lower_bound, upper_bound, resolution=None,
                 default=None):
        super(Parameter, self).__init__(name)
        
        if resolution is None:
            resolution = []
        
        for entry in resolution:
            if not ((entry >= lower_bound) and (entry <= upper_bound)):
                raise ValueError(('resolution not consistent with lower and ' 
                                  'upper bound'))
        
        if lower_bound >= upper_bound:
            raise ValueError('upper bound should be larger than lower bound')
        
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.resolution = resolution
        self.default = default
        
    def __eq__ (self, other):
        comparison = [all(hasattr(self, key) == hasattr(other, key) and
                          getattr(self, key) == getattr(other, key) for key 
                          in self.__dict__.keys())]
        comparison.append(self.__class__ == other.__class__)
        return all(comparison)

    def __str__(self):
        return self.name
            
            
class RealParameter(Parameter):
    ''' real valued model input parameter
    
    Parameters
    ----------
    name : str
    lower_bound : int or float
    upper_bound : int or float
    resolution : iterable 
    
    Raises
    ------
    ValueError 
        if lower bound is larger than upper bound
    ValueError 
        if entries in resolution are outside range of lower_bound and
        upper_bound
    
    '''
    
    
    def __init__(self, name, lower_bound, upper_bound, resolution=None, 
                 default=None):
        super(RealParameter, self).__init__(name, lower_bound, upper_bound,
                                        resolution=resolution, default=default)
        
        self.dist = Parameter.UNIFORM
        
    @property
    def params(self):
        return (self.lower_bound, self.upper_bound-self.lower_bound)
        
        
class IntegerParameter(Parameter):
    ''' integer valued model input parameter
    
    Parameters
    ----------
    name : str
    lower_bound : int or float
    upper_bound : int or float
    resolution : iterable
    
    Raises
    ------
    ValueError 
        if lower bound is larger than upper bound
    ValueError 
        if entries in resolution are outside range of lower_bound and
        upper_bound, or not an numbers.Integral instance
    ValueError 
        if lower_bound or upper_bound is not an numbers.Integral instance
    
    '''
    
    def __init__(self, name, lower_bound, upper_bound, resolution=None, 
                 default=None):
        super(IntegerParameter, self).__init__(name, lower_bound, upper_bound, 
                                        resolution=resolution, default=default)
        
        lb_int = isinstance(lower_bound, numbers.Integral) 
        up_int = isinstance(upper_bound, numbers.Integral)
        
        if not (lb_int or up_int):
            raise ValueError('lower bound and upper bound must be integers')
        
        for entry in self.resolution:
            if not isinstance(entry, numbers.Integral):
                raise ValueError(('all entries in resolution should be ' 
                                  'integers'))
        
        self.dist = Parameter.INTEGER
        
    @property
    def params(self):
        return (self.lower_bound, self.upper_bound)


class CategoricalParameter(IntegerParameter):
    ''' categorical model input parameter
    
    Parameters
    ----------
    name : str
    categories : collection of obj
    
    '''
    
    def __init__(self, name, categories, default=None):
        lower_bound = 0
        upper_bound = len(categories)

        super(CategoricalParameter, self).__init__(name, lower_bound, 
                                upper_bound, resolution=None, default=default)
        self.resolution = list(categories)
        
    def index_for_cat(self, category):
        '''return index of category
        
        Parameters
        ----------
        category : object
        
        Returns
        -------
        int
        
        
        '''
        
        return self.resolution.index(category)
    
    def cat_for_index(self, index):
        '''return category associated with index
        
        Parameters
        ----------
        index  : int
        
        Returns
        -------
        object
        
        '''
        
        return self.resolution[index]
        
        
    def transform(self, value):
        '''transform an integer to a category 
        
        Parameters
        ----------
        name : int
               value for which you want the category
               
        Raises
        ------
        IndexError
            if value is out of bounds
        '''
        warnings.warn('deprecated, use cat_for_index instead')

        return self.cat_for_index(value)
    
    def invert(self, name):
        ''' invert a category to an integer
        
        Parameters
        ----------
        name : obj
               category
               
        Raises
        ------
        ValueError
            if category is not found
        
        '''
        warnings.warn('deprecated, use index_for_cat instead')
        
        return self.index_for_cat(name)

class Policy(NamedDict):
    pass

class Scenario(NamedDict):
    pass

class Experiment(NamedObject):

    def __init__(self, name, model, policy, experiment_id, **kwargs):
        super(Experiment, self).__init__(name)
        
#         if not isinstance(model, AbstractModel):
#             raise ValueError('pass the actual model instance')
         
        self.policy = policy
        self.model = model
        self.experiment_id = experiment_id
        self.scenario = Scenario(name, **kwargs)
