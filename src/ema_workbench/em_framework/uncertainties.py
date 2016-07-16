'''

This module contains various classes that can be used for specifying different
types of uncertainties.


'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
import abc
from .parameters import RealParameter, CategoricalParameter, IntegerParameter

# Created on 16 aug. 2011
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


__all__ = ['AbstractUncertainty',
           'RealUncertainty',
           'IntegerUncertainty',
           'CategoricalUncertainty']



class AbstractUncertainty(object):
    '''
    :class:`AbstractUncertainty` provides a template for specifying different
    types of uncertainties. 
    
    Parameters
    ----------
    values : tuple
    name : str
    
    Attributes
    ----------
    values : tuple
    name : str 
    dist : {INTEGER, UNIFORM}
    
    '''
    __metaclass__ = abc.ABCMeta
    
    values = None
    name = None
    dist = None
    
    def __init__(self, values, name):
        '''
        
        Parameters
        ----------
        values: tuple
                the values for specifying the uncertainty from which to 
                sample
        name: str
              name of the uncertainty
        
        '''
        
        super(AbstractUncertainty, self).__init__()
        self.values = values
        self.name = name

    
    def __eq__ (self, other):
        comparison = [all(hasattr(self, key) == hasattr(other, key) and
                          getattr(self, key) == getattr(other, key) for key 
                          in self.__dict__.keys())]
        comparison.append(self.__class__ == other.__class__)
        return all(comparison)

    def __str__(self):
        return self.name

class RealUncertainty(RealParameter):
    """
    :class:`RealUncertainty` is used for specifying real valued parametric 
    uncertainties. An uncertainty is real if the range is continuous from
    the lower bound to the upper bound.

    Parameters
    ----------
    name : str
    lower_bound : int or float
    upper_bound : int or float
    resolution : iterable
    factorial : bool
             
    Raises
    ------
    ValueError 
        if lower bound is larger than upper bound
    ValueError 
        if entries in resolution are outside range of lower_bound and
        upper_bound
    
    """
    
    
    def __init__(self, name, lower_bound, upper_bound, resolution=None, 
                 factorial=False):
        # TODO:: autogenerate resolution if resolution is int
        self.factorial=factorial
        
        super(RealUncertainty, self).__init__(name, lower_bound, upper_bound,
                                              resolution)


class IntegerUncertainty(IntegerParameter):
    """
    :class:`IntegerUncertainty` is used for specifying continues integer 
    valued parametric  uncertainties.

    Parameters
    ----------
    name : str
    lower_bound : int 
    upper_bound : int 
    resolution : iterable
    factorial :  bool
             
    Raises
    ------
    ValueError 
        if lower bound is larger than upper bound
    ValueError 
        if entries in resolution are outside range of lower_bound and
        upper_bound, or not an numbers.Integral instance
    ValueError 
        if lower_bound or upper_bound is not an numbers.Integral instance
        
    
    """
    
    def __init__(self, name, lower_bound, upper_bound, resolution=None,
                 factorial=False):
        # TODO:: autogenerate resolution if resolution is int
        self.factorial=factorial
        
        super(IntegerUncertainty, self).__init__(name, lower_bound, upper_bound,
                                              resolution)

class CategoricalUncertainty(CategoricalParameter):
    """
    :class:`CategoricalUncertainty` can can be used for sampling over 
    categorical variables. The categories can be of any type, including 
    Strings, Integers, Floats, Tuples, or any Object. 

    Underneath, this is treated as a IntegerUncertainty. That is,
    an integer parametric uncertainty is used with each integer corresponding
    to a particular category.  This class  called by the sampler to transform 
    the integer back to the appropriate category.
    
    Parameters
    ----------
    name : str
    categories : collection of obj
    factorial : bool
    
    """
    
    def __init__(self, name, categories, factorial=False):
        super(CategoricalUncertainty, self).__init__(name, categories)
        self.factorial=False