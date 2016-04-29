'''

This module contains various classes that can be used for specifying different
types of uncertainties.


'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
import abc
import numpy as np

# Created on 16 aug. 2011
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


__all__ = ['AbstractUncertainty',
           'ParameterUncertainty',
           'CategoricalUncertainty']

INTEGER = 'integer'
UNIFORM = 'uniform'

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


class ParameterUncertainty(AbstractUncertainty ):
    """
    :class:`ParameterUncertainty` is used for specifying parametric 
    uncertainties. An uncertainty is parametric if the range is continuous from
    the lower bound to the upper bound.

    Parameters
    ----------
    values : tuple
             the values for specifying the uncertainty from which to 
             sample. Values should be a tuple with the lower and
             upper bound for the uncertainty. These bounds are
             inclusive. 
    name: str
          name of the uncertainty
    integer: bool, optional
             if True, the parametric uncertainty is an integer
    factorial: bool, optional
               if true, and sampler is partial factorial sampler, include
               this uncertainty as part of the factorial design
    resolution: int, or iterable, optional
                the resolution of specific values to use in case of inclusion
                in a factorial design. 
             
    Raises
    ------
    ValueError
        if the length of values is not equal to 2, or when the first
        element in values is larger than the second element in values
        
    TODO:: Note that in case that integer is true, resolution will be cast
    to integer.
    
    """
    
    
    @property
    def resolution(self):
        return self._resolution
    
    @resolution.setter
    def resolution(self, value):
        # TODO:: what to do with resolution of 1
        # the addition of resolution to parameter uncertainty makes the
        # discretization in full factorial sampler for parameter uncertainties
        # redundant
        
        try:
            value = np.linspace(self.values[0], self.values[1], value)
        except TypeError:
            list(value).sort()
            if value[0]<self.values[0] or value[-1]>self.values[1]:
                raise ValueError(('resolution larger than range specified by' 
                                  'values'))
            
        if self.dist==INTEGER:
            value = [int(entry) for entry in value]
            
        self._resolution = tuple(value)
    
    def __init__(self, values, name, integer=False, factorial=False,
                 resolution=3):
        if len(values)!=2:
            raise ValueError("length of values for %s incorrect " % name)
        if (values[0] >= values[1]):
            raise ValueError("upper limit is not larger than lower limit")
       
        super(ParameterUncertainty, self).__init__(values, name)
        
        # self.dist should be a string. This string should be a key     
        # in the distributions dictionary 
        if integer:
            self.dist = INTEGER
            self.params = (values[0], values[1]+1)
        else:
            self.dist = UNIFORM
            #params for initializing self.dist
            self.params = (self.values[0], 
                           self.values[1]-self.values[0])

        self.factorial = factorial
        self.resolution = resolution


class CategoricalUncertainty(ParameterUncertainty):
    """
    :class:`CategoricalUncertainty` can can be used for sampling over 
    categorical variables. The categories can be of any type, including 
    Strings, Integers, Floats, Tuples, or any Object. As values the categories 
    are specified in a collection. 

    Underneath, this is treated as a integer parametric uncertainty. That is,
    an integer parametric uncertainty is used with each integer corresponding
    to a particular category.  This class  called by the sampler to transform 
    the integer back to the appropriate category.
    
    Parameters
    ----------
    categories: collection
            the values for specifying the uncertainty from which to 
            sample. Values should be a collection.
    name:str 
         name of the uncertainty
    
    Attributes
    ----------
    categories  : list or tuple
                  list or tuple with the categories
    
    """

    categories = None
    
    def __init__(self, categories, name, factorial=False):
        self.categories = categories
        values = (0, len(categories)-1)
        resolution = np.arange(values[0], values[1]+1)

        super(CategoricalUncertainty, self).__init__(values, 
                                                     name, 
                                                     integer=True,
                                                     factorial=factorial,
                                                     resolution=resolution)
        self.integer = True
                
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

        return self.categories[value]
    
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
        return self.categories.index(name)

