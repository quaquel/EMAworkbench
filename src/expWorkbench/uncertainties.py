'''

Created on 16 aug. 2011

This module contains various classes that can be used for specifying different
types of uncertainties.

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

'''
import abc
from expWorkbench.ema_exceptions import EMAError

__all__ = ['AbstractUncertainty',
           'ParameterUncertainty',
           'CategoricalUncertainty'
           ]

INTEGER = 'integer'
UNIFORM = 'uniform'

class AbstractUncertainty(object):
    '''
    :class:`AbstractUncertainty` provides a template for specifying different
    types of uncertainties.
    '''
    __metaclass__ = abc.ABCMeta
    
    
    #: the values that specify the uncertainty
    values = None
   
    #: the name of the uncertainty
    name = None
    
    #: a string denoting the type of distribution to be used in sampling
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
    
    def get_values(self):
        ''' get values'''
        return self.values

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
    
    Parametric uncertainties are either floats or integers. 
        
    """
    
    def __init__(self, values, name, integer=False):
        '''
        
        Parameters
        ----------
        values : tuple
                 the values for specifying the uncertainty from which to 
                 sample. Values should be a tuple with the lower and
                 upper bound for the uncertainty. These bounds are
                 inclusive. 
        name: str
              name of the uncertainty
        integer: bool 
                 if True, the parametric uncertainty is an integer
                 
        Raises
        ------
        ValueError
            if the lenght of values is not equal to 2, or when the first
            element in values is larger than the second element in values
                 
        
        '''
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
            self.params = (self.get_values()[0], 
                           self.get_values()[1]-self.get_values()[0])


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
    """
    
    #: the categories of the uncertainty
    categories = None
    
    def __init__(self, values, name):
        '''
        
        Parameters
        ----------
        values: collection
                the values for specifying the uncertainty from which to 
                sample. Values should be a collection.
        name:str 
             name of the uncertainty
        
        '''
        self.categories = values
        values = (0, len(values)-1)

        super(CategoricalUncertainty, self).__init__(values, 
                                                     name, 
                                                     integer=True)
        self.integer = True
                
    def transform(self, value):
        '''transform an integer to a category 
        
        Parameter
        ---------
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
        
        Parameter
        ---------
        name : obj
               category
               
        Raises
        ------
        ValueError
            if category is not found
        
        '''
        return self.categories.index(name)

