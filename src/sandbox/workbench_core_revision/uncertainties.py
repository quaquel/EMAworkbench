'''

Created on 16 aug. 2011

This module contains various classes that can be used for specifying different
types of uncertainties.

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

'''
import numpy as np

from sets import ImmutableSet
from expWorkbench.ema_exceptions import EMAError

SVN_ID = '$Id: uncertainties.py 1029 2012-11-21 20:12:09Z jhkwakkel $'
__all__ = ['AbstractUncertainty',
           'ParameterUncertainty',
           'CategoricalUncertainty'
           ]

INTEGER = 'integer'
UNIFORM = 'uniform'

#==============================================================================
# uncertainty classes
#==============================================================================
class AbstractUncertainty(object):
    '''
    :class:`AbstractUncertainty` provides a template for specifying different
    types of uncertainties.
    '''
    
    #: the values that specify the uncertainty
    values = None
    
    #: the type of integer
    type = None
    
    #: the name of the uncertainty
    name = None
    
#    #: the datatype of the uncertainty
#    dtype = None    
    
    #: a string denoting the type of distribution to be used in sampling
    dist = None
    
    def __init__(self, values, name):
        '''
        
        :param values: the values for specifying the uncertainty from which to 
                       sample
        :param name: name of the uncertainty
        
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
    
    #: optional attribute for specifying default value for uncertainty
    default = None
    
    def __init__(self, values, name, integer=False, default = None):
        '''
        
        :param values: the values for specifying the uncertainty from which to 
                       sample. Values should be a tuple with the lower and
                       upper bound for the uncertainty. These bounds are
                       inclusive. 
        :param name: name of the uncertainty
        :param integer: boolean, if True, the parametric uncertainty is 
                        an integer
        :param default: optional argument for providing a default value
        
        '''
       
        super(ParameterUncertainty, self).__init__(values, name)
        if default: 
            self.default = default
        else: 
            self.default = abs(self.values[0]-self.values[1])
        
        self.type = "parameter"
        
        if len(values) != 2:
            raise EMAError("length of values for %s incorrect " % name)

        # self.dist should be a string. This string should be a key     
        # in the distributions dictionary 
        if integer:
            self.dist = "integer"
            self.params = (values[0], values[1]+1)
            self.default = int(round(self.default))
        else:
            self.dist = "uniform"
            #params for initializing self.dist
            self.params = (self.get_values()[0], 
                           self.get_values()[1]-self.get_values()[0])
        
    def get_default_value(self):
        ''' return default value'''
        
        return self.default

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
    
    def __init__(self, values, name, default = None):
        '''
        
        :param values: the values for specifying the uncertainty from which to 
                       sample. Values should be a collection.
        :param name: name of the uncertainty
        :param default: optional argument for providing a default value
        
        '''
        self.categories = values
        values = (0, len(values)-1)
        if default != None:
            default = self.invert(default)
        
        self.default = default
        super(CategoricalUncertainty, self).__init__(values, 
                                                     name, 
                                                     integer=True,
                                                     default=default)
        self.integer = True
                
    def transform(self, param):
        '''transform an integer to a category '''
        return self.categories[param]
    
    def invert(self, name):
        ''' transform a category to an integer'''
        return self.categories.index(name)

#==============================================================================
# test functions
#==============================================================================
#def test_uncertainties():
#    import ema_logging
#    ema_logging.log_to_stderr(ema_logging.INFO)
#    params = [
##              CategoricalUncertainty(('1', '5',  '10'), 
##                                        "blaat", 
##                                        default = '5'),
##              ParameterUncertainty((0, 1), "blaat2"),
#              ParameterUncertainty((0, 10), "blaat3"),
#              ParameterUncertainty((0, 5), "blaat4", integer=True)
#              ]
#
#    sampler = FullFactorialSampler()
#    a = sampler.generateDesign(params, 10)
#    a = [combo for combo in a[0]]
#    for entry in a:
#        print entry
#    
#    print len(a)
   
#=============================================================================
# running the module stand alone
#==============================================================================
#if __name__=='__main__':
#    test_uncertainties()