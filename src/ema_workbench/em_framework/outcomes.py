'''
Module for outcome classes

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
import abc

from .util import NamedObject


# Created on 24 mei 2011
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['ScalarOutcome', 'TimeSeriesOutcome']

class Outcome(NamedObject):
    '''
    Base Outcome class
    
    Parameters
    ----------
    name : str
           Name of the outcome.
    kind : {INFO, MINIMZE, MAXIMIZE}, optional
    
    Attributes
    ----------
    name : str
    kind : int
    
    '''
    __metaclass__ = abc.ABCMeta

    MINIMIZE = -1
    MAXIMIZE = 1
    INFO = 0
    
    def __init__(self, name, kind=INFO):
        super(Outcome, self).__init__(name)
        self.kind = kind
    
    def __eq__ (self, other):
        comparison = [all(hasattr(self, key) == hasattr(other, key) and
                          getattr(self, key) == getattr(other, key) for key 
                          in self.__dict__.keys())]
        comparison.append(self.__class__ == other.__class__)
        return all(comparison)


class ScalarOutcome(Outcome):
    '''
    Scalar Outcome class
    
    Parameters
    ----------
    name : str
           Name of the outcome.
    kind : {INFO, MINIMZE, MAXIMIZE}, optional
    
    Attributes
    ----------
    name : str
    kind : int
    
    '''   
    
    def __init__(self, name, kind=Outcome.INFO):
        super(ScalarOutcome, self).__init__(name, kind)


class TimeSeriesOutcome(Outcome):
    '''
    TimeSeries Outcome class
    
    Parameters
    ----------
    name : str
           Name of the outcome.
    kind : {INFO, MINIMZE, MAXIMIZE}, optional
    reduce : callable, optional
             a callable which returns a scalar when called. Is only used
             when the outcome is used in an optimization context
    
    Raises
    ------
    ValueError
        if kind is MINIMIZE or MAXIMIZE and callable is not provided or
        not a callable
    
    Attributes
    ----------
    name : str
    kind : int
    reduce : callable
    
    
    '''   
    
    def __init__(self, name, kind=Outcome.INFO, reduce=None):
        super(TimeSeriesOutcome, self).__init__(name, kind)
        
        if (not self.kind==Outcome.INFO) and (not callable(reduce)):
            raise ValueError(('reduce needs to be specified when using'
                              ' TimeSeriesOutcome in optimization' ))
        self.reduce = reduce
        
    
        