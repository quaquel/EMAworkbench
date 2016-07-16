'''
Module for outcome classes

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
from ema_workbench.em_framework.parameters import NamedObject

# Created on 24 mei 2011
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['ScalarOutcome', 'TimeSeriesOutcome']

class Outcome(NamedObject):
    '''
    Outcome class
    
    Parameters
    ----------
    name : str
           Name of the outcome.
    time: bool, optional
          specifies whether the outcome is a time series or not 
          (Default = False).  
    
    Attributes
    ----------
    name : str
    time : bool
           If true, outcome is a time series. Default is false.
    
    '''

    MINIMIZE = -1
    MAXIMIZE = 1
    INFO = 0
    
    time = False
    
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
    
    # TODO:: let it type an index / callable so you can transform a time series
    # to a value of interest
    
    def __init__(self, name, kind=Outcome.INFO):
        super(ScalarOutcome, self).__init__(name, kind)
        
class TimeSeriesOutcome(Outcome):
    
    def __init__(self, name, kind=Outcome.INFO, index=-1):
        super(TimeSeriesOutcome, self).__init__(name, kind)
        self.index = index
        
    
        