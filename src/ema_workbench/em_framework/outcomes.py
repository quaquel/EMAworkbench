'''
Module for outcome classes

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

# Created on 24 mei 2011
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['Outcome',
           'TIME']

TIME = "TIME"

class Outcome(object):
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

    name = None
    time = False
    
    def __init__(self, name, time=False):
        self.name = name
        self.time = time
    
    def __eq__ (self, other):
        comparison = [all(hasattr(self, key) == hasattr(other, key) and
                          getattr(self, key) == getattr(other, key) for key 
                          in self.__dict__.keys())]
        comparison.append(self.__class__ == other.__class__)
        return all(comparison)