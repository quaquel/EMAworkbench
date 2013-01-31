'''
Created on 24 mei 2011

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

'''

__all__ = ['Outcome',
           'TIME']

TIME = "TIME"

class Outcome(object):
    '''
    Outcome class
    '''
    
    #: name of the outcome
    name = None
    
    #: boolean, indication of outcome is a time series or not
    time = False
    
    def __init__(self, name, time = False):
        '''
        init

        :param name: Name of the outcome.
        :param time: Boolean, specifies whether the outcome is a time
                     series or not (Default = False).  
        
        '''
        
        self.name = name
        self.time = time
    
    def __eq__ (self, other):
        comparison = [all(hasattr(self, key) == hasattr(other, key) and
                          getattr(self, key) == getattr(other, key) for key 
                          in self.__dict__.keys())]
        comparison.append(self.__class__ == other.__class__)
        return all(comparison)