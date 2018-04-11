'''

Exceptions and warning used internally by the EMA workbench. In line with 
advice given in `PEP 8 <http://www.python.org/dev/peps/pep-0008/>`_.
'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

# Created on 31 mei 2011
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['EMAError',
           'EMAWarning',
           'CaseError',
           'EMAParallelError']


class EMAError(BaseException):
    '''
    Base EMA error
    '''
    def __init__(self, *args):
        self.args = args

    def __str__(self):
        if len(self.args) == 1:
            return str(self.args[0])
        else:
            return str(self.args)

    def __repr__(self):
        return "%s(*%s)" % (self.__class__.__name__, repr(self.args))
    
class EMAWarning(EMAError):
    '''
    base EMA warning class
    '''
    pass

class CaseError(EMAError):
    '''
    error to be used when a particular run creates an error. The character of 
    the error can be specified as the message, and the actual case that 
    gave rise to the error. 
    
    '''
    def __init__(self, message, case, policy=None):
        self.message = message
        self.case = case
        self.args = (message, case)
        try:
            self.policy = policy.name
        except AttributeError:
            self.policy = 'None'
    
    def __str__(self):
        keys = sorted(self.case.keys())
        
        
        c = ""
        for key in keys:
            value = self.case.get(key)
            c += key
            c += ":"
            c += str(value)
            c += ', '
        c+= 'policy:'+self.policy
        
        return self.message + ' case: {' + c + "}"

    def __repr__(self):
        return "%s case: %s " % (self.message, repr(self.case))        

class EMAParallelError(EMAError):
    '''
    parallel EMA error
    '''
    pass