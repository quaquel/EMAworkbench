'''

This module contains various classes that can be used for specifying different
types of uncertainties.

This module is deprecated. Use parameters instead

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import warnings

from .parameters import RealParameter, CategoricalParameter, IntegerParameter

# Created on 16 aug. 2011
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


__all__ = ['ParameterUncertainty', 'CategoricalUncertainty']

def ParameterUncertainty(values, name, integer=False, factorial=False,
                         resolution=[]):
        if integer:
            warnings.warn(('ParameterUncertainty is deprecated use '
                           'IntegerParameter instead'))
            return IntegerParameter(name, values[0], values[1],
                                    resolution=resolution)
        else:
            warnings.warn(('ParameterUncertainty is deprecated use '
                           'RealParameter instead'))
            return RealParameter(name, values[0], values[1], 
                                 resolution=resolution)
    

def CategoricalUncertainty(values, name, factorial=False):
    warnings.warn('deprecated use CategoricalParameter instead')
    
    return CategoricalParameter(name, values)