'''
Module for outcome classes

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
import abc
import warnings

import pandas

from .util import NamedObject


# Created on 24 mei 2011
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['Outcome', 'ScalarOutcome', 'TimeSeriesOutcome']


# TODO:: have two names for an outcome, one is the name as it is known
# to the user, the other is the name of the variable in the model
# the return value for the variable can be passed to a callable known
# to the outcome. This makes is possible to have e.g. a peak infection, which
# takes a time series on the infection, and finds the maximum over the time
# series

# TODO:: we need a output map, this map calls the outcomes to do
# any transformation as outlined above

def Outcome(name, time=False):
    if time:
        warnings.warn('Deprecated, use TimeSeriesOutcome instead')
        return ScalarOutcome(name)
    else:
        warnings.warn('Deprecated, use ScalarOutcome instead')
        return TimeSeriesOutcome(name)
    

class AbstractOutcome(NamedObject):
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
        super(AbstractOutcome, self).__init__(name)
        self.kind = kind
    
    def __eq__ (self, other):
        comparison = [all(hasattr(self, key) == hasattr(other, key) and
                          getattr(self, key) == getattr(other, key) for key 
                          in self.__dict__.keys())]
        comparison.append(self.__class__ == other.__class__)
        return all(comparison)


class ScalarOutcome(AbstractOutcome):
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
    
    def __init__(self, name, kind=AbstractOutcome.INFO):
        super(ScalarOutcome, self).__init__(name, kind)


class TimeSeriesOutcome(AbstractOutcome):
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
    
    def __init__(self, name, kind=AbstractOutcome.INFO, reduce=None):
        super(TimeSeriesOutcome, self).__init__(name, kind)
        
        if (not self.kind==AbstractOutcome.INFO) and (not callable(reduce)):
            raise ValueError(('reduce needs to be specified when using'
                              ' TimeSeriesOutcome in optimization' ))
        self.reduce = reduce
        

def create_outcomes(outcomes):
    '''Helper function for creating multiple outcomes
    
    Parameters
    ----------
    outcomes : str, list of dict, or dataframe
               if str, should be path to csv file
               each entry should specify name and the type, where type
               is 'scalar' or 'timeseries'
    
    
    '''
    
    [{'name':'a', 'type':'scalar'}]
    [('a','scalar')('b', 'timeseries')]
    {'a':'scalar', 'b':'time_series'}
    
    if isinstance(outcomes, list):
        outcomes = {str(i):entry for i, entry in enumerate(outcomes)}
        outcomes = pandas.DataFrame.from_dict(outcomes)
    elif isinstance(outcomes, str):
        outcomes = pandas.read_csv(outcomes)
    elif not isinstance(outcomes, pandas.DataFrame):
        raise ValueError('unable to convert outcomes to a dataframe')
    
    temp_outcomes = []
    for _, row in outcomes.iteritems():
        name = row.ix['name']
        kind = row.ix['type']
        
        if kind=='scalar':
            outcome = ScalarOutcome(name)
        elif kind=='timeseries':
            outcome = TimeSeriesOutcome(name)
        else:
            raise ValueError('unknown type for '+name)
        temp_outcomes.append(outcome)
    return temp_outcomes
        