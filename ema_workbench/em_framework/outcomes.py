'''
Module for outcome classes

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
import abc
import collections
import numbers
import six
import warnings

import pandas

from .util import Variable
from ema_workbench.util.ema_exceptions import EMAError


# Created on 24 mei 2011
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['Outcome', 'ScalarOutcome', 'TimeSeriesOutcome']


def Outcome(name, time=False):
    if time:
        warnings.warn('Deprecated, use TimeSeriesOutcome instead')
        return ScalarOutcome(name)
    else:
        warnings.warn('Deprecated, use ScalarOutcome instead')
        return TimeSeriesOutcome(name)


class AbstractOutcome(Variable):
    '''
    Base Outcome class

    Parameters
    ----------
    name : str
           Name of the outcome.
    kind : {INFO, MINIMZE, MAXIMIZE}, optional
    variable_name : str, optional
                    if the name of the outcome in the underlying model
                    is different from the name of the outcome, you can 
                    supply the variable name as an optional argument,
                    if not provided, defaults to name
    function : callable, optional
               a callable to perform postprocessing on data retrieved from
               model

    Attributes
    ----------
    name : str
    kind : int

    '''
    __metaclass__ = abc.ABCMeta

    MINIMIZE = -1
    MAXIMIZE = 1
    INFO = 0

    def __init__(self, name, kind=INFO, variable_name=None, function=None):
        super(AbstractOutcome, self).__init__(name)

        if function is not None and not callable(function):
            raise ValueError('function must be a callable')
        if variable_name:
            if (not isinstance(variable_name, six.string_types)) and (not all(isinstance(elem, six.string_types) for elem in variable_name)):
                raise ValueError(
                    'variable name must be a string or list of strings')

        self.kind = kind
        self.variable_name = variable_name
        self.function = function

    def process(self, values):
        if self.function:
            var_names = self.variable_name

            n_variables = len(var_names)
            try:
                n_values = len(values)
            except TypeError:
                n_values = None

            if (n_values == None) and (n_variables == 1):
                return self.function(values)
            elif n_variables != n_values:
                raise ValueError(('number of variables is {}, '
                                  'number of outputs is {}').format(n_variables, n_values))
            else:
                return self.function(*values)
        else:
            if len(values) > 1:
                raise EMAError(('more than one value returned without '
                                'processing function'))

            return values[0]

    def __eq__(self, other):
        comparison = [all(hasattr(self, key) == hasattr(other, key) and
                          getattr(self, key) == getattr(other, key) for key
                          in self.__dict__.keys())]
        comparison.append(self.__class__ == other.__class__)
        return all(comparison)

    def __repr__(self, *args, **kwargs):
        klass = self.__class__.__name__
        name = self.name

        rep = '{}(\'{}\''.format(klass, name)

        if self.variable_name != [self.name]:
            rep += ', variable_name={}'.format(self.variable_name)
        if self.function:
            rep += ', function={}'.format(self.function)

        rep += ')'

        return rep


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

    def __init__(self, name, kind=AbstractOutcome.INFO, variable_name=None,
                 function=None):
        super(ScalarOutcome, self).__init__(name, kind,
                                            variable_name=variable_name,
                                            function=function)

    def process(self, values):
        values = super(ScalarOutcome, self).process(values)
        if not isinstance(values, numbers.Number):
            raise EMAError("outcome {} should be a scalar".format(self.name))
        return values


class TimeSeriesOutcome(AbstractOutcome):
    '''
    TimeSeries Outcome class

    Parameters
    ----------
    name : str
           Name of the outcome.
    variable_name :  str, or collection of str
    function : callable, optional

    Attributes
    ----------
    name : str
    kind : int
    function : callable

    '''

    def __init__(self, name, variable_name=None,
                 function=None):
        super(TimeSeriesOutcome, self).__init__(name, kind=AbstractOutcome.INFO,
                                                variable_name=variable_name,
                                                function=function)

    def process(self, values):
        values = super(TimeSeriesOutcome, self).process(values)
        if not isinstance(values, collections.Iterable):
            raise EMAError(
                "outcome {} should be a collection".format(self.name))
        return values


class Constraint(ScalarOutcome):
    '''Constraints class that can be used when defining constrained 
    optimization problems.

    Parameters
    ----------
    name : str
    variable_name : str or collection of str
    function : callable

    The function should return the distance from the feasibility threshold,
    given the model outputs with a variable name. The distance should be
    0 if the constraint is met.  

    '''

    def __init__(self, name, parameter_names=None, outcome_names=None,
                 function=None):
        assert callable(function)
        if not parameter_names:
            parameter_names = []
        elif isinstance(parameter_names, six.string_types):
            parameter_names = [parameter_names]

        if not outcome_names:
            outcome_names = []
        elif isinstance(outcome_names, six.string_types):
            outcome_names = [outcome_names]

        variable_names = parameter_names + outcome_names

        super(Constraint, self).__init__(name, kind=AbstractOutcome.INFO,
                                         variable_name=variable_names,
                                         function=function)

        self.parameter_names = parameter_names
        self.outcome_names = outcome_names

    def process(self, values):
        value = super(Constraint, self).process(values)
        assert value >= 0
        return value


def create_outcomes(outcomes, **kwargs):
    '''Helper function for creating multiple outcomes

    Parameters
    ----------
    outcomes : DataFrame, or something convertable to a DataFrame
               in case of string, the string will be passed

    Returns
    -------
    list

    '''

    if isinstance(outcomes, six.string_types):
        outcomes = pandas.read_csv(outcomes, **kwargs)
    elif not isinstance(outcomes, pandas.DataFrame):
        outcomes = pandas.DataFrame.from_dict(outcomes)

    for entry in ['name', 'type']:
        if entry not in outcomes.columns:
            raise ValueError('no {} column in dataframe'.format(entry))

    temp_outcomes = []
    for _, row in outcomes.iterrows():
        name = row['name']
        kind = row['type']

        if kind == 'scalar':
            outcome = ScalarOutcome(name)
        elif kind == 'timeseries':
            outcome = TimeSeriesOutcome(name)
        else:
            raise ValueError('unknown type for '+name)
        temp_outcomes.append(outcome)
    return temp_outcomes
