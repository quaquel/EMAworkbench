'''
Module for outcome classes

'''
import abc
import collections
import numbers
import six

import pandas

from .util import Variable
from ema_workbench.util.ema_exceptions import EMAError
from ..util import get_module_logger

# Created on 24 mei 2011
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['Outcome', 'ScalarOutcome', 'ArrayOutcome', 'TimeSeriesOutcome',
           'Constraint']
_logger = get_module_logger(__name__)


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
               a callable to perform postprocessing on data retrieved
               from model
    expected_range : 2 tuple, optional
                     expected min and max value for outcome,
                     used by HyperVolume convergence metric
    shape : {tuple, None} optional

    Attributes
    ----------
    name : str
    kind : int
    variable_name : str
    function : callable
    shape : tuple

    '''
    __metaclass__ = abc.ABCMeta

    MINIMIZE = -1
    MAXIMIZE = 1
    INFO = 0

    def __init__(self, name, kind=INFO, variable_name=None,
                 function=None, expected_range=None,
                 shape=None):
        super(AbstractOutcome, self).__init__(name)

        if function is not None and not callable(function):
            raise ValueError('function must be a callable')
        if variable_name:
            if (not isinstance(variable_name, six.string_types)) and (
                    not all(isinstance(elem, six.string_types) for elem in variable_name)):
                raise ValueError(
                    'variable name must be a string or list of strings')
        if expected_range is not None and len(expected_range) != 2:
            raise ValueError('expected_range must be a min-max tuple')
        self.kind = kind
        self.variable_name = variable_name
        self.function = function
        self._expected_range = expected_range
        self.shape = shape

    def process(self, values):
        if self.function:
            var_names = self.variable_name

            n_variables = len(var_names)
            try:
                n_values = len(values)
            except TypeError:
                n_values = None

            if (n_values is None) and (n_variables == 1):
                return self.function(values)
            elif n_variables != n_values:
                raise ValueError(
                    ('number of variables is {}, '
                     'number of outputs is {}').format(
                        n_variables, n_values))
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
    variable_name : str, optional
                    if the name of the outcome in the underlying model
                    is different from the name of the outcome, you can
                    supply the variable name as an optional argument,
                    if not provided, defaults to name
    function : callable, optional
               a callable to perform postprocessing on data retrieved
               from model
    expected_range : 2 tuple, optional
                     expected min and max value for outcome,
                     used by HyperVolume convergence metric

    Attributes
    ----------
    name : str
    kind : int
    variable_name : str
    function : callable
    shape : tuple
    expected_range : tuple

    '''

    @property
    def expected_range(self):
        if self._expected_range is None:
            raise ValueError(
                'no expected_range is set for {}'.format(
                    self.variable_name))
        return self._expected_range

    @expected_range.setter
    def expected_range(self, expected_range):
        self._expected_range = expected_range

    def __init__(self, name, kind=AbstractOutcome.INFO, variable_name=None,
                 function=None, expected_range=None):
        super(ScalarOutcome, self).__init__(name, kind,
                                            variable_name=variable_name,
                                            function=function)
        self.expected_range = expected_range

    def process(self, values):
        values = super(ScalarOutcome, self).process(values)
        if not isinstance(values, numbers.Number):
            raise EMAError(
                f"outcome {self.name} should be a scalar, but is {type(values)}: {values}".format())
        return values


class ArrayOutcome(AbstractOutcome):
    '''Array Outcome class for n-dimensional collections

    Parameters
    ----------
    name : str
           Name of the outcome.
    variable_name : str, optional
                    if the name of the outcome in the underlying model
                    is different from the name of the outcome, you can
                    supply the variable name as an optional argument,
                    if not provided, defaults to name
    function : callable, optional
               a callable to perform postprocessing on data retrieved
               from model
    expected_range : 2 tuple, optional
                     expected min and max value for outcome,
                     used by HyperVolume convergence metric
    shape : {tuple, None}, optional

    Attributes
    ----------
    name : str
    kind : int
    variable_name : str
    function : callable
    shape : tuple
    expected_range : tuple


    '''

    def __init__(self, name, variable_name=None,
                 function=None, expected_range=None,
                 shape=None):
        super(
            ArrayOutcome,
            self).__init__(
            name,
            kind=AbstractOutcome.INFO,
            variable_name=variable_name,
            function=function,
            expected_range=expected_range,
            shape=shape)

    def process(self, values):
        values = super(ArrayOutcome, self).process(values)
        if not isinstance(values, collections.abc.Iterable):
            raise EMAError(
                "outcome {} should be a collection".format(self.name))
        return values


class TimeSeriesOutcome(ArrayOutcome):
    '''
    TimeSeries Outcome class

    Parameters
    ----------
    name : str
           Name of the outcome.
    variable_name : str, optional
                    if the name of the outcome in the underlying model
                    is different from the name of the outcome, you can
                    supply the variable name as an optional argument,
                    if not provided, defaults to name
    function : callable, optional
               a callable to perform postprocessing on data retrieved
               from model
    expected_range : 2 tuple, optional
                     expected min and max value for outcome,
                     used by HyperVolume convergence metric
    shape : {tuple, None}, optional

    Attributes
    ----------
    name : str
    kind : int
    variable_name : str
    function : callable
    shape : tuple
    expected_range : tuple

    '''

    def __init__(self, name, variable_name=None,
                 function=None, expected_range=None,
                 shape=None):
        super(
            TimeSeriesOutcome,
            self).__init__(
            name,
            variable_name=variable_name,
            function=function,
            expected_range=expected_range,
            shape=shape)


class Constraint(ScalarOutcome):
    '''Constraints class that can be used when defining constrained
    optimization problems.

    Parameters
    ----------
    name : str
    parameter_names : str or collection of str
    outcome_names : str or collection of str
    function : callable

    Attributes
    ----------
    name : str
    parameter_names : str, list of str
                      name(s) of the uncertain parameter(s) and/or
                      lever parameter(s) to which the constraint applies
    outcome_names : str, list of str
                    name(s) of the outcome(s) to which the constraint applies
    function : callable
               The function should return the distance from the feasibility
               threshold, given the model outputs with a variable name. The
               distance should be 0 if the constraint is met.

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
            raise ValueError('unknown type for ' + name)
        temp_outcomes.append(outcome)
    return temp_outcomes
