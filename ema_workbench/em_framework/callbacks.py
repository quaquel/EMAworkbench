'''

This module provides an abstract base class for a callback and a default
implementation.

If you want to store the data in a way that is different from the
functionality provided by the default callback, you can write your own
extension of callback. For example, you can easily implement a callback
that stores the data in e.g. a NoSQL file.

The only method to implement is the __call__ magic method. To use logging of
progress, always call super.

'''
import abc
import csv
import os
import shutil

import numpy as np
import pandas as pd

from ..util import ema_exceptions, get_module_logger
from .parameters import (CategoricalParameter, IntegerParameter,
                         BooleanParameter)


#
# Created on 22 Jan 2013
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
#

__all__ = ['AbstractCallback',
           'DefaultCallback',
           'FileBasedCallback']
_logger = get_module_logger(__name__)


class AbstractCallback(object):
    '''
    Abstract base class from which different call back classes can be derived.
    Callback is responsible for storing the results of the runs.

    Parameters
    ----------
    uncs : list
            a list of the parameters over which the experiments
            are being run.
    outcomes : list
               a list of outcomes
    nr_experiments : int
                     the total number of experiments to be executed
    reporting_interval : int, optional
                         the interval at which to provide progress
                         information via logging.
    reporting_frequency: int, optional
                         the total number of progress logs


    Attributes
    ----------
    i : int
        a counter that keeps track of how many experiments have been
        saved
    reporting_interval : int,
                         the interval between progress logs

    '''
    __metaclass__ = abc.ABCMeta

    i = 0

    def __init__(self, uncertainties, outcomes, levers,
                 nr_experiments, reporting_interval=None,
                 reporting_frequency=10):
        if reporting_interval is None:
            reporting_interval = max(
                1, int(round(nr_experiments / reporting_frequency)))

        self.reporting_interval = reporting_interval

    @abc.abstractmethod
    def __call__(self, experiment, outcomes):
        '''
        Method responsible for storing results. The implementation in this
        class only keeps track of how many runs have been completed and
        logging this. Any extension of AbstractCallback needs to implement
        this method. If one want to use the logging provided here, call it via
        super.

        Parameters
        ----------
        experiment: Experiment instance
        outcomes: dict
                the outcomes dict

        '''
        #
        # TODO:: https://github.com/alexanderkuk/log-progress
        # can we detect whether we are running within Jupyter?
        # yes:
        # https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
        self.i += 1
        _logger.debug(str(self.i) + " cases completed")

        if self.i % self.reporting_interval == 0:
            _logger.info(str(self.i) + " cases completed")

    @abc.abstractmethod
    def get_results(self):
        """
        method for retrieving the results. Called after all experiments
        have been completed. Any extension of AbstractCallback needs to
        implement this method.
        """


class DefaultCallback(AbstractCallback):
    """
    default callback system
    callback can be used in perform_experiments as a means for
    specifying the way in which the results should be handled. If no
    callback is specified, this default implementation is used. This
    one can be overwritten or replaced with a callback of your own
    design. For example if you prefer to store the result in a database
    or write them to a text file
    """
    i = 0
    cases = None
    results = {}

    shape_error_msg = "can only save up to 2d arrays, this array is {}d"
    constraint_error_msg = ('can only save 1d arrays for constraint, '
                            'this array is {}d')

    def __init__(self, uncs, levers, outcomes, nr_experiments,
                 reporting_interval=100, reporting_frequency=10):
        '''

        Parameters
        ----------
        uncs : list
                a list of the parameters over which the experiments
                are being run.
        outcomes : list
                   a list of outcomes
        nr_experiments : int
                         the total number of experiments to be executed
        reporting_interval : int, optional
                             the interval between progress logs
        reporting_frequency: int, optional
                             the total number of progress logs

        '''
        super(DefaultCallback, self).__init__(uncs, levers, outcomes,
                                              nr_experiments,
                                              reporting_interval,
                                              reporting_frequency)
        self.i = 0
        self.cases = None
        self.results = {}

        self.outcomes = [outcome.name for outcome in outcomes]

        # determine data types of parameters
        columns = []
        dtypes = []
        self.parameters = []

        for parameter in uncs + levers:
            name = parameter.name
            self.parameters.append(name)
            dataType = 'float'

            if isinstance(parameter, CategoricalParameter):
                dataType = 'object'
            elif isinstance(parameter, BooleanParameter):
                dataType = 'bool'
            elif isinstance(parameter, IntegerParameter):
                dataType = 'int'
            columns.append(name)
            dtypes.append(dataType)

        for name in ['scenario', 'policy', 'model']:
            columns.append(name)
            dtypes.append('object')

        df = pd.DataFrame(index=np.arange(nr_experiments))

        for name, dtype in zip(columns, dtypes):
            df[name] = pd.Series(dtype=dtype)
        self.cases = df
        self.nr_experiments = nr_experiments

        for outcome in outcomes:
            shape = outcome.shape
            if shape is not None:
                shape = (nr_experiments, ) + shape
                data = np.empty(shape)
                data[:] = np.nan
                self.results[outcome.name] = data

    def _store_case(self, experiment):
        scenario = experiment.scenario
        policy = experiment.policy
        index = experiment.experiment_id

        self.cases.at[index, 'scenario'] = scenario.name
        self.cases.at[index, 'policy'] = policy.name
        self.cases.at[index, 'model'] = experiment.model_name

        for k, v in scenario.items():
            self.cases.at[index, k] = v

        for k, v in policy.items():
            self.cases.at[index, k] = v

    def _store_outcomes(self, case_id, outcomes):
        for outcome in self.outcomes:
            _logger.debug("storing {}".format(outcome))

            try:
                outcome_res = outcomes[outcome]
            except KeyError:
                message = "%s not specified as outcome in msi" % outcome
                _logger.debug(message)
            else:
                try:
                    self.results[outcome][case_id, ] = outcome_res
                except KeyError:
                    a = np.asarray(outcome_res)

                    shape = a.shape

                    if len(shape) > 2:
                        message = self.shape_error_msg.format(len(shape))
                        raise ema_exceptions.EMAError(message)

                    shape = list(shape)
                    shape.insert(0, self.nr_experiments)

                    self.results[outcome] = np.empty(shape, dtype=a.dtype)
                    self.results[outcome][:] = np.nan
                    self.results[outcome][case_id, ] = outcome_res

    def __call__(self, experiment, outcomes):
        '''
        Method responsible for storing results. This method calls
        :meth:`super` first, thus utilizing the logging provided there.

        Parameters
        ----------
        experiment: Experiment instance
        outcomes: dict
                the outcomes dict

        '''
        super(DefaultCallback, self).__call__(experiment, outcomes)

        # store the case
        self._store_case(experiment)

        # store outcomes
        self._store_outcomes(experiment.experiment_id, outcomes)

    def get_results(self):
        return self.cases, self.results


class FileBasedCallback(AbstractCallback):
    '''
    Callback that stores data in csv files while running

    Parameters
    ----------
    uncs : collection of Parameter instances
    levers : collection of Parameter instances
    outcomes : collection of Outcome instances
    nr_experiments : int
    reporting_interval : int, optional
    reporting_frequency : int, optional

    the data is stored in ./temp, relative to the current
    working directory. If this directory already exists, it will be
    overwritten.

    Warnings
    --------
    This class is still in beta. API is expected to change over the
    coming months.

    '''

    def __init__(self, uncs, levers, outcomes, nr_experiments,
                 reporting_interval=100, reporting_frequency=10):
        super(
            FileBasedCallback,
            self).__init__(
            uncs,
            levers,
            outcomes,
            nr_experiments,
            reporting_interval=reporting_interval,
            reporting_frequency=reporting_frequency)

        self.i = 0
        self.nr_experiments = nr_experiments
        self.outcomes = [outcome.name for outcome in outcomes]
        self.parameters = [parameter.name for parameter in uncs + levers]

        self.directory = os.path.abspath('./temp')
        if os.path.exists(self.directory):
            shutil.rmtree(self.directory)
        os.makedirs(self.directory)

        self.experiments_fh = open(os.path.join(self.directory,
                                                'experiments.csv'), 'w')

        header = self.parameters + ['scenario_id', 'policy', 'model']
        writer = csv.writer(self.experiments_fh)
        writer.writerow(header)

        self.outcome_fhs = {}
        for outcome in self.outcomes:
            self.outcome_fhs[outcome] = open(
                os.path.join(self.directory, outcome + '.csv'), 'w')

    def _store_case(self, experiment):
        scenario = experiment.scenario
        policy = experiment.policy

        case = []
        for parameter in self.parameters:
            try:
                value = scenario[parameter]
            except KeyError:
                try:
                    value = policy[parameter]
                except KeyError:
                    value = np.nan
            finally:
                case.append(value)

        case.append(scenario.name)
        case.append(policy.name)
        case.append(experiment.model_name)

        writer = csv.writer(self.experiments_fh)
        writer.writerow(case)

    def _store_outcomes(self, outcomes):
        for outcome in self.outcomes:
            data = outcomes[outcome]

            try:
                data = [str(entry) for entry in data]
            except TypeError:
                data = [str(data)]

            fh = self.outcome_fhs[outcome]
            writer = csv.writer(fh)
            writer.writerow(data)

    def __call__(self, experiment, outcomes):
        '''
        Method responsible for storing results. This method calls
        :meth:`super` first, thus utilizing the logging provided there.

        Parameters
        ----------
        experiment: Experiment instance
        outcomes: dict
                the outcomes dict

        '''
        super(FileBasedCallback, self).__call__(experiment, outcomes)

        # store the case
        self._store_case(experiment)

        # store outcomes
        self._store_outcomes(outcomes)

    def get_results(self):

        # TODO:: metadata

        self.experiments_fh.close()
        for value in self.outcome_fhs.items():
            value.close

        return self.directory
