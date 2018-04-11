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
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import abc
import collections
from threading import Lock

import numpy as np
import pandas as pd

from ..util import ema_logging, ema_exceptions
from .parameters import CategoricalParameter, IntegerParameter

#
# Created on 22 Jan 2013
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
#

__all__ = ['AbstractCallback',
           'DefaultCallback']


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
                         the interval at which to provide progress information
                         via logging.
    reporting_frequency: int, optional
                         the total number of progress logs


    Attributes
    ----------
    i : int
        a counter that keeps track of how many experiments have been saved
    reporting_interval : int, 
                         the interval between progress logs

    '''
    __metaclass__ = abc.ABCMeta

    i = 0
    reporting_interval = 100

    def __init__(self, uncertainties, outcomes, constraints, levers,
                 nr_experiments, reporting_interval=None,
                 reporting_frequency=10):
        if reporting_interval is None:
            reporting_interval = max(1,
                             int(round(nr_experiments/reporting_frequency)))

        self.reporting_interval = reporting_interval

    @abc.abstractmethod
    def __call__(self, experiment, outcomes, constraints):
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
        constraints: dict
                the outcomes dict

        '''
        #
        # TODO:: replace with optional tqdm based progress bar
        # http://thelivingpearl.com/2012/12/31/creating-progress-bars-with-python/
        # idea: have a bar that ships with workbench, which is used as a
        # fallback if tqdm is not available
        self.i += 1
        ema_logging.debug(str(self.i)+" cases completed")

        if self.i % self.reporting_interval == 0:
            ema_logging.info(str(self.i)+" cases completed")

    @abc.abstractmethod
    def get_results(self):
        """
        method for retrieving the results. Called after all experiments have
        been completed. Any extension of AbstractCallback needs to implement
        this method.
        """


class DefaultCallback(AbstractCallback):
    """
    default callback system
    callback can be used in performExperiments as a means for specifying
    the way in which the results should be handled. If no callback is
    specified, this default implementation is used. This one can be
    overwritten or replaced with a callback of your own design. For
    example if you prefer to store the result in a database or write
    them to a text file
    """
    i = 0
    cases = None
    results = {}

    shape_error_msg = "can only save up to 2d arrays, this array is {}d"
    constraint_error_msg = ('can only save 1d arrays for constraint, '
                            'this array is {}d')

    def __init__(self, uncs, levers, outcomes, constraints, nr_experiments,
                 reporting_interval=100, reporting_frequency=10):
        '''

        Parameters
        ----------
        uncs : list
                a list of the parameters over which the experiments
                are being run.
        outcomes : list
                   a list of outcomes
        constraints : list
                      a list of constraints
        nr_experiments : int
                         the total number of experiments to be executed
        reporting_interval : int, optional
                             the interval between progress logs
        reporting_frequency: int, optional
                             the total number of progress logs

        '''
        super(DefaultCallback, self).__init__(uncs, levers, outcomes,
                                              constraints, nr_experiments,
                                              reporting_interval, 
                                              reporting_frequency)
        self.i = 0
        self.cases = None
        self.results = {}
        self.lock = Lock()

        self.outcomes = [outcome.name for outcome in outcomes]

        self.constraintnames = [c.name for c in constraints]
        self.constraints = pd.DataFrame(np.empty((nr_experiments,
                                                  len(self.constraintnames))),
                                        columns=self.constraintnames)

        # determine data types of parameters
        self.dtypes = []
        self.parameters = []

        for parameter in uncs + levers:
            name = parameter.name
            self.parameters.append(name)
            dataType = float

            if isinstance(parameter, CategoricalParameter):
                dataType = object
            elif isinstance(parameter, IntegerParameter):
                dataType = int
            self.dtypes.append((str(name), dataType))
        self.dtypes.append((str('scenario_id'), object))
        self.dtypes.append((str('policy'), object))
        self.dtypes.append((str('model'), object))

        self.cases = np.empty((nr_experiments,), dtype=self.dtypes)
        self.cases[:] = np.NAN
        self.nr_experiments = nr_experiments

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
        case = tuple(case)
        self.cases[experiment.experiment_id] = case

    def _store_outcomes(self, case_id, outcomes):
        for outcome in self.outcomes:
            ema_logging.debug("storing {}".format(outcome))

            try:
                outcome_res = outcomes[outcome]
            except KeyError:
                message = "%s not specified as outcome in msi" % outcome
                ema_logging.debug(message)
            else:
                try:
                    self.results[outcome][case_id, ] = outcome_res
                except KeyError:
                    shape = np.asarray(outcome_res).shape

                    if len(shape) > 2:
                        message = self.shape_error_msg.format(len(shape))
                        raise ema_exceptions.EMAError(message)

                    shape = list(shape)
                    shape.insert(0, self.nr_experiments)

                    self.results[outcome] = np.empty(shape)
                    self.results[outcome][:] = np.NAN
                    self.results[outcome][case_id, ] = outcome_res

    def _store_constraints(self, experiment_id, constraints):
        temp_constraints = collections.defaultdict(lambda: np.nan)

        for constraint in self.constraintnames:
            ema_logging.debug("storing {}".format(constraint))

            try:
                constraint_res = constraints[constraint]
            except KeyError:
                ema_logging.debug(('{} not specified as '
                                   'constraint in msi').format(constraint))
            temp_constraints[constraint] = constraint_res

        self.constraints.iloc[experiment_id] = temp_constraints

    def __call__(self, experiment, outcomes, constraints):
        '''
        Method responsible for storing results. This method calls
        :meth:`super` first, thus utilizing the logging provided there.

        Parameters
        ----------
        experiment: Experiment instance
        outcomes: dict
                the outcomes dict
        constraints: dict
                the outcomes dict

        '''
        super(DefaultCallback, self).__call__(experiment, outcomes,
                                              constraints)

        self.lock.acquire()

        # store the case
        self._store_case(experiment)

        # store outcomes
        self._store_outcomes(experiment.experiment_id, outcomes)

        # store constraints
        self._store_constraints(experiment.experiment_id, constraints)

        self.lock.release()

    def get_results(self):
        return self.cases, self.results
