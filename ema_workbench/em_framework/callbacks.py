"""

This module provides an abstract base class for a callback and a default
implementation.

If you want to store the data in a way that is different from the
functionality provided by the default callback, you can write your own
extension of callback. For example, you can easily implement a callback
that stores the data in e.g. a NoSQL file.

The only method to implement is the __call__ magic method. To use logging of
progress, always call super.

"""

import abc
import csv
import os
import shutil

import numpy as np
import pandas as pd

from .parameters import CategoricalParameter, IntegerParameter, BooleanParameter
from .util import ProgressTrackingMixIn
from ..util import ema_exceptions, get_module_logger

#
# Created on 22 Jan 2013
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
#

__all__ = ["AbstractCallback", "DefaultCallback", "FileBasedCallback"]
_logger = get_module_logger(__name__)


class AbstractCallback(ProgressTrackingMixIn, metaclass=abc.ABCMeta):
    """
    Abstract base class from which different call back classes can be derived.
    Callback is responsible for storing the results of the runs.

    Parameters
    ----------
    uncertainties : list
                    list of uncertain parameters
    levers : list
             list of lever parameters
    outcomes : list
               a list of outcomes
    nr_experiments : int
                     the total number of experiments to be executed
    reporting_interval : int, optional
                         the interval between progress logs
    reporting_frequency: int, optional
                         the total number of progress logs
    log_progress : bool, optional
                   if true, progress is logged, if false, use
                   tqdm progress bar.


    Attributes
    ----------
    i : int
        a counter that keeps track of how many experiments have been
        saved
    nr_experiments: int
    outcomes : list
    parameters : list
                 combined list of uncertain parameters and lever parameters
    reporting_interval : int,
                         the interval between progress logs

    """

    def __init__(
        self,
        uncertainties,
        levers,
        outcomes,
        nr_experiments,
        reporting_interval=None,
        reporting_frequency=10,
        log_progress=False,
    ):
        super().__init__(nr_experiments, reporting_frequency, _logger, log_progress)

        self.i = 0
        self.nr_experiments = nr_experiments
        self.outcomes = outcomes
        self.parameters = uncertainties + levers

        if reporting_interval is None:
            reporting_interval = max(1, int(round(nr_experiments / reporting_frequency)))

        self.reporting_interval = reporting_interval

    @abc.abstractmethod
    def __call__(self, experiment, outcomes):
        """
        Method responsible for storing results.

        The implementation in this class only keeps track of how many runs
        have been completed and logging this. Any extension of
        AbstractCallback needs to implement this method. If one want
        to use the logging provided here, call it via super.

        Parameters
        ----------
        experiment: Experiment instance
        outcomes: dict
                  the outcomes dict

        """
        super().__call__(1)

    @abc.abstractmethod
    def get_results(self):
        """
        method for retrieving the results. Called after all experiments
        have been completed. Any extension of AbstractCallback needs to
        implement this method.
        """


class DefaultCallback(AbstractCallback):
    """Default callback class

    Parameters
    ----------
    uncertainties : list
                    list of uncertain parameters
    levers : list
             list of lever parameters
    outcomes : list
               a list of outcomes
    nr_experiments : int
                     the total number of experiments to be executed
    reporting_interval : int, optional
                         the interval between progress logs
    reporting_frequency: int, optional
                         the total number of progress logs
    log_progress : bool, optional
                   if true, progress is logged, if false, use
                   tqdm progress bar.


    Callback can be used in perform_experiments as a means for
    specifying the way in which the results should be handled. If no
    callback is specified, this default implementation is used. This
    one can be overwritten or replaced with a callback of your own
    design. For example if you prefer to store the result in a database
    or write them to a text file.
    """

    shape_error_msg = "can only save up to 2d arrays, this array is {}d"
    constraint_error_msg = "can only save 1d arrays for constraint, " "this array is {}d"

    def __init__(
        self,
        uncertainties,
        levers,
        outcomes,
        nr_experiments,
        reporting_interval=100,
        reporting_frequency=10,
        log_progress=False,
    ):
        """

        Parameters
        ----------
        uncertainties : list
                        list of uncertain parameters
        levers : list
                 list of lever parameters
        outcomes : list
                   a list of outcomes
        nr_experiments : int
                         the total number of experiments to be executed
        reporting_interval : int, optional
                             the interval between progress logs
        reporting_frequency: int, optional
                             the total number of progress logs
        log_progress : bool, optional
                       if true, progress is logged, if false, use
                       tqdm progress bar.

        """
        super().__init__(
            uncertainties,
            levers,
            outcomes,
            nr_experiments,
            reporting_interval,
            reporting_frequency,
            log_progress,
        )
        self.cases = np.empty((nr_experiments, len(uncertainties) + len(levers)), dtype=object)
        self.uncertainty_and_lever_labels = [(entry.name, "") for entry in uncertainties + levers]
        self.uncertainties = [u.name for u in uncertainties]
        self.levers = [l.name for l in levers]
        self.results = {}

        dtypes = []
        for parameter in self.parameters:
            dtype = "float"
            if isinstance(parameter, BooleanParameter):
                dtype = "bool"
            elif isinstance(parameter, CategoricalParameter):
                dtype = "object"
            elif isinstance(parameter, IntegerParameter):
                dtype = "int"
            dtypes.append((parameter.name, dtype))

        dtypes.extend(
            [
                ("scenario", "object"),
                ("policy", "object"),
                ("model", "object"),
            ]
        )
        self.dtypes = dtypes
        self.cases = np.empty(nr_experiments, dtype=dtypes)

        for outcome in self.outcomes:
            shape = outcome.shape
            if shape is not None:
                shape = (nr_experiments,) + shape
                self.results[outcome.name] = self._setup_outcomes_array(shape, dtype=outcome.dtype)

    def _store_case(self, experiment):
        scenario = experiment.scenario
        policy = experiment.policy
        index = experiment.experiment_id

        self.cases[index] = (
            tuple([scenario[u] for u in self.uncertainties])
            + tuple([policy[l] for l in self.levers])
            + (scenario.name, policy.name, experiment.model_name)
        )

    def _store_outcomes(self, case_id, outcomes):
        for outcome in self.outcomes:
            outcome_name = outcome.name

            try:
                outcome_res = outcomes[outcome_name]
            except KeyError:
                message = f"outcome {outcome_name} not specified as outcome in model(s)"
                _logger.debug(message)
            else:
                try:
                    self.results[outcome_name][case_id,] = outcome_res
                except KeyError:
                    data = np.asarray(outcome_res)
                    shape = data.shape
                    if len(shape) > 2:
                        message = self.shape_error_msg.format(len(shape))
                        raise ema_exceptions.EMAError(message)

                    shape = list(shape)
                    shape.insert(0, self.nr_experiments)

                    self.results[outcome_name] = self._setup_outcomes_array(shape, data.dtype)
                    self.results[outcome_name][case_id,] = outcome_res

    def __call__(self, experiment, outcomes):
        """
        Method responsible for storing results. This method calls
        :meth:`super` first, thus utilizing the logging provided there.

        Parameters
        ----------
        experiment: Experiment instance
        outcomes: dict
                the outcomes dict

        """
        super().__call__(experiment, outcomes)
        self._store_case(experiment)
        self._store_outcomes(experiment.experiment_id, outcomes)

    def get_results(self):
        results = {}
        for k, v in self.results.items():
            if not np.ma.is_masked(v):
                results[k] = v.data
            else:
                _logger.warning("some experiments have failed, returning masked result arrays")
                results[k] = v

        cases = pd.DataFrame.from_records(self.cases)

        # we want to ensure the dtypes for the columns in the experiments dataframe match
        # the type of uncertainty. The exception is needed in case their are missing values (i.e. nans).
        # nans can only ever be a float.
        for name, dtype in self.dtypes:
            try:
                if dtype == "object":
                    dtype = "category"
                cases[name] = cases[name].astype(dtype)
            except Exception:
                pass

        return cases, results

    def _setup_outcomes_array(self, shape, dtype):
        array = np.ma.empty(shape, dtype=dtype)
        array.mask = True
        return array


class FileBasedCallback(AbstractCallback):
    """Callback that stores data in csv files while running th model

    Parameters
    ----------
    uncertainties : list
                    list of uncertain parameters
    levers : list
             list of lever parameters
    outcomes : list
               a list of outcomes
    nr_experiments : int
                     the total number of experiments to be executed
    reporting_interval : int, optional
                         the interval between progress logs
    reporting_frequency: int, optional
                         the total number of progress logs
    log_progress : bool, optional
                   if true, progress is logged, if false, use
                   tqdm progress bar.

    Warnings
    --------
    This class is still in beta.
    the data is stored in ./temp, relative to the current
    working directory. If this directory already exists, it will be
    overwritten.

    """

    def __init__(
        self,
        uncertainties,
        levers,
        outcomes,
        nr_experiments,
        reporting_interval=100,
        reporting_frequency=10,
    ):
        super().__init__(
            uncertainties,
            levers,
            outcomes,
            nr_experiments,
            reporting_interval=reporting_interval,
            reporting_frequency=reporting_frequency,
        )

        self.directory = os.path.abspath("./temp")
        if os.path.exists(self.directory):
            shutil.rmtree(self.directory)
        os.makedirs(self.directory)

        self.experiments_fh = open(os.path.join(self.directory, "experiments.csv"), "w")

        # write experiments.csv header row
        header = [p.name for p in self.parameters] + ["scenario_id", "policy", "model"]
        writer = csv.writer(self.experiments_fh)
        writer.writerow(header)

        self.outcome_fhs = {}
        for outcome in self.outcomes:
            name = outcome.name
            self.outcome_fhs[name] = open(os.path.join(self.directory, f"{name}.csv"), "w")

    def _store_case(self, experiment):
        scenario = experiment.scenario
        policy = experiment.policy

        case = []
        for parameter in self.parameters:
            name = parameter.name
            try:
                value = scenario[name]
            except KeyError:
                try:
                    value = policy[name]
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
            name = outcome.name
            data = outcomes[name]

            try:
                data = [str(entry) for entry in data]
            except TypeError:
                data = [str(data)]

            fh = self.outcome_fhs[name]
            writer = csv.writer(fh)
            writer.writerow(data)

    def __call__(self, experiment, outcomes):
        """
        Method responsible for storing results. This method calls
        :meth:`super` first, thus utilizing the logging provided there.

        Parameters
        ----------
        experiment: Experiment instance
        outcomes: dict
                the outcomes dict

        """
        super().__call__(experiment, outcomes)

        # store the case
        self._store_case(experiment)

        # store outcomes
        self._store_outcomes(outcomes)

    def get_results(self):
        # TODO:: metadata

        self.experiments_fh.close()
        for value in self.outcome_fhs.values():
            value.close()

        return self.directory
