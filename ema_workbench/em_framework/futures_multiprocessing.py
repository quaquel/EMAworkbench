"""
support for using the multiprocessing library in combination with the workbench

"""

import logging
import multiprocessing
import os
import queue
import random
import shutil
import string
import sys
import threading
import traceback
from logging import handlers
import warnings

from .experiment_runner import ExperimentRunner
from .model import AbstractModel
from .util import NamedObjectMap
from ..util import get_module_logger, ema_logging
from .evaluators import BaseEvaluator, experiment_generator
from .futures_util import setup_working_directories, finalizer, determine_rootdir

# Created on 22 Feb 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ["MultiprocessingEvaluator"]

_logger = get_module_logger(__name__)


def initializer(*args):
    """initializer for a worker process

    Parameters
    ----------
    models : list of AbstractModel instances


    This function initializes the worker. This entails
    * initializing the experiment runner
    * setting up the working directory
    * setting up the logging
    """
    global experiment_runner, current_process

    current_process = multiprocessing.current_process()
    models, queue, log_level, root_dir = args

    # setup the experiment runner
    msis = NamedObjectMap(AbstractModel)
    msis.extend(models)
    experiment_runner = ExperimentRunner(msis)

    # setup the logging
    setup_logging(queue, log_level)

    # setup the working directories
    # make a root temp
    # copy each model directory
    tmpdir = setup_working_directories(models, root_dir)

    # register a cleanup finalizer function
    # remove the root temp
    if tmpdir:
        multiprocessing.util.Finalize(
            None, finalizer(experiment_runner), args=(os.path.abspath(tmpdir),), exitpriority=10
        )


def setup_logging(queue, log_level):
    """helper function for enabling logging from the workers to the main
    process

    Parameters
    ----------
    queue : multiprocessing.Queue instance
    log_level : int

    """

    # create a logger
    logger = logging.getLogger(ema_logging.LOGGER_NAME + ".subprocess")
    ema_logging._logger = logger
    logger.handlers = []

    # add the handler
    handler = handlers.QueueHandler(queue)
    handler.setFormatter(ema_logging.LOG_FORMAT)
    logger.addHandler(handler)

    # set the log_level
    logger.setLevel(log_level)


def worker(experiment):
    """the worker function for executing an individual experiment

    Parameters
    ----------
    experiment : dict

    """
    return experiment, experiment_runner.run_experiment(experiment)


class LogQueueReader(threading.Thread):
    """

    thread to write subprocesses log records to main process log

    This thread reads the records written by subprocesses and writes them to
    the handlers defined in the main process's handlers.

    found `online <https://stackoverflow.com/questions/641420/how-should-i-log-while-using-multiprocessing-in-python>`_

    TODO:: should be generalized with logwatcher used with ipyparallel
    TODO:: might be replaced by QueueListener from logging.handlers

    """

    def __init__(self, queue):
        threading.Thread.__init__(self, name="log queue reader")
        self.queue = queue
        self.daemon = True

    def run(self):
        """
        read from the queue and write to the log handlers

        The logging documentation says logging is thread safe, so there
        shouldn't be contention between normal logging (from the main
        process) and this thread.

        Note that we're using the name of the original logger.

        """

        while True:
            try:
                record = self.queue.get()
                # get the logger for this record
                if record is None:
                    _logger.debug("no record received from queue")
                    break

                logger = logging.getLogger(record.name)
                logger.callHandlers(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except EOFError:
                break
            except TypeError:
                break
            except BaseException:
                traceback.print_exc(file=sys.stderr)


class ExperimentFeeder(threading.Thread):
    def __init__(self, pool, results_queue, experiments):
        threading.Thread.__init__(self, name="task feeder")
        self.pool = pool
        self.experiments = experiments
        self.results_queue = results_queue

        self.daemon = True

    def run(self):
        for experiment in self.experiments:
            result = self.pool.apply_async(worker, [experiment])
            self.results_queue.put(result)


class ResultsReader(threading.Thread):
    def __init__(self, queue, callback):
        threading.Thread.__init__(self, name="results reader")
        self.queue = queue
        self.callback = callback
        self.daemon = True

    def run(self):
        while True:
            try:
                result = self.queue.get()
                # get the logger for this record
                if result is None:
                    _logger.debug("no record received from queue")
                    break

                self.callback(*result.get())
            except (KeyboardInterrupt, SystemExit):
                raise
            except EOFError:
                break
            except TypeError:
                break
            except BaseException:
                traceback.print_exc(file=sys.stderr)


def add_tasks(n_processes, pool, experiments, callback):
    """add experiments to pool

    Parameters
    ----------
    n_processes  : int
    pool : Pool instance
    experiments : collection
    callback : callable

    """
    # by limiting task queue, we avoid putting all experiments on queue in
    # one go
    results_queue = queue.Queue(maxsize=5 * n_processes)

    feeder = ExperimentFeeder(pool, results_queue, experiments)
    reader = ResultsReader(results_queue, callback)
    feeder.start()
    reader.start()

    feeder.join()
    results_queue.put(None)
    reader.join()


class MultiprocessingEvaluator(BaseEvaluator):
    """evaluator for experiments using a multiprocessing pool

    Parameters
    ----------
    msis : collection of models
    n_processes : int (optional)
                  A negative number can be inputted to use the number of logical cores minus the negative cores.
                  For example, on a 12 thread processor, -2 results in using 10 threads.
    max_tasks : int (optional)


    note that the maximum number of available processes is either multiprocessing.cpu_count()
    and in case of windows, this never can be higher then 61

    """

    def __init__(self, msis, n_processes=None, maxtasksperchild=None, **kwargs):
        super().__init__(msis, **kwargs)
        self.root_dir = None
        self._pool = None

        # Calculate n_processes if negative value is inputted
        max_processes = multiprocessing.cpu_count()
        if sys.platform == "win32":
            # on windows the max number of processes is currently
            # still limited to 61
            max_processes = min(max_processes, 61)

        if isinstance(n_processes, int):
            if n_processes > 0:
                if max_processes < n_processes:
                    warnings.warn(
                        f"The number of processes cannot be more then {max_processes}", UserWarning
                    )
                self.n_processes = min(n_processes, max_processes)
            else:
                self.n_processes = max(max_processes + n_processes, 1)
        elif n_processes is None:
            self.n_processes = max_processes
        else:
            raise ValueError(f"max_processes must be an integer or None, not {type(n_processes)}")

        self.maxtasksperchild = maxtasksperchild

    def initialize(self):
        log_queue = multiprocessing.Queue()

        log_queue_reader = LogQueueReader(log_queue)
        log_queue_reader.start()

        try:
            loglevel = ema_logging._rootlogger.getEffectiveLevel()
        except AttributeError:
            loglevel = 30

        # check if we need a working directory
        self.root_dir = determine_rootdir(self._msis)

        self._pool = multiprocessing.Pool(
            self.n_processes,
            initializer,
            (self._msis, log_queue, loglevel, self.root_dir),
            self.maxtasksperchild,
        )
        self.n_processes = self._pool._processes
        _logger.info(f"pool started with {self.n_processes} workers")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _logger.info("terminating pool")

        if exc_type is not None:
            # When an exception is thrown stop accepting new jobs
            # and abort pending jobs without waiting.
            self._pool.terminate()
            return False

        super().__exit__(exc_type, exc_value, traceback)

    def finalize(self):
        # Stop accepting new jobs and wait for pending jobs to finish.
        self._pool.close()
        self._pool.join()

        if self.root_dir:
            shutil.rmtree(self.root_dir)

    def evaluate_experiments(self, scenarios, policies, callback, combine="factorial"):
        ex_gen = experiment_generator(scenarios, self._msis, policies, combine=combine)
        add_tasks(self.n_processes, self._pool, ex_gen, callback)
