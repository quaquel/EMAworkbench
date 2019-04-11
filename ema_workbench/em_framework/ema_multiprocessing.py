'''
support for using the multiprocessing library in combination with the workbench

'''
from collections import defaultdict

import logging
import multiprocessing
import os
import sys
import threading
import time
import shutil
import traceback
import queue

from ..util import get_module_logger, ema_logging
from .experiment_runner import ExperimentRunner
from .util import NamedObjectMap
from .model import AbstractModel


# Created on 22 Feb 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = []

_logger = get_module_logger(__name__)


def initializer(*args):
    '''initializer for a worker process

    Parameters
    ----------
    models : list of AbstractModel instances


    This function initializes the worker. This entails
    * initializing the experiment runner
    * setting up the working directory
    * setting up the logging
    '''
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
        multiprocessing.util.Finalize(None, finalizer,
                                      args=(os.path.abspath(tmpdir), ),
                                      exitpriority=10)


def finalizer(tmpdir):
    '''cleanup'''
    global experiment_runner
    _logger.info("finalizing")

    experiment_runner.cleanup()
    del experiment_runner

    time.sleep(1)

    if tmpdir:
        try:
            shutil.rmtree(tmpdir)
        except OSError:
            pass


def setup_logging(queue, log_level):
    '''helper function for enabling logging from the workers to the main
    process

    Parameters
    ----------
    queue : multiprocessing.Queue instance
    log_level : int

    '''

    # create a logger
    logger = logging.getLogger(ema_logging.LOGGER_NAME + '.subprocess')
    ema_logging._logger = logger
    logger.handlers = []

    # add the handler
    handler = logging.handlers.QueueHandler(queue)
    handler.setFormatter(ema_logging.LOG_FORMAT)
    logger.addHandler(handler)

    # set the log_level
    logger.setLevel(log_level)


def setup_working_directories(models, root_dir):
    '''copies the working directory of each model to a process specific
    temporary directory and update the working directory of the model

    Parameters
    ----------
    models : list
    root_dir : str

    '''

    # group models by working directory to avoid copying the same directory
    # multiple times
    wd_by_model = defaultdict(list)
    for model in models:
        try:
            wd = model.working_directory
        except AttributeError:
            pass
        else:
            wd_by_model[wd].append(model)

    # if the dict is not empty
    if wd_by_model:
        # make a directory with the process id as identifier
        tmpdir_name = "tmp{}".format(os.getpid())
        tmpdir = os.path.join(root_dir, tmpdir_name)
        os.mkdir(tmpdir)

        _logger.debug("setting up working directory: {}".format(tmpdir))

        for key, value in wd_by_model.items():
            # we need a sub directory in the process working directory
            # for each unique model working directory
            subdir = os.path.basename(os.path.normpath(key))
            new_wd = os.path.join(tmpdir, subdir)

            # the copy operation
            shutil.copytree(key, new_wd)

            for model in value:
                model.working_directory = new_wd
        return tmpdir
    else:
        return None


def worker(experiment):
    '''the worker function for executing an individual experiment

    Parameters
    ----------
    experiment : dict

    '''
    global experiment_runner
    return experiment, experiment_runner.run_experiment(experiment)


class LogQueueReader(threading.Thread):
    """

    thread to write subprocesses log records to main process log

    This thread reads the records written by subprocesses and writes them to
    the handlers defined in the main process's handlers.

    found `online <http://stackoverflow.com/questions/641420/how-should-i-log-while-using-multiprocessing-in-python>`_

    TODO:: should be generalized with logwatcher used with ipyparallel

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
                    _logger.debug("none received")
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
                    _logger.debug("none received")
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
    '''add experiments to pool

    Parameters
    ----------
    n_processes  : int
    pool : Pool instance
    experiments : collection
    callback : callable

    '''
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
