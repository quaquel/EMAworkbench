'''


'''
from __future__ import (unicode_literals, print_function, absolute_import,
                        division)

from collections import defaultdict
import io
import logging
import multiprocessing
import os
import sys
import threading
import time
import shutil
import traceback

from ..util import ema_logging
from .experiment_runner import ExperimentRunner
from .util import NamedObjectMap
from .model import AbstractModel

# Created on 22 Feb 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = []


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
        multiprocessing.util.Finalize(None, finalizer, args=(os.path.abspath(tmpdir), ), 
                                  exitpriority=10)  # @UndefinedVariable


def finalizer(tmpdir):
    '''cleanup'''
    global experiment_runner
    ema_logging.info("finalizing")
    
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
    logger = logging.getLogger(ema_logging.LOGGER_NAME+'.subprocess')
    ema_logging._logger = logger
    logger.handlers = []

    # add the handler
    handler = SubProcessLogHandler(queue)
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
        
        ema_logging.debug("setting up working directory: {}".format(tmpdir))
        
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
    return experiment_runner.run_experiment(experiment)


class SubProcessLogHandler(logging.Handler):
    """handler used by subprocesses

    It simply puts items on a Queue for the main process to log.

    adapted a bit using code found in same stack overflow thread 
    so that exceptions can be logged. Exception stack traces cannot be pickled
    so they cannot be put into the queue. Therefore they are formatted first 
    and then put into the queue as a normal message
    
    Found `online <http://stackoverflow.com/questions/641420/how-should-i-log-while-using-multiprocessing-in-python>`_

    """

    def __init__(self, queue):
        logging.Handler.__init__(self)
        self.queue = queue

    def emit(self, record):
        if record.exc_info:
            # can't pass exc_info across processes so just format now
            record.exc_text = self.formatException(record.exc_info)
            record.exc_info = None
        self.queue.put(record)

    def formatException(self, ei):
        sio = io.StringIO()
        traceback.print_exception(ei[0], ei[1], ei[2], None, sio)
        s = sio.getvalue()
        sio.close()
        if s[-1] == "\n":
            s = s[:-1]
        return s


class LogQueueReader(threading.Thread):
    """
    
    thread to write subprocesses log records to main process log

    This thread reads the records written by subprocesses and writes them to
    the handlers defined in the main process's handlers.
    
    found `online <http://stackoverflow.com/questions/641420/how-should-i-log-while-using-multiprocessing-in-python>`_
    
    
    TODO:: should be generalized with logwatcher used with ipyparallel

    """

    def __init__(self, queue):
        threading.Thread.__init__(self,name="log queue reader")
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
                    ema_logging.debug("none received")
                    break
                
                logger = logging.getLogger(record.name)
                logger.callHandlers(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except EOFError:
                break
            except TypeError:
                break
            except:
                traceback.print_exc(file=sys.stderr)


def result_handler(callback, experiment):
    '''handler for the results
    
    to link experiment and output, we use a functional programming
    trick.
    
    '''
    
    def my_actual_callback(result):
        callback(experiment, result)
    return my_actual_callback


def add_tasks(pool, experiments, callback):
    '''add experiments to pool
    
    '''
    
    results = []
    for e in experiments:
        
        # TODO:: code won't work on Python 3.4 or lower
        # error_callback only exists in 3.5 and up
        res = pool.apply_async(worker, [e], 
                                     callback=result_handler(callback, e))
        results.append(res)

    for res in results:
        res.wait()