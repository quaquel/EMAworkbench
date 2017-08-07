'''
calculator pool 
adapted from  the pool provided in multiprocessing. This version is based 
on the python 2.6 version.

multiprocessing logging ideas based on code examples found on StackTrace:
http://stackoverflow.com/questions/641420/how-should-i-log-while-using-multiprocessing-in-python

the implementation here allows the user to not be bothered with many of the 
details of multiprocessing. For example, logging will work the same regardless 
of whether multiprocessing is used. The process class here modifies 
the _logger in ema_logging to refer to the logger for its particular subprocess

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import io
import itertools
import logging
import multiprocessing
import os
import six
from ema_workbench.em_framework.model import FileModel

try:
    import queue
except ImportError:
    import Queue as queue
import random
import shutil
import string
import sys
import threading
import time
import traceback


import multiprocessing.pool as pool
from multiprocessing.util import Finalize
try:
    from multiprocessing import get_context
except ImportError:
    def get_context():
        return None

from .experiment_runner import ExperimentRunner
from ..util import ema_logging, EMAError, EMAParallelError

# Created on 21 dec. 2010
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['CalculatorPool']


def worker(inqueue, 
           outqueue, 
           model_interfaces):
    #
    # Code run by worker processes
    #    
        
    ema_logging.debug("worker started")
    
    if hasattr(inqueue, '_writer'):
        inqueue._writer.close()
        outqueue._reader.close()

    runner = ExperimentRunner(model_interfaces)

    while 1:
        try:
            task = inqueue.get()
        except (EOFError, IOError):
            ema_logging.debug('worker got EOFError or IOError -- exiting')
            break
        if task is None:
            ema_logging.debug('worker got sentinel -- exiting')
            runner.cleanup()
            break

        _, experiment = task
        experiment_id = experiment.experiment_id

        try:
            result = runner.run_experiment(experiment)
        except EMAError as inst:
            result = inst
            success = False
        except Exception as inst:
            result = EMAParallelError("failure to initialize: "+str(inst))
            success = False
        else:
            success = True
        
        ema_logging.debug('putting result on outqueue')
        outqueue.put((experiment_id, (success, result)))

class CalculatorPool(pool.Pool):
    '''
    Class representing a pool of processes that handles the parallel 
    calculation heavily derived from the standard pool provided with the 
    multiprocessing library
    '''

    def __init__(self, 
                 msis, 
                 processes=None):
        '''
        
        Parameters
        ----------
        msis : list 
               iterable of model structure interface instances
        processes: int
                   nr. of processes to spawn, if none, it is set to equal the 
                   nr. of cores
        kwargs : dict
                 kwargs to be pased to :meth:`model_init`
        '''
        
        if processes is None:
            try:
                processes = multiprocessing.cpu_count()
            except NotImplementedError:
                processes = 1
        ema_logging.info("nr of processes is "+str(processes))
    
        # setup queues etc.
        self._ctx = get_context()
        self._setup_queues()
        self._taskqueue = queue.Queue(processes*2)
        self._cache = {}
        self._state = pool.RUN
        
        # handling of logging
        self.log_queue = multiprocessing.Queue()
        h = ema_logging.NullHandler()
        logging.getLogger(ema_logging.LOGGER_NAME).addHandler(h)
        
        log_queue_reader = LogQueueReader(self.log_queue)
        log_queue_reader.start()

        # setup of the actual pool
        self._pool = []
        working_dirs = []

        ema_logging.debug('generating workers')
        
        worker_root = None
        for i in range(processes):
            # consider adding a progress bar if we need to setup 
            # many processes including substantial copying          
            
            ema_logging.debug('generating worker '+str(i))
            
            workername = self._get_worker_name(i)
            
            #setup working directories for parallel_ema
            for msi in msis:
                if isinstance(msi, FileModel):
                    if worker_root == None:
                        wd = msis[0].working_directory
                        abs_wd = os.path.abspath(wd)
                        worker_root = os.path.dirname(abs_wd)
                    
                    wd_name = workername + msi.name
                    working_directory = os.path.join(worker_root, wd_name)
                    
                    working_dirs.append(working_directory)
                    shutil.copytree(msi.working_directory, 
                                    working_directory, 
                                    )
                    msi.working_directory = working_directory

#             w = multiprocessing.Process(target=worker,
#                                         args=(self._inqueue, 
#                                               self._outqueue, 
#                                               msis,
#                                               kwargs)
#                                         )
            w = LoggingProcess(self.log_queue,
                                level = logging.getLogger(ema_logging.LOGGER_NAME).getEffectiveLevel(),
                                target=worker,
                                args=(self._inqueue, 
                                      self._outqueue, 
                                      msis)
                                )
            self._pool.append(w)
            
            w.name = w.name.replace('Process', workername)
            w.daemon = True
            w.start()
            ema_logging.debug(' worker '+str(i) + ' generated')

        # thread for handling tasks
        self._task_handler = threading.Thread(
                                        target=CalculatorPool._handle_tasks,
                                        name='task handler',
                                        args=(self._taskqueue, 
                                              self._quick_put, 
                                              self._outqueue, 
                                              self._pool
                                              )
                                        )
        self._task_handler.daemon = True
        self._task_handler._state = pool.RUN
        self._task_handler.start()

        # thread for handling results
        self._result_handler = threading.Thread(
                                        target=CalculatorPool._handle_results,
                                        name='result handler',
                                        args=(self._outqueue, 
                                              self._quick_get, 
                                              self._cache, 
                                              self.log_queue)
            )
        self._result_handler.daemon = True
        self._result_handler._state = pool.RUN
        self._result_handler.start()

        # function for cleaning up when finalizing object
        self._terminate = Finalize(self, 
                                   self._terminate_pool,
                                   args=(self._taskqueue, 
                                         self._inqueue, 
                                         self._outqueue, 
                                         self._pool,
                                         self._task_handler, 
                                         self._result_handler, 
                                         self._cache, 
                                         working_dirs,
                                         ),
                                    exitpriority=15
                                    )
        
        ema_logging.info("pool has been set up")


    def _get_worker_name(self, i):
        '''Generate a name with random characters for the worker
        
        Parameters
        ----------
        i : int
            the index of the worker
        
        '''
        
        # generate a random string helps in running repeatedly with
        # crashes
        choice_set = (string.ascii_uppercase + string.digits + 
                     string.ascii_lowercase)
        random_string = ''.join(random.choice(choice_set) for _ in range(5))
        
        workername = 'tpm_{}_PoolWorker_{}'.format(random_string, i)
        return workername

    def run_experiments(self, experiments, callback):
        """starts a feeder thread that will add the experiments to the 
        task queue
        
        Parameters
        ----------
        experiments : iterable
                      iterable of dicts
        callback : a Callback instance
                   callback function for handling the output 
        """
        global job_counter 
        job_counter = itertools.count()

        event = threading.Event()

        self._feeder_thread = threading.Thread(target=CalculatorPool._add_tasks,
                                        name = 'taks feeder',
                                        args=(self, experiments,
                                              callback, 
                                              event)
                                        )
        self._feeder_thread.daemon = True
        self._feeder_thread._state = pool.RUN
        self._feeder_thread.start()
        
        self._feeder_thread.join()
        
        event.wait()

    @staticmethod
    def _handle_tasks(taskqueue, put, outqueue, pool):
        thread = threading.current_thread()

        for task in iter(taskqueue.get, None):
            if thread._state:
                ema_logging.debug('task handler found thread._state != RUN')
                break
            try:
                put(task)
            except IOError:
                ema_logging.debug('could not put task on queue')
                break
            else:
                continue
            break
        else:
            ema_logging.debug('task handler got sentinel')

        try:
            # tell result handler to finish when cache is empty
            ema_logging.debug('task handler sending sentinel to result handler')
            outqueue.put(None)

            # tell workers there is no more work
            ema_logging.debug('task handler sending sentinel to workers')
            for _ in range(2*len(pool)):
                put(None)
        except IOError:
            ema_logging.debug('task handler got IOError when sending sentinels')

        ema_logging.debug('task handler exiting')

    @staticmethod
    def _handle_results(outqueue, get, cache, log_queue):
        thread = threading.current_thread()

        while 1:
            try:
                task = get()
            except (IOError, EOFError):
                ema_logging.debug('result handler got EOFError/IOError -- exiting')
                return

            if thread._state:
                assert thread._state == pool.TERMINATE
                ema_logging.debug('result handler found thread._state=TERMINATE')
                break

            if task is None:
                ema_logging.debug('result handler got sentinel')
                break

            job, experiment = task
            try:
                cache[job]._set(experiment)
            except KeyError:
                pass

        while cache and thread._state != pool.TERMINATE:
            try:
                task = get()
            except (IOError, EOFError):
                ema_logging.debug('result handler got EOFError/IOError -- exiting')
                return

            if task is None:
                ema_logging.debug('result handler ignoring extra sentinel')
                continue
            job, obj = task
            try:
                cache[job]._set(obj)
            except KeyError:
                pass

        if hasattr(outqueue, '_reader'):
            ema_logging.debug('ensuring that outqueue is not full')
            # If we don't make room available in outqueue then
            # attempts to add the sentinel (None) to outqueue may
            # block.  There is guaranteed to be no more than 2 sentinels.
            try:
                for _ in range(10):
                    if not outqueue._reader.poll():
                        break
                    get()
            except (IOError, EOFError):
                pass

        ema_logging.debug('result handler exiting: len(cache)=%s, thread._state=%s',
              len(cache), thread._state)
        
        log_queue.put(None)

    @staticmethod
    def _add_tasks(self, experiments, callback, event):
        for e in experiments:
            self.apply_async(e, callback, event)

    def apply_async(self, experiment, callback, event):
        '''
        Asynchronous equivalent of `apply()` builtin

        Parameters
        ----------
        experiment : dict
        callback : a Callback instance
                   callback function for handling the output
        event : threading.Event instance

        '''
        assert self._state == pool.RUN
        result = EMAApplyResult(self._cache, callback, event, experiment)
        self._taskqueue.put((result._job, experiment))

    @classmethod
    def _terminate_pool(cls, 
                        taskqueue, 
                        inqueue, 
                        outqueue, 
                        pool,
                        task_handler, 
                        result_handler, 
                        cache, 
                        working_dirs,
                        ):
        ema_logging.info("terminating pool")
        
        
        
        # this is guaranteed to only be called once
        ema_logging.debug('finalizing pool')
        TERMINATE = 2

        task_handler._state = TERMINATE
        for p in pool:
            taskqueue.put(None)                 # sentinel
            time.sleep(1)

        ema_logging.debug('helping task handler/workers to finish')
        cls._help_stuff_finish(inqueue, task_handler, len(pool))

        assert result_handler.is_alive() or len(cache) == 0

        result_handler._state = TERMINATE
        outqueue.put(None)                  # sentinel

        if pool and hasattr(pool[0], 'terminate'):
            ema_logging.debug('terminating workers')
            for p in pool:
                p.terminate()

        ema_logging.debug('joining task handler')
        task_handler.join(1e100)

        ema_logging.debug('joining result handler')
        result_handler.join(1e100)

        if pool and hasattr(pool[0], 'terminate'):
            ema_logging.debug('joining pool workers')
            for p in pool:
                p.join()
        
        # cleaning up directories
        # TODO investigate whether the multiprocessing.util tempdirectory  
        # functionality can be used instead
        
        for directory in working_dirs:
            ema_logging.debug("deleting "+str(directory))
            shutil.rmtree(directory)



class EMAApplyResult(object):
    '''
    a modified version of :class:`multiprocessing.ApplyResult`
    
    I could probably have used inheritance here, but for debugging purposes
    etc. I have included the full class specification here.
    
    '''

    def __init__(self, cache, callback, event, experiment):
        self._cond = threading.Condition(threading.Lock())
        self._job = six.next(job_counter)
        self._cache = cache
        self._ready = False
        self._callback = callback
        self._event = event
        self.experiment = experiment
        cache[self._job] = self

    def ready(self):
        return self._ready

    def successful(self):
        assert self._ready
        return self._success

    def wait(self, timeout=None):
        self._cond.acquire()
        try:
            if not self._ready:
                self._cond.wait(timeout)
        finally:
            self._cond.release()

    def get(self, timeout=None):
        self.wait(timeout)
        if not self._ready:
            raise multiprocessing.TimeoutError
        if self._success:
            return self._result
        else:
            raise self._result

    def _set(self, obj):
        self._success, self._result = obj
        
        if self._callback and self._success:
            self._callback(self.experiment, self._result)
        else:
            ema_logging.warning(self._result)
            
        self._cond.acquire()
        try:
            self._ready = True
            self._cond.notify()
        finally:
            self._cond.release()
        del self._cache[self._job]
        
        if not self._cache:
            self._event.set()


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
#                 ema_logging.info("{}, {}, {}".format(id(record), record.msg, record.name))
            except (KeyboardInterrupt, SystemExit):
                raise
            except EOFError:
                break
            except:
                traceback.print_exc(file=sys.stderr)


class LoggingProcess(multiprocessing.Process):
    """
    A small extension of the default :class:`multiprocessing.Process` 
    in order to log from the subprocesses.
    
    Adapted from code found `online <http://stackoverflow.com/questions/641420/how-should-i-log-while-using-multiprocessing-in-python>`_
    to fit into the :mod:`ema_logging` scheme.
    """
    
    _i = 0
    
    def __init__(self, queue, level=None, target=None, args=()):
        super(LoggingProcess, self).__init__(target=target, args=args)
        self.queue = queue
        self.level = level

    def _setupLogger(self):
        # create the logger to use.
        logger = logging.getLogger(ema_logging.LOGGER_NAME+'.subprocess')
        ema_logging._logger = logger
        
        # The only handler desired is the SubProcessLogHandler.  If any others
        # exist, remove them. In this case, on Unix and Linux the StreamHandler
        # will be inherited.

        logger.handlers = []
    
        # add the handler
        handler = SubProcessLogHandler(self.queue)
        handler.setFormatter(ema_logging.LOG_FORMAT)
        logger.addHandler(handler)

        # On Windows, the level will not be inherited.  Also, we could just
        # set the level to log everything here and filter it in the main
        # process handlers.  For now, just set it from the global default.
        logger.setLevel(self.level)
        self.logger = logger

    def run(self):
        self._setupLogger()
        ema_logging.debug('process %s with pid %s started' % (self.name, self.pid))
        #call the run of the super, which in turn will call the worker function
        super(LoggingProcess, self).run()
