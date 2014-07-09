'''
Created on 21 dec. 2010

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

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
import itertools
import shutil
import sys
import traceback
import threading
import logging
import cStringIO
import copy
import os
import time
import Queue
import multiprocessing
import random
import string

from multiprocessing import Process, cpu_count, current_process,\
                            TimeoutError
from multiprocessing.util import Finalize

from pool import RUN, Pool, TERMINATE
from ema_logging import debug, exception, info, warning, NullHandler, LOG_FORMAT
import ema_logging                  
                                     
from expWorkbench.ema_exceptions import CaseError, EMAError, EMAParallelError
import tempfile

__all__ = ['CalculatorPool']

def worker(inqueue, 
           outqueue, 
           model_interfaces, 
           model_kwargs=None):
    #
    # Code run by worker processes
    #    
        
    debug("worker started")
    
    put = outqueue.put
    if hasattr(inqueue, '_writer'):
        inqueue._writer.close()
        outqueue._reader.close()
    
    def cleanup(model_interfaces):
        for msi in model_interfaces:
            msi.cleanup()
            del msi

    msis = {msi.name: msi for msi in model_interfaces}
    msi_initialization_dict = {}

    while 1:
        try:
            task = inqueue.get()
        except (EOFError, IOError):
            debug('worker got EOFError or IOError -- exiting')
            break
        if task is None:
            debug('worker got sentinel -- exiting')
            cleanup(model_interfaces)
            break

        job, experiment = task
        
        policy = experiment.pop('policy')
        debug("running policy {} for experiment {}".format(policy['name'], job))
        msi = experiment.pop('model')
        
        # check whether we already initialized the model for this 
        # policy
        if not msi_initialization_dict.has_key((policy['name'], msi)):
            try:
                debug("invoking model init")
                msis[msi].model_init(copy.deepcopy(policy), 
                                     copy.deepcopy(model_kwargs))
            except (EMAError, NotImplementedError) as inst:
                exception(inst)
                cleanup(model_interfaces)
                result = (False, inst)
                put((job, result))
            except Exception:
                exception("some exception occurred when invoking the init")
                cleanup(model_interfaces)
                result = (False, EMAParallelError("failure to initialize"))
                put((job, result))
                
            debug("initialized model %s with policy %s" % (msi, policy['name']))
            
            #always, only a single initialized msi instance
            msi_initialization_dict = {(policy['name'], msi):msis[msi]}
        msi = msis[msi]

        case = copy.deepcopy(experiment)
        try:
            debug("trying to run model")
            msi.run_model(case)
        except CaseError as e:
            warning(str(e))
            
        debug("trying to retrieve output")
        result = msi.retrieve_output()
        
        result = (True, (experiment, policy, msi.name, result))
        msi.reset_model()
        
        debug("trying to reset model")
        put((job, result))
            

class CalculatorPool(Pool):
    '''
    Class representing a pool of processes that handles the parallel 
    calculation heavily derived from the standard pool provided with the 
    multiprocessing library
    '''

    def __init__(self, 
                 msis, 
                 processes=None, 
                 kwargs=None):
        '''
        
        :param msis: an iterable of model structure interfaces
        :param processes: nr. of processes to spawn, if none, it is 
                                   set to equal the nr. of cores
        :param callback: callback function for handling the output 
        :param kwargs: kwargs to be pased to :meth:`model_init`
        '''
        
        self._setup_queues()
        self._taskqueue = Queue.Queue(cpu_count()*2)
        self._cache = {}
        self._state = RUN

        if processes is None:
            try:
                processes = cpu_count()
            except NotImplementedError:
                processes = 1
        info("nr of processes is "+str(processes))

        self.log_queue = multiprocessing.Queue()
        h = NullHandler()
        logging.getLogger(ema_logging.LOGGER_NAME).addHandler(h)
        
        # This thread will read from the subprocesses and write to the
        # main log's handlers.
        log_queue_reader = LogQueueReader(self.log_queue)
        log_queue_reader.start()

        self._pool = []
        working_dirs = []

        debug('generating workers')
        
        worker_root = None
        for i in range(processes):
            debug('generating worker '+str(i))
            
            # generate a random string helps in running repeatedly with
            # crashes
            choice_set = string.ascii_uppercase + string.digits + string.ascii_lowercase
            random_string = ''.join(random.choice(choice_set) for e in range(5))
            
            workername = 'tpm_{}_PoolWorker_{}'.format(random_string, i)
            
            #setup working directories for parallel_ema
            for msi in msis:
                if msi.working_directory != None:
                    if worker_root == None:
                        worker_root = os.path.dirname(os.path.abspath(msis[0].working_directory))
                    
                    working_directory = os.path.join(worker_root, workername)
                    
#                     working_directory = tempfile.mkdtemp(suffix=workername,
#                                                          prefix='tmp_',
#                                                          dir=worker_root)
                    
                    working_dirs.append(working_directory)
                    shutil.copytree(msi.working_directory, 
                                    working_directory, 
                                    )
                    msi.set_working_directory(working_directory)

            w = LoggingProcess(
                self.log_queue,
                level = logging.getLogger(ema_logging.LOGGER_NAME)\
                                          .getEffectiveLevel(),
                                          target=worker,
                                          args=(self._inqueue, 
                                                self._outqueue, 
                                                msis,
                                                kwargs 
                                                )
                                          )
            self._pool.append(w)
            
            w.name = w.name.replace('Process', workername)
            w.daemon = True
            w.start()
            debug(' worker '+str(i) + ' generated')

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
        self._task_handler._state = RUN
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
        self._result_handler._state = RUN
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
        
        info("pool has been set up")

    def run_experiments(self, experiments, callback):
        """
        
        starts a feeder thread that will add the experiments to the 
        task queue

        """

        event = threading.Event()

        self._feeder_thread = threading.Thread(target=CalculatorPool._add_tasks,
                                        name = 'taks feeder',
                                        args=(self, experiments,
                                              callback, 
                                              event)
                                        )
        self._feeder_thread.daemon = True
        self._feeder_thread._state = RUN
        self._feeder_thread.start()
        
        self._feeder_thread.join()
        
        event.wait()
     
    @staticmethod
    def _handle_tasks(taskqueue, put, outqueue, pool):
        thread = threading.current_thread()

        for task in iter(taskqueue.get, None):
            if thread._state:
                debug('task handler found thread._state != RUN')
                break
            try:
                put(task)
            except IOError:
                debug('could not put task on queue')
                break
            else:
                continue
            break
        else:
            debug('task handler got sentinel')

        try:
            # tell result handler to finish when cache is empty
            debug('task handler sending sentinel to result handler')
            outqueue.put(None)

            # tell workers there is no more work
            debug('task handler sending sentinel to workers')
            for i in range(2*len(pool)):
                put(None)
        except IOError:
            debug('task handler got IOError when sending sentinels')

        debug('task handler exiting')

    @staticmethod
    def _handle_results(outqueue, get, cache, log_queue):
        thread = threading.current_thread()

        while 1:
            try:
                task = get()
            except (IOError, EOFError):
                debug('result handler got EOFError/IOError -- exiting')
                return

            if thread._state:
                assert thread._state == TERMINATE
                debug('result handler found thread._state=TERMINATE')
                break

            if task is None:
                debug('result handler got sentinel')
                break

            job, experiment = task
            try:
                cache[job]._set(experiment)
            except KeyError:
                pass

        while cache and thread._state != TERMINATE:
            try:
                task = get()
            except (IOError, EOFError):
                debug('result handler got EOFError/IOError -- exiting')
                return

            if task is None:
                debug('result handler ignoring extra sentinel')
                continue
            job, obj = task
            try:
                cache[job]._set(obj)
            except KeyError:
                pass

        if hasattr(outqueue, '_reader'):
            debug('ensuring that outqueue is not full')
            # If we don't make room available in outqueue then
            # attempts to add the sentinel (None) to outqueue may
            # block.  There is guaranteed to be no more than 2 sentinels.
            try:
                for i in range(10):
                    if not outqueue._reader.poll():
                        break
                    get()
            except (IOError, EOFError):
                pass

        debug('result handler exiting: len(cache)=%s, thread._state=%s',
              len(cache), thread._state)
        
        log_queue.put(None)

    @staticmethod
    def _add_tasks(self, experiments, callback, event):
        for e in experiments:
            self.apply_async(e, callback, event)

    def apply_async(self, experiment, callback, event):
        '''
        Asynchronous equivalent of `apply()` builtin
        '''
        assert self._state == RUN
        result = EMAApplyResult(self._cache, callback, event)
        self._taskqueue.put((result._job, copy.deepcopy(experiment)))
        

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
        info("terminating pool")
        
        
        
        # this is guaranteed to only be called once
        debug('finalizing pool')
        TERMINATE = 2

        task_handler._state = TERMINATE
        for p in pool:
            taskqueue.put(None)                 # sentinel
            time.sleep(1)

        debug('helping task handler/workers to finish')
        cls._help_stuff_finish(inqueue, task_handler, len(pool))

        assert result_handler.is_alive() or len(cache) == 0

        result_handler._state = TERMINATE
        outqueue.put(None)                  # sentinel

        if pool and hasattr(pool[0], 'terminate'):
            debug('terminating workers')
            for p in pool:
                p.terminate()

        debug('joining task handler')
        task_handler.join(1e100)

        debug('joining result handler')
        result_handler.join(1e100)

        if pool and hasattr(pool[0], 'terminate'):
            debug('joining pool workers')
            for p in pool:
                p.join()
        
        # cleaning up directories
        # TODO investigate whether the multiprocessing.util tempdirectory  
        # functionality can be used instead
        
        for directory in working_dirs:
            debug("deleting "+str(directory))
            shutil.rmtree(directory)


job_counter = itertools.count()

class EMAApplyResult(object):
    '''
    a modified version of :class:`multiprocessing.ApplyResult`
    
    I could probably have used inheritance here, but for debugging purposes
    etc. I have included the full class specification here.
    
    '''

    def __init__(self, cache, callback, event):
        self._cond = threading.Condition(threading.Lock())
        self._job = job_counter.next()
        self._cache = cache
        self._ready = False
        self._callback = callback
        self._event = event
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
            raise TimeoutError
        if self._success:
            return self._value
        else:
            raise self._value

    def _set(self, obj):
        self._success, self._value = obj
        if self._callback and self._success:
            self._callback(*self._value)
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
    """
    
    handler used by subprocesses

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
        sio = cStringIO.StringIO()
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
                    debug("none received")
                    break
                
                logger = logging.getLogger(record.name)
                logger.callHandlers(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except EOFError:
                break
            except:
                traceback.print_exc(file=sys.stderr)


class LoggingProcess(Process):
    """
    A small extension of the default :class:`multiprocessing.Process` 
    in order to log from the subprocesses.
    
    Adapted from code found `online <http://stackoverflow.com/questions/641420/how-should-i-log-while-using-multiprocessing-in-python>`_
    to fit into the :mod:`ema_logging` scheme.
    """
    
    def __init__(self, queue, level= None, target=None, args=()):
        Process.__init__(self, target=target, args = args)
        self.queue = queue
        self.level = level

    def _setupLogger(self):
        # create the logger to use.
        logger = logging.getLogger(ema_logging.LOGGER_NAME+'.subprocess')
        ema_logging.LOGGER_NAME+'.subprocess'
        ema_logging._logger = logger
        _logger = logger
        
        # The only handler desired is the SubProcessLogHandler.  If any others
        # exist, remove them. In this case, on Unix and Linux the StreamHandler
        # will be inherited.

        for handler in logger.handlers:
            # just a check for my sanity
            assert not isinstance(handler, SubProcessLogHandler)
            logger.removeHandler(handler)
    
        # add the handler
        handler = SubProcessLogHandler(self.queue)
        handler.setFormatter(LOG_FORMAT)
        logger.addHandler(handler)

        # On Windows, the level will not be inherited.  Also, we could just
        # set the level to log everything here and filter it in the main
        # process handlers.  For now, just set it from the global default.
        logger.setLevel(self.level)
        self.logger = logger

    def run(self):
        self._setupLogger()
        p = current_process()
        debug('process %s with pid %s started' % (p.name, p.pid))
        #call the run of the super, which in turn will call the worker function
        super(LoggingProcess, self).run()
