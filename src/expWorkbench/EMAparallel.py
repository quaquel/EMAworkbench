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
the _logger in EMAlogging to refer to the logger for its particular subprocess

'''
import Queue
import itertools
import shutil
import sys
import traceback
import multiprocessing 
import threading
import logging
import cStringIO
import copy
import os
import time

from multiprocessing import cpu_count
from multiprocessing.util import Finalize

import EMAlogging
from pool import RUN, ApplyResult, Pool
from EMAlogging import debug, exception, info
from expWorkbench.EMAexceptions import CaseError, EMAError, EMAParallelError

__all__ = ['CalculatorPool']


def worker(inqueue, 
           outqueue, 
           modelInterfaces, 
           modelInitKwargs=None):
    #
    # Code run by worker processes
    #    
        
    debug("worker started")
    
    put = outqueue.put
    get = inqueue.get
    if hasattr(inqueue, '_writer'):
        inqueue._writer.close()
        outqueue._reader.close()
    
    def cleanup(modelInterfaces):
        for msi in modelInterfaces:
            msi.cleanup()
            del msi
    

    oldPolicy = {}
    modelInitialized = False
    while 1:
        try:
            task = get()
        except (EOFError, IOError):
            debug('worker got EOFError or IOError -- exiting')
            break
        if task is None:
            debug('worker got sentinel -- exiting')
            cleanup(modelInterfaces)
            break

        job, i, case, policy = task
        for modelInterface in modelInterfaces:
            if policy != oldPolicy:
                modelInitialized = False
                try:
                    debug("invoking model init")
                    modelInterface.model_init(policy, modelInitKwargs)
                    debug("model initialized successfully")
                    modelInitialized = True
                except EMAError as e:
                    exception("init not implemented")
                    raise
                except Exception:
                    exception("some exception occurred when invoking the init")
            if modelInitialized:
                try:
                    try:
                        debug("trying to run model")
                        modelInterface.run_model(copy.deepcopy(case))
                    except CaseError as e:
                        EMAlogging.warning(e)
                    debug("trying to retrieve output")
                    result = modelInterface.retrieve_output()
                    
                    debug("trying to reset model")
                    modelInterface.reset_model()
                    result = (True, (case, policy, modelInterface.name, result))
                except Exception as e:
                    result = (False, e)
            else:
                result = (False, EMAParallelError("failure to initialize"))
            put((job, i, result))
            oldPolicy = policy

class CalculatorPool(Pool):
    '''
    Class representing a pool of processes that handles the parallel 
    calculation heavily derived from the standard pool provided with the 
    multiprocessing library
    '''

    def __init__(self, 
                 modelStructureInterfaces, 
                 processes=None, 
                 callback = None, 
                 kwargs=None):
        '''
        
        :param modelStructureInterface: modelInterface class
        :param processes: nr. of processes to spawn, if none, it is 
                                   set to equal the nr. of cores
        :param callback: callback function for handling the output 
        :param kwargs: kwargs to be pased to :meth:`model_init`
        '''
        
        self._setup_queues()
        self._taskqueue = Queue.Queue()
        self._cache = {}
        self._state = RUN

        self._callback = callback

        if processes is None:
            try:
                processes = cpu_count()
            except NotImplementedError:
                processes = 1
        info("nr of processes is "+str(processes))

        self.Process = LoggingProcess
        self.logQueue = multiprocessing.Queue()
        h = EMAlogging.NullHandler()
        logging.getLogger(EMAlogging.LOGGER_NAME).addHandler(h)
        
        # This thread will read from the subprocesses and write to the
        # main log's handlers.
        log_queue_reader = LogQueueReader(self.logQueue)
        log_queue_reader.start()

        self._pool = []

        workingDirectories = []
        debug('generating workers')
        
        
        workerRoot = None
        for i in range(processes):
            debug('generating worker '+str(i))
            
            workerName = 'PoolWorker'+str(i)
            
            def ignore_function(path, names):
                if path.find('.svn') != -1:
                    return names
                else:
                    return []
            
            #setup working directories for parallelEMA
            
            for msi in modelStructureInterfaces:
                if msi.workingDirectory != None:
                    if workerRoot == None:
                        workerRoot = os.path.dirname(os.path.abspath(modelStructureInterfaces[0].workingDirectory))
                    
                    workingDirectory = os.path.join(workerRoot, workerName, msi.name)
                    
                    workingDirectories.append(workingDirectory)
                    shutil.copytree(msi.workingDirectory, 
                                    workingDirectory,
                                    ignore = ignore_function)
                    msi.set_working_directory(workingDirectory)


            w = self.Process(
                self.logQueue,
                level = logging.getLogger(EMAlogging.LOGGER_NAME)\
                                          .getEffectiveLevel(),
                                          target=worker,
                                          args=(self._inqueue, 
                                                self._outqueue, 
                                                modelStructureInterfaces, 
                                                kwargs)
                                          )
            self._pool.append(w)
            
            w.name = w.name.replace('Process', workerName)
            w.daemon = True
            w.start()
            debug(' worker '+str(i) + ' generated')

        self._task_handler = threading.Thread(
                                        target=CalculatorPool._handle_tasks,
                                        args=(self._taskqueue, 
                                              self._quick_put, 
                                              self._outqueue, 
                                              self._pool, 
                                              self.logQueue)
                                        )
        self._task_handler.daemon = True
        self._task_handler._state = RUN
        self._task_handler.start()

        self._result_handler = threading.Thread(
            target=CalculatorPool._handle_results,
            args=(self._outqueue, self._quick_get, self._cache)
            )
        self._result_handler.daemon = True
        self._result_handler._state = RUN
        self._result_handler.start()

        self._terminate = Finalize(self, 
                                   self._terminate_pool,
                                   args=(self._taskqueue, 
                                         self._inqueue, 
                                         self._outqueue, 
                                         self._pool,
                                         self._task_handler, 
                                         self._result_handler, 
                                         self._cache, 
                                         workingDirectories,
                                         ),
                                    exitpriority=15
                                    )
        
        EMAlogging.info("pool has been set up")
     
    @staticmethod
    def _handle_tasks(taskqueue, put, outqueue, pool, logQueue):
        thread = threading.current_thread()

        for taskseq, set_length in iter(taskqueue.get, None):
            i = -1
            for i, task in enumerate(taskseq):
                if thread._state:
                    debug('task handler found thread._state != RUN')
                    break
                try:
                    put(task)
                except IOError:
                    debug('could not put task on queue')
                    break
            else:
                if set_length:
                    debug('doing set_length()')
                    set_length(i+1)
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
        time.sleep(2)
        
        logQueue.put(None)

    def runExperiments(self, cases, policies):
        """
        convenient function to wrap around the apply assync function
        """
        experiments = [(case, policy) for policy in policies for case in cases]
        results = [self.apply_async(experiment[0], 
                                experiment[1]) for experiment in experiments]
        return results

    def apply_async(self, case, policy):
        '''
        Asynchronous equivalent of `apply()` builtin
        '''
        assert self._state == RUN
        result = EMAApplyResult(self._cache, callback = self._callback)
        self._taskqueue.put(([(result._job, None, case, policy)], None))
        return result

    @classmethod
    def _terminate_pool(cls, 
                        taskqueue, 
                        inqueue, 
                        outqueue, 
                        pool,
                        task_handler, 
                        result_handler, 
                        cache, 
                        workingDirectories):

        EMAlogging.info("terminating pool")
        
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

        
        for directory in workingDirectories:
            directory = os.path.dirname(directory)
            EMAlogging.debug("deleting "+str(directory))
            shutil.rmtree(directory)

job_counter = itertools.count()

class EMAApplyResult(ApplyResult):
    '''
    small extension to :class:`multiprocessing.ApplyResult`
    
    '''

    def _set(self, i, obj):
        self._success, self._value = obj
        if self._callback and self._success:
            self._value = self._callback(self._value[0], self._value[1], self._value[2], 
                           self._value[3])
        self._cond.acquire()
        try:
            self._ready = True
            self._cond.notify()
        finally:
            self._cond.release()
        del self._cache[self._job]


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
        threading.Thread.__init__(self)
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
                    EMAlogging.debug("none received")
                    break
                
                logger = logging.getLogger(record.name)
                logger.callHandlers(record)
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
    to fit into the :mod:`EMAlogging` scheme.
    """
    
    def __init__(self, queue, level= None, target=None, args=()):
        multiprocessing.Process.__init__(self, target=target, args = args)
        self.queue = queue
        self.level = level

    def _setupLogger(self):
        # create the logger to use.
        logger = logging.getLogger(EMAlogging.LOGGER_NAME+'.subprocess')
        EMAlogging.LOGGER_NAME = EMAlogging.LOGGER_NAME+'.subprocess'
        EMAlogging._logger = logger
        
        # The only handler desired is the SubProcessLogHandler.  If any others
        # exist, remove them. In this case, on Unix and Linux the StreamHandler
        # will be inherited.

        for handler in logger.handlers:
            # just a check for my sanity
            assert not isinstance(handler, SubProcessLogHandler)
            logger.removeHandler(handler)
    
        # add the handler
        handler = SubProcessLogHandler(self.queue)
        handler.setFormatter(EMAlogging.formatter)
        logger.addHandler(handler)

        # On Windows, the level will not be inherited.  Also, we could just
        # set the level to log everything here and filter it in the main
        # process handlers.  For now, just set it from the global default.
        logger.setLevel(self.level)
        self.logger = logger

    def run(self):
        self._setupLogger()
        p = multiprocessing.current_process()
        debug('process %s with pid %s started' % (p.name, p.pid))
        #call the run of the super, which in turn will call the worker function
        super(LoggingProcess, self).run()


#==============================================================================
# test functions and classes
#==============================================================================

if __name__ == '__main__':
    from examples.pythonExample import SimplePythonModel
    from model import SimpleModelEnsemble
    
    EMAlogging.log_to_stderr(logging.INFO)
    
    modelInterface = SimplePythonModel(r'D:\jhkwakkel\workspace\EMA workbench\models\test', "testModel")
    
    ensemble = SimpleModelEnsemble()
    ensemble.setModelStructure(modelInterface)
    
    ensemble.parallel = True
    
    ensemble.performExperiments(10)
    