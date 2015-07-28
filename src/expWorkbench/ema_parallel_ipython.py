'''
This module provides functionality for combining the EMA workbench
with IPython parallel

Created on Jul 16, 2015

.. codeauthor::  jhkwakkel
'''
import collections
import logging
import os
import shutil
import socket
import threading
import zmq

import IPython
from IPython.config import Application

from expWorkbench import EMAError, EMAParallelError
import ema_logging
from ema_logging import debug, info
import model_ensemble

SUBTOPIC = "EMA"
engine = None

class EngingeLoggerAdapter(logging.LoggerAdapter):
    '''LoggerAdapter that inserts EMA as a topic into log messages
    '''

    def __init__(self, logger, topic):
        self.logger = logger
        self.topic = topic
        
    def process(self, msg, kwargs):
        
        msg = '{topic}::{msg}'.format(topic=self.topic, msg=msg)
        
        return msg, kwargs


class LogWatcher(object):
    """A  class that receives messages on a SUB socket, as published
    by subclasses of `zmq.log.handlers.PUBHandler`, and logs them itself.
    
    This LogWatcher subscribes to all topics and aggregates them by logging
    to the EMA logger. 
    
    It is possible to filter topics before they are being logged on the EMA
    logger. This filtering is done on a loglevel and topic basis. By default,
    filtering is active on the DEBUG level, with EMA as topic.   
    
    This class is adapted from the LogWatcher in IPython.paralle.apps to 
    fit the needs of the workbench. 
    """

    LOG_FORMAT = '[%(levelname)s] %(message)s'
    
    topic_subscriptions = {logging.DEBUG : set([SUBTOPIC])}
    
    
    def __init__(self, url):
        '''
        
        Parameters
        ----------
        url : string
              the url on which to listen for log messages
        
        '''
        
        super(LogWatcher, self).__init__()
        self.context = zmq.Context()
        self.loop = zmq.eventloop.ioloop.IOLoop() # @UndefinedVariable
        self.url = url
        
        s = self.context.socket(zmq.SUB) # @UndefinedVariable
        s.bind(self.url)
        
        # setup up the aggregate EMA logger
        self.logger = ema_logging.get_logger()

        # add check to avoid double stream handlers
        if not any([isinstance(h, logging.StreamHandler) for h in 
                    self.logger.handlers]):
            handler = logging.StreamHandler()
            formatter = logging.Formatter(self.LOG_FORMAT)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.stream = zmq.eventloop.zmqstream.ZMQStream(s, self.loop) # @UndefinedVariable
        self.subscribe()
        
    def start(self):
        '''start the log watcher'''
        
        info('start watching on {}'.format(self.url))
        self.stream.on_recv(self.log_message)
    
    def stop(self):
        '''stop the log watcher'''
        self.stream.stop_on_recv()
 
    def subscribe(self):
        """Update our SUB socket's subscriptions."""
        debug("Subscribing to: everything")
        self.stream.setsockopt(zmq.SUBSCRIBE, '') # @UndefinedVariable
        
    def _extract_level(self, topic_str):
        """Turn 'engine.0.INFO.extra' into (logging.INFO, 'engine.0.extra')"""

        topics = topic_str.split('.')
        for idx,t in enumerate(topics):
            level = getattr(logging, t, None)

            if level is not None:
                break
        
        if level is None:
            level = logging.INFO
        else:
            topics.pop(idx)
        
        return level, topics

    def subscribe_to_topic(self, level, topic):
        '''add a topic subscription for the specified level
        
        Parameters
        ----------
        level : int 
                the logging level to which the topic subscription applies
        topic : string
                topic name
        
        '''
        
        try:
            self.topic_subscriptions[level].update([topic])
        except KeyError:
            self.topic_subscriptions[level] = set([topic])
        
    def log_message(self, raw):
        """receive and parse a message, then log it."""

        
        if len(raw) != 2 or '.' not in raw[0]:
            self.logger.error("Invalid log message: %s"%raw)
            return
        else:
            raw = [entry.strip() for entry in raw]
            
            topic, msg = raw
            topic = topic.strip()
            level, topics = self._extract_level(topic)
            
            topic = '.'.join(topics)
            subtopic = '.'.join(topics[2::])
            
            try:
                subscriptions = self.topic_subscriptions[level]
            except KeyError:
                self.logger.log(level, "[%s] %s" % (topic, msg))
            else:
                if subtopic in subscriptions:
                    self.logger.log(level, "[%s] %s" % (topic, msg))
        

def start_logwatcher(url):
    '''convenience function for starting the LogWatcher 
    
    Parameters
    ----------
    url : string
          the url on which to listen for log messages

    Returns
    -------
    LogWatcher
        the log watcher instance
    Thread
        the log watcher thread
        
    .. note : there can only be one log watcher on a given url. 
    
    '''

    logwatcher = LogWatcher(url)
    
    def starter():
        logwatcher.start()
        try:
            logwatcher.loop.start()
        except KeyboardInterrupt:
            print "Logging Interrupted, shutting down...\n"
    
    logwatcher_thread = threading.Thread(target=starter)
    logwatcher_thread.deamon = True
    logwatcher_thread.start()
    
    return logwatcher, logwatcher_thread


def set_engine_logger():
    '''Updates EMA logging on the engines with an EngineLoggerAdapter 
    This adapter injects EMA as a topic into all messages
    '''
    
    logger = Application.instance().log
    logger.setLevel(ema_logging.DEBUG)
    for handler in logger.handlers:
        if isinstance(handler, IPython.kernel.zmq.log.EnginePUBHandler): # @UndefinedVariable
            handler.setLevel(logging.DEBUG)
    
    adapter = EngingeLoggerAdapter(logger, SUBTOPIC)
    ema_logging._logger = adapter
    
    debug('updated logger')
    

def get_engines_by_host(client):
    ''' returns the engine ids by host
    
    Parameters
    ----------
    client : IPython.parallel.Client instance
    
    Returns
    -------
    dict
        a dict with hostnames as keys, and a list
        of engine ids
    
    '''
    
    def engine_hostname():
        return socket.gethostname()

    results = {i:client[i].apply_sync(engine_hostname) for i in client.ids}

    engines_by_host = collections.defaultdict(list)
    for engine_id, host in results.items():
        engines_by_host[host].append(engine_id)
    return engines_by_host


def update_cwd_on_all_engines(client):
    ''' updates the current working directory on 
    the engines to point to the same working directory
    as this notebook
    
    currently only works if engines are on same 
    machine.
    
    Parameters
    ----------
    client : IPython.parallel.Client instance
    
    '''

    engines_by_host = get_engines_by_host(client)
    
    notebook_host = socket.gethostname()
    for key, value in engines_by_host.items():

        def set_cwd_on_engine(cwd):
            os.chdir(cwd)

        if key == notebook_host:
            cwd = os.getcwd()

            # easy, we now the correct cwd
            for engine in value:
                client[engine].apply(set_cwd_on_engine, cwd)
        else:
            raise NotImplementedError('not yet supported')


class Engine(object):
    '''class for setting up ema specific stuff on each engine
    also functions as a convenient namespace for workbench
    relevant variables
    
    '''


    def __init__(self, engine_id, msis, model_init_kwargs={}):
        self.engine_id = engine_id
        self.msis = msis
        self.msi_initialization = {}
        self.runner = model_ensemble.ExperimentRunner(msis, model_init_kwargs)

    def setup_working_directory(self, dir_name):
        '''setup the root directory for the engine. The working directories 
        associated with the various model structure interfaces will be copied to 
        this root directory

        Parameters
        ----------
        dir_name : string
                   The name of the root directory


        '''
        dir_name = dir_name.format(self.engine_id)
        
        # if the directory already exists, is has not bee
        # cleaned up properly last time
        # let's be stupid and remove it
        # a smarter solution would be to do a dif on the existing
        # directory and what we would like to copy. 
        if os.path.isdir(dir_name):
            shutil.rmtree(dir_name)
        
        os.mkdir(dir_name)
        self.root_dir = dir_name

    def cleanup_working_directory(self):
        '''remove the root working directory of the engine'''
        shutil.rmtree(self.root_dir) 

    def copy_wds_for_msis(self, dirs_to_copy, wd_by_msi):
        '''copy each unique working directory to the engine specific
        folder and update the working directory for the associated model 
        structure interfaces

        Parameters
        ----------
        dirs_to_copy : list of strings
        wd_by_msi : dict
                    a mapping from working directory to associated models 
                    structure interfaces. 


        '''
        for src in dirs_to_copy:
            basename = os.path.basename(src)
            dst = os.path.join(self.root_dir, basename)
            shutil.copytree(src, dst) 

            # set working directory for the associated msi's 
            # on the engine
            for msi_name in wd_by_msi[src]:
                msi = self.msis[msi_name] 
                msi.working_directory = dst
                
    def run_experiments(self, experiment):
        '''run the experiment, the actual running is delegated
        to an ExperimentRunner instance'''
        
        try:
            return self.runner.run_experiment(experiment) 
        except EMAError:
            raise
        except Exception:
            raise EMAParallelError(str(Exception))
       

def initialize_engines(client, msis, model_init_kwargs={}):
    '''initialize engine instances on all engines
    
    Parameters
    ----------
    client : IPython.parallel.Client 
    msis : dict
           dict of model structure interfaces with their names as keys
    model_init_kwargs : dict, optional
                        kwargs to pass to msi.model_init
    
    
    '''
    for i in client.ids:
        client[i].apply_sync(_initialize_engine, i, msis, model_init_kwargs)
        
    setup_working_directories(client, msis)

      
def setup_working_directories(client, msis):
    '''setup working directory structure on all engines and copy files as 
    necessary
        
    Parameters
    ----------
    client : IPython.parallel.Client 
    msis : dict
           dict of model structure interfaces with their names as keys
    
    
    .. note:: multiple hosts not yet supported!
    
    '''
    
    # get the relevant working directories to copy
    # it might be that working directories are shared
    # so we use a defaultdict to store the model interfaces
    # associated with each working directory
    wd_by_msi = collections.defaultdict(list)
    for msi in msis.values():
        wd_by_msi[msi.working_directory].append(msi.name)
        
    # determine the common root of all working directories
    common_root = os.path.commonprefix(wd_by_msi.keys())
    common_root = os.path.dirname(common_root)
    
    engine_wd_name = 'engine{}'
    engine_dir = os.path.join(common_root, engine_wd_name)
        
    # create the root directory for each engine
    # we need to block until directory has been created
    client[:].apply_sync(_setup_working_directory, engine_dir)
    
    # copy the working directories and update msi.working_directory
    dirs_to_copy = wd_by_msi.keys()
    client[:].apply_sync(_copy_working_directories, dirs_to_copy, wd_by_msi)
    
    # TODO:dirs_to_copy should be relative to some kind of home directory
    # this home directory might be host specific
    # using common_prefix we can get the path relative to home
    # and send this to the engines, which again add the their home to it


def cleanup_working_directories(client):
    '''cleanup directory tree structure on all engines '''
    client[:].apply_sync(_cleanun_working_directory)


# engines can only deal with functions, not with object.method calls
# these functions are wrappers around the relevant Engine methods
# the engine instance is part of the namespace of the module. 
def _run_experiment(experiment):
    return engine.run_experiments(experiment)


def _initialize_engine(engine_id, msis, model_init_kwargs):
    global engine
    engine = Engine(engine_id, msis, model_init_kwargs)

    
def _setup_working_directory(dir_name):
    engine.setup_working_directory(dir_name)

    
def _cleanun_working_directory():
    engine.cleanup_working_directory()

    
def _copy_working_directories(dirs_to_copy, wd_by_msi):
    engine.copy_wds_for_msis(dirs_to_copy, wd_by_msi)