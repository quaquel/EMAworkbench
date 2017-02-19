'''
This module provides functionality for combining the EMA workbench
with IPython parallel. 

.. note:: the version provided here is compatible with ipython 4, and 
          ipyparallel. That is, the version after the big split. It will not 
          work with older versions of IPython


'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import collections
import logging
import os
import shutil
import socket
import threading

import zmq
from zmq.eventloop import ioloop, zmqstream

from traitlets.config import Application
from traitlets.config.configurable import LoggingConfigurable
from traitlets import Unicode, Instance, List

from jupyter_client.localinterfaces import localhost


from ipyparallel.engine.log import EnginePUBHandler

from . import experiment_runner
from ..util import ema_exceptions, utilities, ema_logging

# Created on Jul 16, 2015
# 
# .. codeauthor::  jhkwakkel

SUBTOPIC = ema_logging.LOGGER_NAME
engine = None
EMA_PROJECT_HOME_DIR = utilities.get_ema_project_home_dir()

class EngingeLoggerAdapter(logging.LoggerAdapter):
    '''LoggerAdapter that inserts EMA as a topic into log messages
    
    Parameters
    ----------
    logger : logger instance 
    topic : str
    
    '''

    def __init__(self, logger, topic):
        self.logger = logger
        self.topic = topic
        
    def process(self, msg, kwargs):
        msg = '{topic}::{msg}'.format(topic=self.topic, msg=msg)
        return msg, kwargs

def start_logwatcher():
    '''convenience function for starting the LogWatcher 
    
    Returns
    -------
    LogWatcher
        the log watcher instance
    Thread
        the log watcher thread
        
    .. note : there can only be one log watcher on a given url. 
    
    '''

    logwatcher = LogWatcher()
    
    def starter():
        logwatcher.start()
        try:
            logwatcher.loop.start()
        except (zmq.error.ZMQError, IOError, OSError):
            ema_logging.warning('shutting down log watcher')
        except RuntimeError:
            pass
    
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
        if isinstance(handler, EnginePUBHandler): # @UndefinedVariable
            handler.setLevel(ema_logging.DEBUG)
    
    adapter = EngingeLoggerAdapter(logger, SUBTOPIC)
    ema_logging._logger = adapter
    
    ema_logging.debug('updated logger')
    

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
    
    results = {i:client[i].apply_sync(socket.gethostname) for i in client.ids}

    engines_by_host = collections.defaultdict(list)
    for engine_id, host in results.items():
        engines_by_host[host].append(engine_id)
    return engines_by_host


def update_cwd_on_all_engines(client):
    ''' updates the current working directory on the engines to point to the 
    same working directory as this notebook
    
    currently only works if engines are on same 
    machine.
    
    Parameters
    ----------
    client : IPython.parallel.Client instance
    
    '''

    engines_by_host = get_engines_by_host(client)
    
    notebook_host = socket.gethostname()
    for key, value in engines_by_host.items():

        if key == notebook_host:
            cwd = os.getcwd()

            # easy, we know the correct cwd
            for engine in value:
                client[engine].apply(os.chdir, cwd)
        else:
            raise NotImplementedError('not yet supported')


class LogWatcher(LoggingConfigurable):
    """A simple class that receives messages on a SUB socket, as published
    by subclasses of `zmq.log.handlers.PUBHandler`, and logs them itself.
    
    This can subscribe to multiple topics, but defaults to all topics.
    """
    
    # configurables
    topics = List([''],
        help="The ZMQ topics to subscribe to. Default is to subscribe to all messages").tag(config=True)
    url = Unicode(
        help="ZMQ url on which to listen for log messages").tag(config=True)
    def _url_default(self):
        return 'tcp://%s:20202' % localhost()
    
    # internals
    stream = Instance('zmq.eventloop.zmqstream.ZMQStream', allow_none=True)
    
    context = Instance(zmq.Context)
    def _context_default(self):
        return zmq.Context.instance()
    
    loop = Instance(zmq.eventloop.ioloop.IOLoop)  # @UndefinedVariable
    def _loop_default(self):
        return ioloop.IOLoop.instance()
    
    def __init__(self, **kwargs):
        super(LogWatcher, self).__init__(**kwargs)
        s = self.context.socket(zmq.SUB)  # @UndefinedVariable
        s.bind(self.url)
        self.stream = zmqstream.ZMQStream(s, self.loop)
        self.subscribe()
        self.observe(self.subscribe, 'topics')
    
    def start(self):
        self.stream.on_recv(self.log_message)
    
    def stop(self):
        self.stream.setsockopt(zmq.UNSUBSCRIBE, b'')  # @UndefinedVariable
        self.stream.stop_on_recv()
    
    def subscribe(self):
        """Update our SUB socket's subscriptions."""
        self.stream.setsockopt(zmq.UNSUBSCRIBE, b'')  # @UndefinedVariable
        if '' in self.topics:
            self.log.debug("Subscribing to: everything")
            self.stream.setsockopt(zmq.SUBSCRIBE, b'')  # @UndefinedVariable
        else:
            for topic in self.topics:
                self.log.debug("Subscribing to: %r"%(topic))
                self.stream.setsockopt(zmq.SUBSCRIBE, topic)  # @UndefinedVariable
    
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
        
        return level, '.'.join(topics)
            
            
    def log_message(self, raw):
        """receive and parse a message, then log it."""
        raw = [r.decode("utf-8") for r in raw]
        
        if len(raw) != 2 or '.' not in raw[0]:
            logging.getLogger().error("Invalid log message: %s"%raw)
            return
        else:
            topic, msg = raw
            level, topic = self._extract_level(topic)
            
            # bit of a hacky way to filter messages
            # assumes subtopic only contains a single dot
            # main topic is now the substring with the origin of the message
            # so e.g. engine.1
            main_topic, subtopic = topic.rsplit('.',1) 
            log = logging.getLogger(subtopic) 
            
            if msg[-1] == '\n':
                msg = msg[:-1]
#             self.log.log(level, "[%s] %s" % (topic, msg))
            print("[%s] %s" % (main_topic, msg))

            log.log(level, "[%s] %s" % (main_topic, msg))


class Engine(object):
    '''class for setting up ema specific stuff on each engine
    also functions as a convenient namespace for workbench
    relevant variables
    
    Parameters
    ----------
    engine_id : int
    msis : list
    model_init_kwargs : dict, optional
    
    '''

    def __init__(self, engine_id, msis):
        self.engine_id = engine_id
        self.msis = msis
        self.runner = experiment_runner.ExperimentRunner(msis)

    def setup_working_directory(self, dir_name):
        '''setup the root directory for the engine. The working directories 
        associated with the various model structure interfaces will be copied 
        to this root directory.

        Parameters
        ----------
        dir_name : str
                   The name of the root directory


        '''
        dir_name = dir_name.format(self.engine_id)
        
        dir_name = os.path.join(EMA_PROJECT_HOME_DIR, dir_name)
        
        # if the directory already exists, is has not been
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
        
        # this is hacky
        self.runner.cleanup()

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
                
    def run_experiment(self, experiment):
        '''run the experiment, the actual running is delegated
        to an ExperimentRunner instance'''
        
        try:
            return self.runner.run_experiment(experiment) 
        except ema_exceptions.EMAError:
            raise
        except Exception:
            raise ema_exceptions.EMAParallelError(str(Exception))
       

def initialize_engines(client, msis):
    '''initialize engine instances on all engines
    
    Parameters
    ----------
    client : IPython.parallel.Client 
    msis : dict
           dict of model structure interfaces with their names as keys
    
    '''
    for i in client.ids:
        client[i].apply_sync(_initialize_engine, i, msis)
        
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
    for msi in msis:
        try:
            wd_by_msi[msi.working_directory].append(msi.name)
        except AttributeError:
            pass
        
    if len(wd_by_msi) == 0 :
        return
        
    # determine the common root of all working directories
    common_root = os.path.commonprefix(wd_by_msi.keys())
    common_root = os.path.dirname(common_root)
    rel_common_root = os.path.relpath(common_root, EMA_PROJECT_HOME_DIR)
    
    engine_wd_name = 'engine{}'
    engine_dir = os.path.join(rel_common_root, engine_wd_name)
        
    # create the root directory for each engine
    # we need to block until directory has been created
    client[:].apply_sync(_setup_working_directory, engine_dir)
    
    # copy the working directories and update msi.working_directory
    dirs_to_copy = wd_by_msi.keys()
    client[:].apply_sync(_copy_working_directories, dirs_to_copy, wd_by_msi)


def cleanup_working_directories(client):
    '''cleanup directory tree structure on all engines '''
    client[:].apply_sync(_cleanun_working_directory)


# engines can only deal with functions, not with object.method calls
# these functions are wrappers around the relevant Engine methods
# the engine instance is part of the namespace of the module. 
def _run_experiment(experiment):
    return experiment, engine.run_experiment(experiment)


def _initialize_engine(engine_id, msis):
    global engine
    engine = Engine(engine_id, msis)

    
def _setup_working_directory(dir_name):
    engine.setup_working_directory(dir_name)

    
def _cleanun_working_directory():
    engine.cleanup_working_directory()

    
def _copy_working_directories(dirs_to_copy, wd_by_msi):
    engine.copy_wds_for_msis(dirs_to_copy, wd_by_msi)