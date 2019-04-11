'''
This module provides functionality for combining the EMA workbench
with IPython parallel.

'''
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

from ipyparallel.engine.log import EnginePUBHandler
from jupyter_client.localinterfaces import localhost

from . import experiment_runner
from .ema_multiprocessing import setup_working_directories
from .model import AbstractModel
from .util import NamedObjectMap
from ..util import ema_exceptions, ema_logging, get_module_logger

# Created on Jul 16, 2015
#
# .. codeauthor::  jhkwakkel

SUBTOPIC = ema_logging.LOGGER_NAME
engine = None

_logger = get_module_logger(__name__)


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
            _logger.warning('shutting down log watcher')
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
        if isinstance(handler, EnginePUBHandler):  # @UndefinedVariable
            handler.setLevel(ema_logging.DEBUG)

    adapter = EngingeLoggerAdapter(logger, SUBTOPIC)
    ema_logging._logger = adapter

    _logger.debug('updated logger')


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

    results = {i: client[i].apply_sync(socket.gethostname) for i in client.ids}

    engines_by_host = collections.defaultdict(list)
    for engine_id, host in results.items():
        engines_by_host[host].append(engine_id)
    return engines_by_host


def update_cwd_on_all_engines(client):
    ''' updates the current working directory on the engines to point to the
    same working directory as this notebook

    currently only works if engines are on same machine.

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
                  help=('The ZMQ topics to subscribe to. Default is to'
                        'subscribe to all messages')).tag(config=True)
    url = Unicode(
        help="ZMQ url on which to listen for log messages").tag(config=True)

    # internals
    stream = Instance('zmq.eventloop.zmqstream.ZMQStream', allow_none=True)
    context = Instance(zmq.Context)
    loop = Instance('tornado.ioloop.IOLoop')  # @UndefinedVariable

    def _url_default(self):
        return 'tcp://%s:20202' % localhost()

    def _context_default(self):
        return zmq.Context.instance()

    def _loop_default(self):
        return ioloop.IOLoop.instance()

    def __init__(self, **kwargs):
        super(LogWatcher, self).__init__(**kwargs)
        self.s = self.context.socket(zmq.SUB)  # @UndefinedVariable
        self.s.bind(self.url)
        self.stream = zmqstream.ZMQStream(self.s, self.loop)
        self.subscribe()
        self.observe(self.subscribe, 'topics')

    def start(self):
        self.stream.on_recv(self.log_message)

    def stop(self):
        self.stream.setsockopt(zmq.UNSUBSCRIBE, b'')  # @UndefinedVariable
        self.stream.stop_on_recv()
        self.s.close()

    def subscribe(self):
        """Update our SUB socket's subscriptions."""
        self.stream.setsockopt(zmq.UNSUBSCRIBE, b'')  # @UndefinedVariable
        if '' in self.topics:
            self.log.debug("Subscribing to: everything")
            self.stream.setsockopt(zmq.SUBSCRIBE, b'')  # @UndefinedVariable
        else:
            for topic in self.topics:
                self.log.debug("Subscribing to: %r" % (topic))
                # @UndefinedVariable
                self.stream.setsockopt(zmq.SUBSCRIBE, topic)

    def _extract_level(self, topic_str):
        """Turn 'engine.0.INFO.extra' into (logging.INFO, 'engine.0.extra')"""
        topics = topic_str.split('.')
        for idx, t in enumerate(topics):
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
            logging.getLogger().error("Invalid log message: %s" % raw)
            return
        else:
            topic, msg = raw
            level, topic = self._extract_level(topic)

            # bit of a hacky way to filter messages
            # assumes subtopic only contains a single dot
            # main topic is now the substring with the origin of the message
            # so e.g. engine.1
            main_topic, subtopic = topic.rsplit('.', 1)
            log = logging.getLogger(subtopic)

            if msg[-1] == '\n':
                msg = msg[:-1]

            log.log(level, "[%s] %s" % (main_topic, msg))


class Engine(object):
    '''class for setting up ema specific stuff on each engine
    also functions as a convenient namespace for workbench
    relevant variables

    Parameters
    ----------
    engine_id : int
    msis : list
    cwd : str

    '''

    def __init__(self, engine_id, msis, cwd):
        _logger.debug("starting engine {}".format(engine_id))
        self.engine_id = engine_id
        self.msis = msis

        _logger.debug("setting root working directory to {}".format(cwd))
        os.chdir(cwd)

        models = NamedObjectMap(AbstractModel)
        models.extend(msis)
        self.runner = experiment_runner.ExperimentRunner(models)

        self.tmpdir = setup_working_directories(msis, os.getcwd())

    def cleanup_working_directory(self):
        '''remove the root working directory of the engine'''
        if self.tmpdir:
            shutil.rmtree(self.tmpdir)

    def run_experiment(self, experiment):
        '''run the experiment, the actual running is delegated
        to an ExperimentRunner instance'''

        try:
            return self.runner.run_experiment(experiment)
        except ema_exceptions.EMAError:
            raise
        except Exception:
            raise ema_exceptions.EMAParallelError(str(Exception))


def initialize_engines(client, msis, cwd):
    '''initialize engine instances on all engines

    Parameters
    ----------
    client : IPython.parallel.Client
    msis : dict
           dict of model structure interfaces with their names as keys
    cwd : str

    '''
    for i in client.ids:
        client[i].apply_sync(_initialize_engine, i, msis, cwd)


def cleanup(client):
    '''cleanup directory tree structure on all engines '''
    client[:].apply_sync(_cleanup)


# engines can only deal with functions, not with object.method calls
# these functions are wrappers around the relevant Engine methods
# the engine instance is part of the namespace of the module.
def _run_experiment(experiment):
    '''wrapper function for engine.run_experiment'''

    return experiment, engine.run_experiment(experiment)


def _initialize_engine(engine_id, msis, cwd):
    '''wrapper function for initializing an engine'''
    global engine
    engine = Engine(engine_id, msis, cwd)


def _cleanup():
    '''wrapper function for engine.cleanup_working_directory'''
    global engine
    engine.cleanup_working_directory()
    del engine
