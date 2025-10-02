"""Functionality for combining the EMA workbench with IPython parallel."""

import collections
import logging
import os
import shutil
import socket
import threading
from collections.abc import Callable, Iterable

import zmq
from ipyparallel.engine.log import EnginePUBHandler
from jupyter_client.localinterfaces import localhost
from tornado import ioloop
from traitlets import Instance, List, Unicode
from traitlets.config import Application
from traitlets.config.configurable import LoggingConfigurable
from zmq.eventloop import zmqstream

from ..util import ema_exceptions, ema_logging, get_module_logger
from . import experiment_runner
from .evaluators import BaseEvaluator
from .futures_util import setup_working_directories
from .points import Experiment

# Created on Jul 16, 2015
#
# .. codeauthor::  jhkwakkel

__all__ = ["IpyparallelEvaluator"]

SUBTOPIC = ema_logging.LOGGER_NAME
engine = None

_logger = get_module_logger(__name__)


class EngingeLoggerAdapter(logging.LoggerAdapter):
    """LoggerAdapter that inserts EMA as a topic into log messages.

    Parameters
    ----------
    logger : logger instance
    topic : str

    """

    def __init__(self, logger, topic):
        """Init."""
        super().__init__(logger, None)
        self.topic = topic

    def process(self, msg, kwargs):
        """Process the log message."""
        msg = f"{self.topic}::{msg}"
        return msg, kwargs


def start_logwatcher():
    """Convenience function for starting the LogWatcher.

    Returns
    -------
    LogWatcher
        the log watcher instance
    Thread
        the log watcher thread

    .. note : there can only be one log watcher on a given url.

    """
    logwatcher = LogWatcher()

    def starter():
        logwatcher.start()
        try:
            logwatcher.loop.start()
        except (zmq.error.ZMQError, OSError):
            _logger.warning("shutting down log watcher")
        except RuntimeError:
            pass

    logwatcher_thread = threading.Thread(target=starter)
    logwatcher_thread.daemon = True
    logwatcher_thread.start()

    return logwatcher, logwatcher_thread


def set_engine_logger():
    """Updates EMA logging on the engines with an EngineLoggerAdapter.

    This adapter injects EMA as a topic into all messages
    """
    logger = Application.instance().log
    logger.setLevel(ema_logging.DEBUG)

    for handler in logger.handlers:
        if isinstance(handler, EnginePUBHandler):  # @UndefinedVariable
            handler.setLevel(ema_logging.DEBUG)

    adapter = EngingeLoggerAdapter(logger, SUBTOPIC)
    ema_logging._logger = adapter

    _logger.debug("updated logger")


def get_engines_by_host(client):
    """Returns the engine ids by host.

    Parameters
    ----------
    client : IPython.parallel.Client instance

    Returns
    -------
    dict
        a dict with hostnames as keys, and a list
        of engine ids

    """
    results = {i: client[i].apply_sync(socket.gethostname) for i in client.ids}

    engines_by_host = collections.defaultdict(list)
    for engine_id, host in results.items():
        engines_by_host[host].append(engine_id)
    return engines_by_host


def update_cwd_on_all_engines(client):
    """Updates cwd on the engines to point to the same working directory as the notebook.

    Currently only works if engines are on same machine.

    Parameters
    ----------
    client : IPython.parallel.Client instance

    """
    engines_by_host = get_engines_by_host(client)

    notebook_host = socket.gethostname()
    for key, value in engines_by_host.items():
        if key == notebook_host:
            cwd = os.getcwd()

            # easy, we know the correct cwd
            for engine in value:
                client[engine].apply(os.chdir, cwd)
        else:
            raise NotImplementedError("not yet supported")


class LogWatcher(LoggingConfigurable):
    """Helper class for handling logging.

    A simple class that receives messages on a SUB socket, as published
    by subclasses of `zmq.log.handlers.PUBHandler`, and logs them itself.

    This can subscribe to multiple topics, but defaults to all topics.
    """

    # configurables
    topics = List(
        [""],
        help=("The ZMQ topics to subscribe to. Default is tosubscribe to all messages"),
    ).tag(config=True)
    url = Unicode(help="ZMQ url on which to listen for log messages").tag(config=True)

    # internals
    stream = Instance("zmq.eventloop.zmqstream.ZMQStream", allow_none=True)
    context = Instance(zmq.Context)
    loop = Instance("tornado.ioloop.IOLoop")  # @UndefinedVariable

    def _url_default(self):
        """Default url."""
        return f"tcp://{localhost()}:20202"

    def _context_default(self):
        """Default context."""
        return zmq.Context.instance()

    def _loop_default(self):
        """Default io loop."""
        return ioloop.IOLoop.instance()

    def __init__(self, **kwargs):
        """Init."""
        super().__init__(**kwargs)
        self.s = self.context.socket(zmq.SUB)  # @UndefinedVariable
        self.s.bind(self.url)
        self.stream = zmqstream.ZMQStream(self.s, self.loop)
        self.subscribe()
        self.observe(self.subscribe, "topics")

    def start(self):
        """Start the log watcher."""
        self.stream.on_recv(self.log_message)

    def stop(self):
        """Stop the log watcher."""
        self.stream.setsockopt(zmq.UNSUBSCRIBE, b"")  # @UndefinedVariable
        self.stream.stop_on_recv()
        self.s.close()

    def subscribe(self):
        """Update our SUB socket's subscriptions."""
        self.stream.setsockopt(zmq.UNSUBSCRIBE, b"")  # @UndefinedVariable
        if "" in self.topics:
            self.log.debug("Subscribing to: everything")
            self.stream.setsockopt(zmq.SUBSCRIBE, b"")  # @UndefinedVariable
        else:
            for topic in self.topics:
                self.log.debug(f"Subscribing to: {topic!r}")
                # @UndefinedVariable
                self.stream.setsockopt(zmq.SUBSCRIBE, topic)

    def _extract_level(self, topic_str):
        """Turn 'engine.0.INFO.extra' into (logging.INFO, 'engine.0.extra')."""
        topics = topic_str.split(".")
        for idx, t in enumerate(topics):  # noqa: B007
            level = getattr(logging, t, None)
            if level is not None:
                break

        if level is None:
            level = logging.INFO
        else:
            topics.pop(idx)

        return level, ".".join(topics)

    def log_message(self, raw):
        """Receive and parse a message, then log it."""
        raw = [r.decode("utf-8") for r in raw]

        if len(raw) != 2 or "." not in raw[0]:
            logging.getLogger().error(f"Invalid log message: {raw}")
            return
        else:
            topic, msg = raw
            level, topic = self._extract_level(topic)

            # bit of a hacky way to filter messages
            # assumes subtopic only contains a single dot
            # main topic is now the substring with the origin of the message
            # so e.g. engine.1
            main_topic, subtopic = topic.rsplit(".", 1)
            log = logging.getLogger(subtopic)

            if msg[-1] == "\n":
                msg = msg[:-1]

            log.log(level, f"[{main_topic}] {msg}")


class Engine:
    """Engine class.

    class for setting up ema specific stuff on each engine
    also functions as a convenient namespace for workbench
    relevant variables

    Parameters
    ----------
    engine_id : int
    msis : list
    cwd : str

    """

    def __init__(self, engine_id, msis, cwd):
        """Init."""
        _logger.debug(f"starting engine {engine_id}")
        self.engine_id = engine_id
        self.msis = msis

        _logger.debug(f"setting root working directory to {cwd}")
        os.chdir(cwd)

        self.runner = experiment_runner.ExperimentRunner(msis)

        self.tmpdir = setup_working_directories(msis, os.getcwd())

    def cleanup_working_directory(self):
        """Remove the root working directory of the engine."""
        if self.tmpdir:
            shutil.rmtree(self.tmpdir)

    def run_experiment(self, experiment):
        """Run the experiment.

        The actual running is delegated to an ExperimentRunner instance
        """
        try:
            return self.runner.run_experiment(experiment)
        except ema_exceptions.EMAError:
            raise
        except Exception as e:
            raise ema_exceptions.EMAParallelError(str(Exception)) from e


def initialize_engines(client, msis, cwd):
    """Initialize engine instances on all engines.

    Parameters
    ----------
    client : IPython.parallel.Client
    msis : dict
           dict of model structure interfaces with their names as keys
    cwd : str

    """
    for i in client.ids:
        client[i].apply_sync(_initialize_engine, i, msis, cwd)


def cleanup(client):
    """Cleanup directory tree structure on all engines."""
    client[:].apply_sync(_cleanup)


# engines can only deal with functions, not with object.method calls
# these functions are wrappers around the relevant Engine methods
# the engine instance is part of the namespace of the module.
def _run_experiment(experiment):
    """Wrapper function for engine.run_experiment."""
    return experiment, engine.run_experiment(experiment)


def _initialize_engine(engine_id, msis, cwd):
    """Wrapper function for initializing an engine."""
    global engine  # noqa PLW0603
    engine = Engine(engine_id, msis, cwd)


def _cleanup():
    """Wrapper function for engine.cleanup_working_directory."""
    global engine  # noqa: PLW0603
    engine.cleanup_working_directory()
    del engine


class IpyparallelEvaluator(BaseEvaluator):
    """evaluator for using an ipypparallel pool."""

    def __init__(self, msis, client, **kwargs):
        """Init."""
        super().__init__(msis, **kwargs)
        self.client = client

    def initialize(self):
        """Initialize the pool."""
        import ipyparallel

        _logger.debug("starting ipyparallel pool")

        try:
            TIMEOUT_MAX = threading.TIMEOUT_MAX  # noqa: N806
        except AttributeError:
            TIMEOUT_MAX = 1e10  # noqa: N806
        ipyparallel.client.asyncresult._FOREVER = TIMEOUT_MAX
        # update loggers on all engines
        self.client[:].apply_sync(set_engine_logger)

        _logger.debug("initializing engines")
        initialize_engines(self.client, self._msis, os.getcwd())

        self.logwatcher, self.logwatcher_thread = start_logwatcher()

        _logger.debug("successfully started ipyparallel pool")
        _logger.info("performing experiments using ipyparallel")

        return self

    def finalize(self):
        """Finalize the pool."""
        self.logwatcher.stop()
        cleanup(self.client)

    def evaluate_experiments(
        self, experiments: Iterable[Experiment], callback: Callable, **kwargs
    ):
        """Evaluate experiments."""
        lb_view = self.client.load_balanced_view()
        results = lb_view.map(
            _run_experiment, experiments, ordered=False, block=False, **kwargs
        )

        for entry in results:
            callback(*entry)
