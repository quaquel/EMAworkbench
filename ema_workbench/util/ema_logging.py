"""

This module contains code for logging EMA processes. It is modeled on the
default `logging approach that comes with
Python <https://docs.python.org/library/logging.html>`_.
This logging system will also work in case of multiprocessing.

"""
import inspect
import logging
from contextlib import contextmanager
from functools import wraps
from logging import DEBUG, INFO

# Created on 23 dec. 2010
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = [
    "get_rootlogger",
    "get_module_logger",
    "log_to_stderr",
    "temporary_filter",
    "DEBUG",
    "INFO",
    "DEFAULT_LEVEL",
    "LOGGER_NAME",
    "method_logger",
]
LOGGER_NAME = "EMA"
DEFAULT_LEVEL = DEBUG
INFO = INFO


def create_module_logger(name=None):
    if name is None:
        frm = inspect.stack()[1]
        mod = inspect.getmodule(frm[0])
        name = mod.__name__
    logger = logging.getLogger(f"{LOGGER_NAME}.{name}")

    _module_loggers[name] = logger
    return logger


def get_module_logger(name):
    try:
        logger = _module_loggers[name]
    except KeyError:
        logger = create_module_logger(name)

    return logger


_rootlogger = None
_module_loggers = {}
_logger = get_module_logger(__name__)

LOG_FORMAT = "[%(processName)s/%(levelname)s] %(message)s"


class TemporaryFilter(logging.Filter):
    def __init__(self, *args, level=0, funcname=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.level = level
        self.funcname = funcname

    def filter(self, record):
        if self.funcname:
            if self.funcname != record.funcName:
                return True

        return record.levelno > self.level


@contextmanager
def temporary_filter(name=LOGGER_NAME, level=0, functname=None):
    """temporary filter log message

    Params
    ------
    name : str or list of str, optional
           logger on which to apply the filter.
    level: int, or list of int, optional
           don't log message of this level or lower
    funcname : str or list of str, optional
            don't log message of this function

    all modules have their own unique logger
    (e.g. ema_workbench.analysis.prim)

    """
    # TODO:: probably all three should be optionally a list so you
    # might filter multiple log message from different functions
    if isinstance(name, str):
        names = [name]
    else:
        names = name

    if isinstance(level, int):
        levels = [level]
    else:
        levels = level

    if isinstance(functname, str) or functname is None:
        functnames = [functname]
    else:
        functnames = functname
    # get logger
    # add filter
    max_length = max(len(names), len(levels), len(functnames))

    # make a list equal lengths?
    if len(names) < max_length:
        names = [name,] * max_length
    if len(levels) < max_length:
        levels = [level,] * max_length
    if len(functnames) < max_length:
        functnames = [functname,] * max_length

    filters = {}
    for name, level, functname in zip(names, levels, functnames):
        logger = get_module_logger(name)
        filter = TemporaryFilter(level=level, funcname=functname)  # @ReservedAssignment

        if logger == _logger:
            # root logger, add filter to handler rather than logger
            # because filters don't propagate for some unclear reason
            for handler in logger.handlers:
                handler.addFilter(filter)
                filters[filter] = handler
        else:
            logger.addFilter(filter)
            filters[filter] = logger

    yield

    for k, v in filters.items():
        v.removeFilter(k)


def method_logger(name):
    logger = get_module_logger(name)
    classname = inspect.getouterframes(inspect.currentframe())[1][3]

    def real_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # hack, because log is applied to methods, we can get
            # object instance as first arguments in args
            logger.debug(f"calling {func.__name__} on {classname}")
            res = func(*args, **kwargs)
            logger.debug(f"completed calling {func.__name__} on {classname}")
            return res

        return wrapper

    return real_decorator


def get_rootlogger():
    """
    Returns root logger used by the EMA workbench

    Returns
    -------
    the logger of the EMA workbench

    """
    global _rootlogger

    if not _rootlogger:
        _rootlogger = logging.getLogger(LOGGER_NAME)
        _rootlogger.handlers = []
        _rootlogger.addHandler(logging.NullHandler())
        _rootlogger.setLevel(DEBUG)

    return _rootlogger


def log_to_stderr(level=None):
    """
    Turn on logging and add a handler which prints to stderr

    Parameters
    ----------
    level : int
            minimum level of the messages that will be logged

    """

    if not level:
        level = DEFAULT_LEVEL

    logger = get_rootlogger()

    # avoid creation of multiple stream handlers for logging to console
    for entry in logger.handlers:
        if (isinstance(entry, logging.StreamHandler)) and (
            entry.formatter._fmt == LOG_FORMAT
        ):
            return logger

    formatter = logging.Formatter(LOG_FORMAT)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
