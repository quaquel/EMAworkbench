'''

This module contains code for logging EMA processes. It is modeled on the 
default `logging approach that comes with Python <http://docs.python.org/library/logging.html>`_. 
This logging system will also work in case of multiprocessing using 
:mod:`ema_parallel`.

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

from functools import wraps
import inspect
from contextlib import contextmanager

import logging
from logging import Handler, DEBUG, INFO

# Created on 23 dec. 2010
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['debug',
           'info',
           'warning',
           'error',
           'exception',
           'critical',
           'get_logger',
           'get_module_logger',
           'log_to_stderr',
           'temporary_filter',
           'DEBUG',
           'INFO',
           'DEFAULT_LEVEL',
           'LOGGER_NAME']

_logger = None
_module_loggers = {}
LOGGER_NAME = "EMA"
DEFAULT_LEVEL = DEBUG
INFO = INFO

LOG_FORMAT = '[%(processName)s/%(levelname)s] %(message)s'


def create_module_logger(name=None):
    if name is None:
        frm = inspect.stack()[1]
        mod = inspect.getmodule(frm[0])
        name = mod.__name__
    logger = logging.getLogger("{}.{}".format(LOGGER_NAME, name))

    _module_loggers[name] = logger
    return logger

def get_module_logger(name):
    try:
        logger = _module_loggers[name]
    except KeyError:
        logger = create_module_logger(name)
    
    return logger

class TemporaryFilter(logging.Filter):
    
    def __init__(self, *args, level=0, funcname=None, **kwargs):
        super(TemporaryFilter, self).__init__(*args, **kwargs)
        self.level = level
        self.funcname = funcname
    
    def filter(self, record):
        if self.funcname:
            if self.funcname != record.funcName:
                return True  
        
        return record.levelno > self.level


@contextmanager
def temporary_filter(name=LOGGER_NAME, level=0, functname=None):
    ''' temporary filter log message
    
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
    
    TODO:: probably all three should beoptionally be a list so you
    might filter multiple log message from different functions
    
    
    '''
    if isinstance(name, str):
        names = [name]
    else:
        names = name
    
    if isinstance(level, int):
        levels = [level]
    else:
        levels = level
    
    if isinstance(functname, str) or functname==None:
        functnames = [functname]
    else:
        functnames = functname
    # get logger
    # add filter
    max_length = max(len(names), len(levels), len(functnames))
    
    # make a list equal lengths?
    if len(names) < max_length:
        names = [name,]*max_length
    if len(levels) < max_length:
        levels = [level,]*max_length
    if len(functnames) < max_length:
        functnames = [functname, ]* max_length
    
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

    

def method_logger(func):
    classname = inspect.getouterframes(inspect.currentframe())[1][3]

    @wraps(func)
    def wrapper(*args, **kwargs):
        # hack, because log is applied to methods, we can get
        # object instance as first arguments in args
        debug('calling {} on {}'.format(func.__name__, classname))
        res = func(*args, **kwargs)
        debug('completed calling {} on {}'.format(func.__name__, classname))
        return res
    return wrapper


def debug(msg, *args, **kwargs):
    '''
    convenience function for logger.debug

    Parameters
    ----------
    msg : str
          msg to log
    args : list
           args to pass on to the logger
    kwargs : dict
             kwargs to pass on to the logger

    '''
    # frm = inspect.stack()[1]
    # mod = inspect.getmodule(frm[0])
    # name = mod.__name__
    # logger = get_module_logger(name)
    #
    # if _logger:
    #     logger.debug(msg, *args, **kwargs)
    pass


def info(msg, *args):
    '''
    convenience function for logger.info

    Parameters
    ----------
    msg : str
          msg to log
    args : list
           args to pass on to the logger
    kwargs : dict
             kwargs to pass on to the logger

    '''
    # frm = inspect.stack()[1]
    # mod = inspect.getmodule(frm[0])
    # name = mod.__name__
    # logger = get_module_logger(name)
    #
    # if _logger:
    #     logger.info(msg, *args)
    pass

def warning(msg, *args):
    '''
    convenience function for logger.warning

    Parameters
    ----------
    msg : str
          msg to log
    args : list
           args to pass on to the logger
    kwargs : dict
             kwargs to pass on to the logger

    '''
    # frm = inspect.stack()[1]
    # mod = inspect.getmodule(frm[0])
    # name = mod.__name__
    # logger = get_module_logger(name)
    #
    # if _logger:
    #     logger.warning(msg, *args)
    pass

def error(msg, *args):
    '''
    convenience function for logger.error

    Parameters
    ----------
    msg : str
          msg to log
    args : list
           args to pass on to the logger
    kwargs : dict
             kwargs to pass on to the logger

    '''
    # frm = inspect.stack()[1]
    # mod = inspect.getmodule(frm[0])
    # name = mod.__name__
    # logger = get_module_logger(name)
    #
    # if _logger:
    #     logger.error(msg, *args)
    pass

def exception(msg, *args):
    '''
    convenience function for logger.exception

    Parameters
    ----------
    msg : str
          msg to log
    args : list
           args to pass on to the logger
    kwargs : dict
             kwargs to pass on to the logger

    '''
    # frm = inspect.stack()[1]
    # mod = inspect.getmodule(frm[0])
    # name = mod.__name__
    # logger = get_module_logger(name)
    #
    # if _logger:
    #     logger.exception(msg, *args)
    pass

def critical(msg, *args):
    '''
    convenience function for logger.critical

    Parameters
    ----------
    msg : str
          msg to log
    args : list
           args to pass on to the logger
    kwargs : dict
             kwargs to pass on to the logger

    '''
    # frm = inspect.stack()[1]
    # mod = inspect.getmodule(frm[0])
    # name = mod.__name__
    # logger = get_module_logger(name)
    #
    # if _logger:
    #     logger.critical(msg, *args)
    pass

def get_logger():
    '''
    Returns logger used by the EMA workbench

    Returns
    -------
    the logger of the EMA workbench

    '''
    global _logger

    if not _logger:
        _logger = logging.getLogger(LOGGER_NAME)
        _logger.handlers = []
        _logger.addHandler(NullHandler())
        _logger.setLevel(DEBUG)
        _module_loggers[LOGGER_NAME] = _logger

    return _logger


def log_to_stderr(level=None):
    '''
    Turn on logging and add a handler which prints to stderr

    Parameters
    ----------
    level : int
            minimum level of the messages that will be logged

    '''

    if not level:
        level = DEFAULT_LEVEL

    logger = get_logger()

    # avoid creation of multiple stream handlers for logging to console
    for entry in logger.handlers:
        if (isinstance(entry, logging.StreamHandler)) and\
           (entry.formatter._fmt == LOG_FORMAT):
            return logger

    formatter = logging.Formatter(LOG_FORMAT)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


class NullHandler(Handler):
    '''
    convenience handler that does nothing

    '''

    def emit(self, record):
        pass


# class TlsSMTPHandler(SMTPHandler):
#     '''
#     class for using gmail as a server for sending e-mails contain
#     logging messages
#     '''
#
#     def emit(self, record):
#         '''
#         Emit a record.
#
#         Format the record and send it to the specified addressees.
#         code found `online <http://mynthon.net/howto/-/python/python%20-%20logging.SMTPHandler-how-to-use-gmail-smtp-server.txt>`_
#
#         '''
#         try:
#             import smtplib
#             import string # for tls add this line
#             try:
#                 from email.utils import formatdate
#             except ImportError:
#                 formatdate = self.date_time
#             port = self.mailport
#             if not port:
#                 port = smtplib.SMTP_PORT
#             smtp = smtplib.SMTP(self.mailhost, port)
#             msg = self.format(record)
#             msg = "From: %s\r\nTo: %s\r\nSubject: %s\r\nDate: %s\r\n\r\n%s" % (
#                             self.fromaddr,
#                             string.join(self.toaddrs, ","),
#                             self.getSubject(record),
#                             formatdate(), msg)
#             if self.username:
#                 smtp.ehlo() # for tls add this line
#                 smtp.starttls() # for tls add this line
#                 smtp.ehlo() # for tls add this line
#                 smtp.login(self.username, self.password)
#             smtp.sendmail(self.fromaddr, self.toaddrs, msg)
#             smtp.quit()
#         except (KeyboardInterrupt, SystemExit):
#             raise
#         except:
#             self.handleError(record)
