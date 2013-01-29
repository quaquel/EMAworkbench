'''
Created on 23 dec. 2010

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

This module contains code for logging EMA processes. It is modeled on the 
default `logging approach that comes with Python <http://docs.python.org/library/logging.html>`_. 
This logging system will also work in case of multiprocessing using 
:mod:`parallelEMA`.

'''

import multiprocessing.util
from logging.handlers import SMTPHandler
from logging import Handler, DEBUG, INFO

SVN_ID = '$Id: ema_logging.py 818 2012-04-26 14:50:33Z jhkwakkel $'

__all__ =['debug',
          'info',
          'warning',
          'error',
          'exception',
          'critical',
          'get_logger',
          'log_to_stderr',
          'TlsSMTPHandler',
          'DEBUG',
          'INFO',
          'DEFAULT_LEVEL',
          'LOGGER_NAME']

_logger = None
LOGGER_NAME = "EMA"
DEFAULT_LEVEL = DEBUG

formatter = multiprocessing.util.DEFAULT_LOGGING_FORMAT

def debug(msg, *args):
    '''
    convenience function for logger.debug
    
    :param msg: msg to log
    :param args: args to pass on to the logger 
    
    '''
    
    if _logger:
        _logger.debug(msg, *args)

def info(msg, *args):
    '''
    convenience function for logger.info
        
    :param msg: msg to log
    :param args: args to pass on to the logger 
    
    '''
    if _logger:
        _logger.info(msg, *args)

def warning(msg, *args):
    '''
    convenience function for logger.warning
    
    :param msg: msg to log
    :param args: args to pass on to the logger 

    '''    
    
    if _logger:
        _logger.warning(msg, *args)

def error(msg, *args):
    '''
    convenience function for logger.error
    
    :param msg: msg to log
    :param args: args to pass on to the logger 

    '''  

    if _logger:
        _logger.error(msg, *args)

def exception(msg, *args):
    '''
    convenience function for logger.exception
    
    :param msg: msg to log
    :param args: args to pass on to the logger 

    '''      
    
    
    if _logger:
        _logger.exception(msg, *args)

def critical(msg, *args):
    '''
    convenience function for logger.critical
    
    :param msg: msg to log
    :param args: args to pass on to the logger 

    '''      
    
    if _logger:
        _logger.critical(msg, *args)

def get_logger():
    '''
    Returns logger used by the EMA workbench
    
    :returns: the logger of the EMA workbench
    
    '''
    global _logger
    import logging
    
    if not _logger:
            _logger = logging.getLogger(LOGGER_NAME)
            _logger.addHandler(NullHandler())

    return _logger

def log_to_stderr(level=None):
    '''
    Turn on logging and add a handler which prints to stderr
    
    :param level: minimum level of the messages that will be logged
    
    '''
    import logging

    logger = get_logger()
    formatter = logging.Formatter(multiprocessing.util.DEFAULT_LOGGING_FORMAT)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    if level:
        logger.setLevel(level)
    return _logger

class NullHandler(Handler):
    '''
    convenience handler that does nothing
    
    '''
    
    def emit(self, record):
        pass

 
class TlsSMTPHandler(SMTPHandler):
    '''
    class for using gmail as a server for sending e-mails contain
    logging messages
    '''
    
    def emit(self, record):
        '''
        Emit a record.
 
        Format the record and send it to the specified addressees.
        code found `online <http://mynthon.net/howto/-/python/python%20-%20logging.SMTPHandler-how-to-use-gmail-smtp-server.txt>`_
        
        '''
        try:
            import smtplib
            import string # for tls add this line
            try:
                from email.utils import formatdate
            except ImportError:
                formatdate = self.date_time
            port = self.mailport
            if not port:
                port = smtplib.SMTP_PORT
            smtp = smtplib.SMTP(self.mailhost, port)
            msg = self.format(record)
            msg = "From: %s\r\nTo: %s\r\nSubject: %s\r\nDate: %s\r\n\r\n%s" % (
                            self.fromaddr,
                            string.join(self.toaddrs, ","),
                            self.getSubject(record),
                            formatdate(), msg)
            if self.username:
                smtp.ehlo() # for tls add this line
                smtp.starttls() # for tls add this line
                smtp.ehlo() # for tls add this line
                smtp.login(self.username, self.password)
            smtp.sendmail(self.fromaddr, self.toaddrs, msg)
            smtp.quit()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)