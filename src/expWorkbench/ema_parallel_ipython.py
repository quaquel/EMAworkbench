'''
Created on Jul 16, 2015

@author: jhkwakkel
'''
import collections
import logging
import os
import socket
import threading
import zmq

import IPython
from IPython.config import Application

import ema_logging


SUBTOPIC = "EMA"

class EngingeLoggerAdapter(logging.LoggerAdapter):
    '''LoggerAdapter that inserts a topic at the start
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
    logger. This filtering is done on a level and topic basis. By default,
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
        
        ema_logging.info('start watching on {}'.format(self.url))
        self.stream.on_recv(self.log_message)
    
    def stop(self):
        '''stop the log watcher'''
        self.stream.stop_on_recv()
 
    def subscribe(self):
        """Update our SUB socket's subscriptions."""
        ema_logging.debug("Subscribing to: everything")
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
    '''Updates EMA logging with a logger adapter to the logger
    of the engines. This adapter injects EMA as a topic into all messages
    '''
    
    logger = Application.instance().log
    logger.setLevel(ema_logging.DEBUG)
    for handler in logger.handlers:
        if isinstance(handler, IPython.kernel.zmq.log.EnginePUBHandler): # @UndefinedVariable
            handler.setLevel(logging.DEBUG)
    
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
    
    def engine_hostname():
        import socket
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
            import os
            os.chdir(cwd)

        if key == notebook_host:
            cwd = os.getcwd()

            # easy, we now the correct cwd
            for engine in value:
                client[engine].apply(set_cwd_on_engine, cwd)
        else:
            raise NotImplementedError('not yet supported')