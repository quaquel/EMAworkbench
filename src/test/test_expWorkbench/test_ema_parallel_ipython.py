'''
Created on Jul 28, 2015

test code for ema_parallel_ipython. The setup and teardown of the cluster is
taken from the ipyparallel test code with some minor adaptations

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
import ctypes
import logging
import mock
import os
import subprocess
import time
import unittest


from IPython.utils.path import get_ipython_dir
from IPython.utils.localinterfaces import localhost
import IPython.parallel as parallel

from IPython.parallel.apps.launcher import (LocalProcessLauncher,
                                                  ipengine_cmd_argv,
                                                  ipcontroller_cmd_argv,
                                                  SIGKILL,
                                                  ProcessStateError)

import expWorkbench.ema_parallel_ipython as ema
from expWorkbench import ema_logging

launchers =[]
blackhole = open(os.devnull, 'w')

# Launcher class
class TestProcessLauncher(LocalProcessLauncher):
    """subclass LocalProcessLauncher, to prevent extra sockets and threads being created 
    on Windows"""
    def start(self):
        if self.state == 'before':
            # Store stdout & stderr to show with failing tests.
            # This is defined in IPython.testing.iptest
            self.process = subprocess.Popen(self.args,
                stdout=blackhole, stderr=subprocess.STDOUT,
                env=os.environ,
                cwd=self.work_dir
            )
            self.notify_start(self.process.pid)
            self.poll = self.process.poll
        else:
            s = 'The process was already started and has state: %r' % self.state
            raise ProcessStateError(s)


def add_engines(n=1, profile='default', total=False):
    """add a number of engines to a given profile.
    
    If total is True, then already running engines are counted, and only
    the additional engines necessary (if any) are started.
    """
    rc = parallel.Client(profile=profile)
    base = len(rc)
    
    if total:
        n = max(n - base, 0)
    
    eps = []
    for _ in range(n):
        ep = TestProcessLauncher()
        ep.cmd_and_args = ipengine_cmd_argv + [
            '--profile=%s' % profile,
            '--log-level=50',
            '--InteractiveShell.colors=nocolor'
            ]
        ep.start()
        launchers.append(ep)
        eps.append(ep)
    tic = time.time()
    while len(rc) < base+n:
        if any([ ep.poll() is not None for ep in eps ]):
            raise RuntimeError("A test engine failed to start.")
        elif time.time()-tic > 15:
            raise RuntimeError("Timeout waiting for engines to connect.")
        time.sleep(.1)
        rc.spin()
    rc.close()
    return eps

def setUpModule():
    cluster_dir = os.path.join(get_ipython_dir(), 'profile_default')
    engine_json = os.path.join(cluster_dir, 'security', 'ipcontroller-engine.json')
    client_json = os.path.join(cluster_dir, 'security', 'ipcontroller-client.json')
    for json in (engine_json, client_json):
        if os.path.exists(json):
            os.remove(json)
    
    cp = TestProcessLauncher()
    cp.cmd_and_args = ipcontroller_cmd_argv + \
                ['--profile=default', '--log-level=20']
    cp.start()
    launchers.append(cp)
    tic = time.time()
    while not os.path.exists(engine_json) or not os.path.exists(client_json):
        if cp.poll() is not None:
            raise RuntimeError("The test controller exited with status %s" % cp.poll())
        elif time.time()-tic > 15:
            raise RuntimeError("Timeout waiting for the test controller to start.")
        time.sleep(0.1)
        
    add_engines(2, profile='default', total=True)
    
    client = parallel.Client(profile='default')
    print client.ids

def tearDownModule():
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        return
    while launchers:
        p = launchers.pop()
        if p.poll() is None:
            try:
                p.stop()
            except Exception as e:
                print(e)
                pass
        if p.poll() is None:
            try:
                time.sleep(.25)
            except KeyboardInterrupt:
                return
        if p.poll() is None:
            try:
                print('cleaning up test process...')
                p.signal(SIGKILL)
            except:
                print("couldn't shutdown process: ", p)
    blackhole.close()

class TestEngineLoggerAdapter(unittest.TestCase):
     
    def tearDown(self):
        ema_logging._logger = None
        ema_logger = logging.getLogger(ema_logging.LOGGER_NAME)
        ema_logger.handlers = []
 
    def test_directly(self):
        with mock.patch('expWorkbench.ema_logging._logger') as mocked_logger:
            adapter = ema.EngingeLoggerAdapter(mocked_logger, ema.SUBTOPIC)
            self.assertEqual(mocked_logger, adapter.logger)
            self.assertEqual(ema.SUBTOPIC, adapter.topic)
  
  
            input_msg = 'test'
            input_kwargs = {}
            msg, kwargs = adapter.process(input_msg, input_kwargs)
              
            self.assertEqual('{}::{}'.format(ema.SUBTOPIC, input_msg), 
                             msg)
            self.assertEqual(input_kwargs, kwargs)
         
 
    def test_on_cluster(self):
        client = parallel.Client(profile='default')
        client[:].apply_sync(ema.set_engine_logger)
         
        def test_engine_logger():
            from expWorkbench import ema_logging # @Reimport
            from expWorkbench import ema_parallel_ipython as ema # @Reimport
             
            logger = ema_logging._logger
             
            tests = []
            tests.append((type(logger) == ema.EngingeLoggerAdapter,
                          'logger adapter'))
            tests.append((logger.logger.level == ema_logging.DEBUG,
                          'logger level'))
            tests.append((logger.topic == ema.SUBTOPIC,
                          'logger subptopic'))
            return tests
             
        for engine in client.ids:
            tests = client[engine].apply_sync(test_engine_logger)
            for test in tests:
                test, msg = test
                self.assertTrue(test, msg)
         
        client.clear(block=True)        


class TestLogWatcher(unittest.TestCase):
     
    @classmethod
    def setUpClass(cls):
        with mock.patch('expWorkbench.ema_logging._logger') as mocked_logger:   
            cls.client = parallel.Client(profile='default')
            cls.url = 'tcp://{}:20202'.format(localhost())
            cls.watcher, cls.thread = ema.start_logwatcher(cls.url)

    @classmethod
    def tearDownClass(cls):
        cls.watcher.stop()
        
        time.sleep(2)
        
        # horrible hack to kill the watcher thread
        # for some reason despite it being a deamon thread, it
        # is not terminated properly
        if cls.thread.isAlive():
            
            exc = ctypes.py_object(KeyboardInterrupt)
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(cls.thread.ident), exc)
            if res == 0:
                raise ValueError("nonexistent thread id")
            elif res > 1:
                # """if it returns a number greater than one, you're in trouble,
                # and you should call it again with exc=NULL to revert the effect"""
                ctypes.pythonapi.PyThreadState_SetAsyncExc(cls.thread.ident, None)
                raise SystemError("PyThreadState_SetAsyncExc failed")

    def tearDown(self):
        self.client.clear(block=True)
 
    def test_init(self):
        self.assertEqual(self.url, self.watcher.url)
        self.assertEqual({ema_logging.DEBUG:{'EMA'}}, self.watcher.topic_subscriptions)
 
    def test_extract_level(self):
        level = 'INFO'
        topic = ema.SUBTOPIC
        topic_str = 'engine.1.{}.{}'.format(level, topic)
        extracted_level, extracted_topics = self.watcher._extract_level(topic_str)
        
        self.assertEqual(ema_logging.INFO, extracted_level)
        self.assertEqual(['engine', '1', topic], extracted_topics)
        
        topic = ema.SUBTOPIC
        topic_str = 'engine.1.{}'.format(topic)
        extracted_level, extracted_topics = self.watcher._extract_level(topic_str)
        
        self.assertEqual(ema_logging.INFO, extracted_level)
        self.assertEqual(['engine', '1', topic], extracted_topics)
     
    def test_subscribe_to_topic(self):

        self.watcher.subscribe_to_topic(ema_logging.INFO, 'EMA')

        self.assertEqual({ema_logging.DEBUG:{'EMA'},
                          ema_logging.INFO: {'EMA'}}, self.watcher.topic_subscriptions)

     
    def test_log_message(self):
        # no subscription on level
        with mock.patch('expWorkbench.ema_logging._logger') as mocked_logger:   
            self.watcher.logger = mocked_logger
            raw = ['engine.1', 'test']
            self.watcher.log_message(raw)
            mocked_logger.log.assert_called_once_with(ema_logging.INFO, '[engine.1] test')
        
        with mock.patch('expWorkbench.ema_logging._logger') as mocked_logger:   
            self.watcher.logger = mocked_logger
            raw = ['engine.1.DEBUG.EMA', 'test']
            self.watcher.log_message(raw)
            mocked_logger.log.assert_called_once_with(ema_logging.DEBUG, '[engine.1.EMA] test')

        with mock.patch('expWorkbench.ema_logging._logger') as mocked_logger:   
            self.watcher.logger = mocked_logger
            raw = ['engine.1.DEBUG', 'test', 'more']
            self.watcher.log_message(raw)
            mocked_logger.error.assert_called_once_with("Invalid log message: %s"%raw)

        with mock.patch('expWorkbench.ema_logging._logger') as mocked_logger:   
            self.watcher.logger = mocked_logger
            raw = ['engine1DEBUG', 'test']
            self.watcher.log_message(raw)
            mocked_logger.error.assert_called_once_with("Invalid log message: %s"%raw)


class TestEngine(unittest.TestCase):

    def testName(self):
        pass


if __name__ == "__main__":

    unittest.main()