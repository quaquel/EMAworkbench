'''
Created on Jul 28, 2015

test code for ema_parallel_ipython. The setup and teardown of the cluster is
taken from the ipyparallel test code with some minor adaptations

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
import logging

try:
    import unittest.mock as mock
except ImportError:
    import mock

import os
import socket
import subprocess
import time
import unittest

import IPython
from IPython.utils.path import get_ipython_dir
from IPython.utils.localinterfaces import localhost
import IPython.parallel as parallel

from IPython.parallel.apps.launcher import (LocalProcessLauncher,
                                            ipengine_cmd_argv,
                                            ipcontroller_cmd_argv,
                                            SIGKILL,
                                            ProcessStateError)

from ...em_framework import ema_parallel_ipython as ema
from ...em_framework import experiment_runner
from ... import em_framework

from ...util import ema_logging, EMAError, EMAParallelError


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
        with mock.patch('ema_workbench.util.ema_logging._logger') as mocked_logger:
            adapter = ema.EngingeLoggerAdapter(mocked_logger, ema.SUBTOPIC)
            self.assertEqual(mocked_logger, adapter.logger)
            self.assertEqual(ema.SUBTOPIC, adapter.topic)
  
  
            input_msg = 'test'
            input_kwargs = {}
            msg, kwargs = adapter.process(input_msg, input_kwargs)
              
            self.assertEqual('{}::{}'.format(ema.SUBTOPIC, input_msg), 
                             msg)
            self.assertEqual(input_kwargs, kwargs)
    
    @mock.patch('ema_workbench.em_framework.ema_parallel_ipython.EngingeLoggerAdapter')  
    @mock.patch('ema_workbench.em_framework.ema_parallel_ipython.Application')
    def test_engine_logger(self, mocked_application, mocked_adapter):
        logger = ema_logging.get_logger()
        mocked_logger = mock.Mock(spec=logger)
        mocked_logger.handlers = []
        mocked_logger.manager = mock.Mock(spec=logging.Manager)
        mocked_logger.manager.disable = 0
        ema_logging._logger = mocked_logger
        
        mocked_application.instance.return_value = mocked_application
        mocked_application.log = mocked_logger
        
        # no handlers    
        ema.set_engine_logger()
        logger = ema_logging._logger
#         self.assertTrue(type(logger) == type(mocked_adapter))
        mocked_logger.setLevel.assert_called_once_with(ema_logging.DEBUG)
        mocked_adapter.assert_called_with(mocked_logger, ema.SUBTOPIC)
        
        # with handlers
        mocked_logger = mock.create_autospec(logging.Logger)
        mock_engine_handler = mock.create_autospec(IPython.kernel.zmq.log.EnginePUBHandler)# @UndefinedVariable
        mocked_logger.handlers = [mock_engine_handler] 
        
        mocked_application.instance.return_value = mocked_application
        mocked_application.log = mocked_logger
        
        ema.set_engine_logger()
        logger = ema_logging._logger
#         self.assertTrue(type(logger) == ema.EngingeLoggerAdapter)
        mocked_logger.setLevel.assert_called_once_with(ema_logging.DEBUG)
        mocked_adapter.assert_called_with(mocked_logger, ema.SUBTOPIC)
        mock_engine_handler.setLevel.assert_called_once_with(ema_logging.DEBUG)
        
 
#     def test_on_cluster(self):
#         client = parallel.Client(profile='default')
#         client[:].apply_sync(ema.set_engine_logger)
#          
#         def test_engine_logger():
#             from em_framework import ema_logging # @Reimport
#             from em_framework import ema_parallel_ipython as ema # @Reimport
#              
#             logger = ema_logging._logger
#              
#             tests = []
#             tests.append((type(logger) == ema.EngingeLoggerAdapter,
#                           'logger adapter'))
#             tests.append((logger.logger.level == ema_logging.DEBUG,
#                           'logger level'))
#             tests.append((logger.topic == ema.SUBTOPIC,
#                           'logger subptopic'))
#             return tests
#              
#         for engine in client.ids:
#             tests = client[engine].apply_sync(test_engine_logger)
#             for test in tests:
#                 test, msg = test
#                 self.assertTrue(test, msg)
#          
#         client.clear(block=True)        


class TestLogWatcher(unittest.TestCase):
      
    @classmethod
    def setUpClass(cls):
        logger = ema_logging.get_logger()
        mocked_logger = mock.Mock(spec=logger)
        mocked_logger.handlers = []
        ema_logging._logger = mocked_logger

        cls.client = parallel.Client(profile='default')
        cls.url = 'tcp://{}:20202'.format(localhost())
        cls.watcher = ema.start_logwatcher(cls.url)
 
    @classmethod
    def tearDownClass(cls):
        cls.watcher.stop()

    @mock.patch('ema_workbench.util.ema_logging._logger', autospec=True)
    def test_stop(self, mocked_logger):
        url = 'tcp://{}:20201'.format(localhost())
        watcher = ema.start_logwatcher(url)
        
        watcher.stop()
        time.sleep(3)
        mocked_logger.warning.assert_called_once_with('shutting down log watcher')

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
        with mock.patch('ema_workbench.util.ema_logging._logger') as mocked_logger:   
            self.watcher.logger = mocked_logger
            raw = ['engine.1', 'test']
            self.watcher.log_message(raw)
            mocked_logger.log.assert_called_once_with(ema_logging.INFO, '[engine.1] test')
         
        with mock.patch('ema_workbench.util.ema_logging._logger') as mocked_logger:   
            self.watcher.logger = mocked_logger
            raw = ['engine.1.DEBUG.EMA', 'test']
            self.watcher.log_message(raw)
            mocked_logger.log.assert_called_once_with(ema_logging.DEBUG, '[engine.1.EMA] test')
 
        with mock.patch('ema_workbench.util.ema_logging._logger') as mocked_logger:   
            self.watcher.logger = mocked_logger
            raw = ['engine.1.DEBUG', 'test', 'more']
            self.watcher.log_message(raw)
            mocked_logger.error.assert_called_once_with("Invalid log message: %s"%raw)
 
        with mock.patch('ema_workbench.util.ema_logging._logger') as mocked_logger:   
            self.watcher.logger = mocked_logger
            raw = ['engine1DEBUG', 'test']
            self.watcher.log_message(raw)
            mocked_logger.error.assert_called_once_with("Invalid log message: %s"%raw)


class TestEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = parallel.Client(profile='default')
 
    @classmethod
    def tearDownClass(cls):
        pass
    
    @mock.patch('ema_workbench.em_framework.ema_parallel_ipython.get_engines_by_host')
    @mock.patch('ema_workbench.em_framework.ema_parallel_ipython.os')
    @mock.patch('ema_workbench.em_framework.ema_parallel_ipython.socket')
    def test_update_cwd_on_all_engines(self, mock_socket, mock_os, 
                                       mock_engines_by_host):
        mock_socket.gethostname.return_value = 'test host'
        
        mock_client = mock.create_autospec(parallel.Client)
        mock_client.ids = [0, 1] # pretend we have two engines
        mock_view = mock.create_autospec(parallel.client.view.View) #@ @UndefinedVariable
        mock_client.__getitem__.return_value = mock_view  
        
        mock_engines_by_host.return_value = {'test host':[0, 1]}
        
        mock_os.getcwd.return_value = '/test'
        
        # engines on same host
        ema.update_cwd_on_all_engines(mock_client)
        mock_view.apply.assert_called_with(mock_os.chdir, '/test')
        
        # engines on another host 
        mock_engines_by_host.return_value = {'other host':[0, 1]}
        self.assertRaises(NotImplementedError, 
                          ema.update_cwd_on_all_engines, mock_client)

    
    def test_get_engines_by_host(self):
        engines_by_host = ema.get_engines_by_host(self.client)
        self.assertEqual({ socket.gethostname(): [0,1]},engines_by_host)
    
    @mock.patch('ema_workbench.em_framework.ema_parallel_ipython.shutil')
    @mock.patch('ema_workbench.em_framework.ema_parallel_ipython.os')  
    def test_copy_wds_for_msis(self, mock_os, mock_shutil):
        mock_os.path.join.return_value = '.'
        
        mock_msi = mock.create_autospec(em_framework.ModelStructureInterface) # @UndefinedVariable
        mock_msi.name = 'test'
        
        kwargs = {}
        msis = {mock_msi.name: mock_msi}
        engine_id = 0
        engine = ema.Engine(engine_id, msis, kwargs)
        engine.root_dir = '/dir_name'        
        
        dirs_to_copy = ['/test']
        wd_by_msi = {'/test':[mock_msi.name]}
        engine.copy_wds_for_msis(dirs_to_copy, wd_by_msi)
        
        mock_os.path.basename.called_once_with(dirs_to_copy[0])
        mock_os.path.join.called_once_with('/dir_name' , dirs_to_copy[0])
        mock_shutil.copytree.assert_called_once_with('/test','.')
        self.assertEqual('.', mock_msi.working_directory)
        
    def test_init(self):
        kwargs = {}
        msis = {}
        engine_id = 0
        engine = ema.Engine(engine_id, msis, kwargs)
        
        self.assertEqual(engine_id, engine.engine_id)
        self.assertEqual(msis, engine.msis)
        self.assertEqual(kwargs, engine.runner.model_kwargs)
        self.assertEqual(experiment_runner.ExperimentRunner, type(engine.runner))
    
    @mock.patch('ema_workbench.em_framework.ema_parallel_ipython.os') 
    @mock.patch('ema_workbench.em_framework.ema_parallel_ipython.shutil') 
    def test_setup_wd(self, mock_shutil, mock_os):
        kwargs = {}
        msis = {}
        engine_id = 0
        engine = ema.Engine(engine_id, msis, kwargs)
        
        # directory does not exist
        mock_os.path.isdir.return_value = False
        mock_os.path.join.return_value = './test 0'

        wd = './test {}'
        engine.setup_working_directory(wd)
        mock_os.path.isdir.assert_called_once_with(wd.format(engine_id))
        mock_os.mkdir.assert_called_once_with(wd.format(engine_id))

        # directory already exists
        mock_os.path.isdir.return_value = True
        mock_os.path.join.return_value = './test 0'
        
        engine.setup_working_directory(wd)
        mock_shutil.rmtree.assert_called_once_with(wd.format(engine_id))
       
    def test_run_experiment(self):
        mock_msi = mock.create_autospec(em_framework.ModelStructureInterface) # @UndefinedVariable
        mock_msi.name = 'test'
        
        mock_runner = mock.create_autospec(experiment_runner.ExperimentRunner)
        
        kwargs = {}
        msis = {mock_msi.name: mock_msi}
        engine_id = 0
        engine = ema.Engine(engine_id, msis, kwargs)
        engine.runner = mock_runner
        
        experiment = {'a': 1}
        engine.run_experiment(experiment)
        
        mock_runner.run_experiment.assert_called_once_with(experiment)
        
        mock_runner.run_experiment.side_effect = EMAError
        self.assertRaises(EMAError, engine.run_experiment, experiment)
        
        
        mock_runner.run_experiment.side_effect = Exception
        self.assertRaises(EMAParallelError, engine.run_experiment, experiment)
        

class TestIpyParallelUtilFunctions(unittest.TestCase):

    @mock.patch('ema_workbench.em_framework.ema_parallel_ipython.setup_working_directories')
    def test_initialize_engines(self, mocked_setup_working_directories):
        mock_msi = mock.create_autospec(em_framework.ModelStructureInterface) # @UndefinedVariable
        mock_msi.name = 'test'
        msis = {mock_msi.name: mock_msi}
        
        mock_client = mock.create_autospec(parallel.Client)
        mock_client.ids = [0, 1] # pretend we have two engines
        mock_view = mock.create_autospec(parallel.client.view.View) #@ @UndefinedVariable
        mock_client.__getitem__.return_value = mock_view  
        
        ema.initialize_engines(mock_client, msis)
        
        mock_view.apply_sync.assert_any_call(ema._initialize_engine, 0, msis, {})
        mock_view.apply_sync.assert_any_call(ema._initialize_engine, 1, msis, {})
        
        mocked_setup_working_directories.assert_called_with(mock_client, msis)

    @mock.patch('ema_workbench.em_framework.ema_parallel_ipython.os')
    def test_setup_working_directories(self, mock_os):
        mock_msi = mock.create_autospec(em_framework.ModelStructureInterface) # @UndefinedVariable
        mock_msi.name = 'test'
        msis = {mock_msi.name: mock_msi}
        
        mock_client = mock.create_autospec(parallel.Client)
        mock_client.ids = [0, 1] # pretend we have two engines
        mock_view = mock.create_autospec(parallel.client.view.View) #@ @UndefinedVariable
        mock_client.__getitem__.return_value = mock_view  
        
        ema.setup_working_directories(mock_client, msis)
        #TODO assertion statements

if __name__ == "__main__":
    unittest.main()