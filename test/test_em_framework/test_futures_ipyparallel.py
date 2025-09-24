"""Tests for ipyparallelEvaluator.

Test code for ema_ipyparallel. The setup and teardown of the cluster is
taken from the ipyparallel test code with some minor adaptations


"""

# Created on Jul 28, 2015
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

import logging
import os
import time
import unittest
import unittest.mock as mock
import warnings
from subprocess import STDOUT, Popen

import ipyparallel
from ipyparallel import Client
from ipyparallel.cluster.launcher import (
    SIGKILL,
    LocalProcessLauncher,
    ProcessStateError,
    ipcontroller_cmd_argv,
    ipengine_cmd_argv,
)
from IPython.paths import get_ipython_dir
from jupyter_client.localinterfaces import localhost

import ema_workbench
from ema_workbench.em_framework import Model, experiment_runner
from ema_workbench.em_framework import futures_ipyparallel as ema
from ema_workbench.em_framework.futures_ipyparallel import LogWatcher
from ema_workbench.util import EMAError, EMAParallelError, ema_logging

launchers = []
blackhole = os.open(os.devnull, os.O_WRONLY)

warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*/IPython/.*")


# Launcher class
class MockProcessLauncher(LocalProcessLauncher):
    """subclass LocalProcessLauncher, to prevent extra sockets and threads being created on Windows."""

    def start(self):
        if self.state == "before":
            # Store stdout & stderr to show with failing tests.
            # This is defined in IPython.testing.iptest
            self.process = Popen(
                self.args,
                stdout=blackhole,
                stderr=STDOUT,
                env=os.environ,
                cwd=self.work_dir,
            )
            self.notify_start(self.process.pid)
            self.poll = self.process.poll
        else:
            s = f"The process was already started and has state: {self.state!r}"
            raise ProcessStateError(s)


def add_engines(n=1, profile="iptest", total=False):
    """Add a number of engines to a given profile.

    If total is True, then already running engines are counted, and only
    the additional engines necessary (if any) are started.
    """
    rc = Client(profile=profile)
    base = len(rc)

    if total:
        n = max(n - base, 0)

    eps = []
    for i in range(n):
        ep = MockProcessLauncher()
        ep.cmd_and_args = ipengine_cmd_argv + [
            f"--profile={profile}",
            "--InteractiveShell.colors=nocolor",
        ]
        ep.start()
        launchers.append(ep)
        eps.append(ep)
    tic = time.time()
    while len(rc) < base + n:
        if any(ep.poll() is not None for ep in eps):
            raise RuntimeError("A test engine failed to start.")
        elif time.time() - tic > 15:
            raise RuntimeError("Timeout waiting for engines to connect.")
        time.sleep(0.1)
    rc.close()
    return eps


def setUpModule():
    cluster_dir = os.path.join(get_ipython_dir(), "profile_default")
    engine_json = os.path.join(cluster_dir, "security", "ipcontroller-engine.json")
    client_json = os.path.join(cluster_dir, "security", "ipcontroller-client.json")
    for json in (engine_json, client_json):
        if os.path.exists(json):
            os.remove(json)

    cp = MockProcessLauncher()
    cp.cmd_and_args = ipcontroller_cmd_argv + ["--profile=default", "--log-level=20"]
    cp.start()
    launchers.append(cp)
    tic = time.time()
    while not os.path.exists(engine_json) or not os.path.exists(client_json):
        if cp.poll() is not None:
            raise RuntimeError(f"The test controller exited with status {cp.poll()}")
        elif time.time() - tic > 15:
            raise RuntimeError("Timeout waiting for the test controller to start.")
        time.sleep(0.1)

    add_engines(2, profile="default", total=True)


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
        if p.poll() is None:
            try:
                time.sleep(0.25)
            except KeyboardInterrupt:
                return
        if p.poll() is None:
            try:
                print("cleaning up test process...")
                p.signal(SIGKILL)
            except:
                print("couldn't shutdown process: ", p)


#     blackhole.close()


class TestEngineLoggerAdapter(unittest.TestCase):
    def tearDown(self):
        ema_logging._logger = None
        ema_logger = logging.getLogger(ema_logging.LOGGER_NAME)
        ema_logger.handlers = []

    def test_directly(self):
        with mock.patch("ema_workbench.util.ema_logging._logger") as mocked_logger:
            adapter = ema.EngingeLoggerAdapter(mocked_logger, ema.SUBTOPIC)
            self.assertEqual(mocked_logger, adapter.logger)
            self.assertEqual(ema.SUBTOPIC, adapter.topic)

            input_msg = "test"
            input_kwargs = {}
            msg, kwargs = adapter.process(input_msg, input_kwargs)

            self.assertEqual(f"{ema.SUBTOPIC}::{input_msg}", msg)
            self.assertEqual(input_kwargs, kwargs)

    @mock.patch("ema_workbench.em_framework.futures_ipyparallel.EngingeLoggerAdapter")
    @mock.patch("ema_workbench.em_framework.futures_ipyparallel.Application")
    def test_engine_logger(self, mocked_application, mocked_adapter):
        logger = ema_logging.get_rootlogger()
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

        #         ipyparallel.

        #         mock_engine_handler = mock.create_autospec(ipyparallel.log.EnginePUBHandler)
        mocked_logger.handlers = []  # [mock_engine_handler]

        mocked_application.instance.return_value = mocked_application
        mocked_application.log = mocked_logger

        ema.set_engine_logger()
        logger = ema_logging._logger
        #         self.assertTrue(type(logger) == ema.EngingeLoggerAdapter)
        mocked_logger.setLevel.assert_called_once_with(ema_logging.DEBUG)
        mocked_adapter.assert_called_with(mocked_logger, ema.SUBTOPIC)


#         mock_engine_handler.setLevel.assert_called_once_with(ema_logging.DEBUG)


#     def test_on_cluster(self):
#         client = ipyparallel.Client(profile='default')
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
        logger = ema_logging.get_rootlogger()
        mocked_logger = mock.Mock(spec=logger)
        mocked_logger.handlers = []
        ema_logging._logger = mocked_logger

        cls.client = mock.Mock(spec=ipyparallel.Client)
        cls.url = f"tcp://{localhost()}:20202"
        #         cls.watcher, cls.thread = ema.start_logwatcher()
        cls.watcher = LogWatcher()

    @classmethod
    def tearDownClass(cls):
        cls.watcher.stop()
        # TODO use some way to signal the thread to terminate
        # despite that it is a daemon thread

    def tearDown(self):
        self.client.clear(block=True)

    def test_init(self):
        self.assertEqual(self.url, self.watcher.url)

    def test_extract_level(self):
        level = "INFO"
        topic = ema.SUBTOPIC
        topic_str = f"engine.1.{level}.{topic}"
        extracted_level, extracted_topics = self.watcher._extract_level(topic_str)

        self.assertEqual(ema_logging.INFO, extracted_level)
        self.assertEqual(f"engine.1.{topic}", extracted_topics)

        topic = ema.SUBTOPIC
        topic_str = f"engine.1.{topic}"
        extracted_level, extracted_topics = self.watcher._extract_level(topic_str)

        self.assertEqual(ema_logging.INFO, extracted_level)
        self.assertEqual(f"engine.1.{topic}", extracted_topics)

    def test_log_message(self):
        # no subscription on level
        with mock.patch("logging.getLogger") as mocked:
            mocked_logger = mock.Mock(spec=logging.Logger)
            mocked.return_value = mocked_logger
            raw = [b"engine.1.INFO.EMA", b"test"]
            self.watcher.log_message(raw)
            mocked_logger.log.assert_called_once_with(
                ema_logging.INFO, "[engine.1] test"
            )

        with mock.patch("logging.getLogger") as mocked:
            mocked_logger = mock.Mock(spec=logging.Logger)
            mocked.return_value = mocked_logger
            raw = [b"engine.1.DEBUG.EMA", b"test"]
            self.watcher.log_message(raw)
            mocked_logger.log.assert_called_once_with(
                ema_logging.DEBUG, "[engine.1] test"
            )

        with mock.patch("logging.getLogger") as mocked:
            mocked_logger = mock.Mock(spec=logging.Logger)
            mocked.return_value = mocked_logger
            raw = [b"engine.1.DEBUG", b"test", b"more"]
            self.watcher.log_message(raw)
            raw = [r.decode("utf-8") for r in raw]
            mocked_logger.error.assert_called_once_with(f"Invalid log message: {raw}")

        with mock.patch("logging.getLogger") as mocked:
            mocked_logger = mock.Mock(spec=logging.Logger)
            mocked.return_value = mocked_logger
            raw = [b"engine1DEBUG", b"test"]
            self.watcher.log_message(raw)
            raw = [r.decode("utf-8") for r in raw]
            mocked_logger.error.assert_called_once_with(f"Invalid log message: {raw}")


class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = mock.Mock(spec=ipyparallel.Client)

    #         cls.client = ipyparallel.Client()

    @classmethod
    def tearDownClass(cls):
        pass

    @mock.patch("ema_workbench.em_framework.futures_ipyparallel.get_engines_by_host")
    @mock.patch("ema_workbench.em_framework.futures_ipyparallel.os")
    @mock.patch("ema_workbench.em_framework.futures_ipyparallel.socket")
    def test_update_cwd_on_all_engines(
        self, mock_socket, mock_os, mock_engines_by_host
    ):
        mock_socket.gethostname.return_value = "test host"

        mock_client = mock.create_autospec(ipyparallel.Client)
        mock_client.ids = [0, 1]  # pretend we have two engines
        mock_view = mock.create_autospec(
            ipyparallel.client.view.View
        )  # @ @UndefinedVariable
        mock_client.__getitem__.return_value = mock_view

        mock_engines_by_host.return_value = {"test host": [0, 1]}

        mock_os.getcwd.return_value = "/test"

        # engines on same host
        ema.update_cwd_on_all_engines(mock_client)
        mock_view.apply.assert_called_with(mock_os.chdir, "/test")

        # engines on another host
        mock_engines_by_host.return_value = {"other host": [0, 1]}
        self.assertRaises(
            NotImplementedError, ema.update_cwd_on_all_engines, mock_client
        )

    @mock.patch("ema_workbench.em_framework.futures_ipyparallel.socket")
    def test_get_engines_by_host(self, mock_socket):
        mock_client = mock.create_autospec(ipyparallel.Client)
        mock_client.ids = [0, 1]  # pretend we have two engines
        mock_view = mock.create_autospec(
            ipyparallel.client.view.View
        )  # @ @UndefinedVariable
        mock_client.__getitem__.return_value = mock_view
        mock_view.apply_sync.return_value = "test host"

        engines_by_host = ema.get_engines_by_host(mock_client)
        print(engines_by_host)
        self.assertEqual({"test host": [0, 1]}, engines_by_host)

    def test_init(self):
        msis = []
        engine_id = 0
        engine = ema.Engine(engine_id, msis, ".")

        self.assertEqual(engine_id, engine.engine_id)
        self.assertEqual(msis, engine.msis)
        self.assertEqual(experiment_runner.ExperimentRunner, type(engine.runner))

    def test_run_experiment(self):
        function = mock.Mock()
        mock_msi = Model("test", function)
        mock_runner = mock.create_autospec(experiment_runner.ExperimentRunner)

        msis = [mock_msi]
        engine_id = 0
        engine = ema.Engine(engine_id, msis, ".")
        engine.runner = mock_runner

        experiment = {"a": 1}
        engine.run_experiment(experiment)

        mock_runner.run_experiment.assert_called_once_with(experiment)

        mock_runner.run_experiment.side_effect = EMAError
        self.assertRaises(EMAError, engine.run_experiment, experiment)

        mock_runner.run_experiment.side_effect = Exception
        self.assertRaises(EMAParallelError, engine.run_experiment, experiment)


class TestIpyParallelUtilFunctions(unittest.TestCase):
    def test_initialize_engines(self):
        function = mock.Mock()
        mock_msi = Model("test", function)
        msis = {mock_msi.name: mock_msi}

        mock_client = mock.create_autospec(ipyparallel.Client)
        mock_client.ids = [0, 1]  # pretend we have two engines
        mock_view = mock.create_autospec(
            ipyparallel.client.view.View
        )  # @ @UndefinedVariable
        mock_client.__getitem__.return_value = mock_view

        cwd = "."
        ema.initialize_engines(mock_client, msis, cwd)

        mock_view.apply_sync.assert_any_call(ema._initialize_engine, 0, msis, cwd)
        mock_view.apply_sync.assert_any_call(ema._initialize_engine, 1, msis, cwd)


class TestIpyParallelEvaluator(unittest.TestCase):
    @mock.patch("ema_workbench.em_framework.futures_ipyparallel.set_engine_logger")
    @mock.patch("ema_workbench.em_framework.futures_ipyparallel.initialize_engines")
    @mock.patch("ema_workbench.em_framework.futures_ipyparallel.start_logwatcher")
    @mock.patch("ema_workbench.em_framework.evaluators.DefaultCallback")
    def test_ipyparallel_evaluator(
        self,
        mocked_callback,
        mocked_start,
        mocked_initialize,
        mocked_set,
    ):
        model = mock.Mock(spec=ema_workbench.Model)
        model.name = "test"

        mocked_generator = mock.Mock(
            "ema_workbench.em_framework.points.experiment_generator"
        )
        mocked_generator.return_value = [1]
        mocked_start.return_value = mocked_start, None

        client = mock.MagicMock(spec=ipyparallel.Client)
        lb_view = mock.Mock()
        lb_view.map.return_value = [(1, ({}, {}))]

        client.load_balanced_view.return_value = lb_view

        with ema.IpyparallelEvaluator(model, client) as evaluator:
            evaluator.evaluate_experiments(mocked_generator, mocked_callback)
            lb_view.map.assert_called_once()


if __name__ == "__main__":
    unittest.main()
    time.sleep(1)
