""" """

import multiprocessing
import threading
import unittest
import unittest.mock as mock

import pytest

import ema_workbench
from ema_workbench import EMAError, Model, RealParameter, ScalarOutcome
from ema_workbench.em_framework import futures_multiprocessing

# Created on 14 Mar 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = []


class TestEvaluators(unittest.TestCase):
    @mock.patch("ema_workbench.em_framework.futures_multiprocessing.multiprocessing")
    @mock.patch("ema_workbench.em_framework.evaluators.DefaultCallback")
    @mock.patch("ema_workbench.em_framework.futures_multiprocessing.add_tasks")
    def test_multiprocessing_evaluator(
        self, mocked_add_task, mocked_callback, mocked_multiprocessing
    ):
        model = mock.Mock(spec=ema_workbench.Model)
        model.name = "test"
        mocked_generator = mock.Mock(
            "ema_workbench.em_framework.points.experiment_generator"
        )
        mocked_generator.return_value = [1]
        mocked_multiprocessing.cpu_count.return_value = 4

        with futures_multiprocessing.MultiprocessingEvaluator(model, 2) as evaluator:
            evaluator.evaluate_experiments(mocked_generator, mocked_callback)

            mocked_add_task.assert_called_once()


def failing_model(x=0.0):
    """Model that fails for part of the input space."""
    if x > 0.5:
        raise KeyError("model failed")
    return {"y": x}


def _run_failing_model(n_experiments, result):
    model = Model("failing", function=failing_model)
    model.uncertainties = [RealParameter("x", 0, 1)]
    model.outcomes = [ScalarOutcome("y")]

    try:
        with futures_multiprocessing.MultiprocessingEvaluator(
            model, n_processes=2
        ) as evaluator:
            evaluator.perform_experiments(
                n_experiments, uncertainty_sampling_kwargs={"rng": 42}
            )
    except BaseException as e:
        result["exception"] = e
    else:
        result["exception"] = None


def test_multiprocessing_model_error_does_not_hang():
    """Regression test: an exception in a model must raise, not hang.

    Previously, the EMAError raised in the worker killed the worker process
    because it derived from BaseException, so the result never arrived. With
    many experiments, the feeder thread also blocked forever on a full queue.
    """
    # more experiments than fit on the results queue (5 * n_processes)
    for n_experiments in (6, 60):
        result = {}
        thread = threading.Thread(
            target=_run_failing_model, args=(n_experiments, result), daemon=True
        )
        thread.start()
        thread.join(timeout=60)

        assert not thread.is_alive(), "evaluator hangs after an exception in a model"
        assert isinstance(result["exception"], EMAError)
        assert "model failed" in str(result["exception"])


def test_emaerror_is_exception():
    """EMAError must be catchable with except Exception."""
    assert issubclass(EMAError, Exception)


_LOGGING_MODEL = """
from ema_workbench.util import ema_logging

_logger = ema_logging.get_module_logger(__name__)


def model(x=0.0):
    _logger.info("info from worker %d", int(x * 100))
    _logger.debug("debug from worker")
    return {"y": x}
"""

_LOGGING_SCRIPT = """
import multiprocessing
import sys

from ema_workbench import Model, MultiprocessingEvaluator, RealParameter, ScalarOutcome
from ema_workbench.util import ema_logging

import logging_model

if __name__ == "__main__":
    multiprocessing.set_start_method(sys.argv[1])
    ema_logging.log_to_stderr(ema_logging.INFO)
    model = Model("logging", function=logging_model.model)
    model.uncertainties = [RealParameter("x", 0, 1)]
    model.outcomes = [ScalarOutcome("y")]
    with MultiprocessingEvaluator(model, n_processes=2) as evaluator:
        evaluator.perform_experiments(6)
"""


@pytest.mark.parametrize("start_method", ["fork", "spawn"])
def test_worker_log_messages_reach_main_process(tmp_path, start_method):
    """Regression test: log messages from the workers reach the main process.

    Previously, the handler that sends log records to the main process was
    attached to a logger that no module logger propagates to. With fork, the
    messages were only shown because the workers inherited the handlers of
    the main process, and with spawn they were lost.
    """
    import os  # noqa: PLC0415
    import re  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"start method {start_method} not available")

    (tmp_path / "logging_model.py").write_text(_LOGGING_MODEL)
    (tmp_path / "logging_script.py").write_text(_LOGGING_SCRIPT)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), *sys.path])}

    result = subprocess.run(
        [sys.executable, "logging_script.py", start_method],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

    # each message arrives once, formatted once, with the name of the worker
    messages = re.findall(r"\[([\w-]+)/INFO\] info from worker \d+", result.stderr)
    assert len(messages) == 6, result.stderr
    assert all("PoolWorker" in process for process in messages)
    assert "%(" not in result.stderr

    # the log level of the main process applies to the workers
    assert "debug from worker" not in result.stderr
