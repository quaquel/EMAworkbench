""" """

import threading
import unittest
import unittest.mock as mock

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
