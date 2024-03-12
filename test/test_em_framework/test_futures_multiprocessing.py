"""


"""

import unittest.mock as mock
import unittest

import ema_workbench

from ema_workbench.em_framework import futures_multiprocessing

# Created on 14 Mar 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = []


class TestEvaluators(unittest.TestCase):
    @mock.patch("ema_workbench.em_framework.futures_multiprocessing.multiprocessing")
    @mock.patch("ema_workbench.em_framework.evaluators.DefaultCallback")
    @mock.patch("ema_workbench.em_framework.futures_multiprocessing.experiment_generator")
    @mock.patch("ema_workbench.em_framework.futures_multiprocessing.add_tasks")
    def test_multiprocessing_evaluator(
        self, mocked_add_task, mocked_generator, mocked_callback, mocked_multiprocessing
    ):
        model = mock.Mock(spec=ema_workbench.Model)
        model.name = "test"
        mocked_generator.return_value = [1]
        mocked_multiprocessing.cpu_count.return_value = 4

        with futures_multiprocessing.MultiprocessingEvaluator(model, 2) as evaluator:
            evaluator.evaluate_experiments(10, 10, mocked_callback)

            mocked_add_task.assert_called_once()
