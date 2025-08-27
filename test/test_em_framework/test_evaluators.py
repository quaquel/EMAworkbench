""" """

import unittest.mock as mock
import unittest

import ema_workbench
from ema_workbench.em_framework import evaluators

# Created on 14 Mar 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


class TestEvaluators(unittest.TestCase):
    @mock.patch("ema_workbench.em_framework.evaluators.DefaultCallback")
    @mock.patch("ema_workbench.em_framework.evaluators.experiment_generator")
    @mock.patch("ema_workbench.em_framework.evaluators.ExperimentRunner")
    def test_sequential_evalutor(self, mocked_runner, mocked_generator, mocked_callback):
        model = mock.Mock(spec=ema_workbench.Model)
        model.name = "test"
        mocked_generator.return_value = [1]
        mocked_runner.return_value = mocked_runner  # return the mock upon initialization
        mocked_runner.run_experiment.return_value = {}, {}

        evaluator = evaluators.SequentialEvaluator(model)
        evaluator.evaluate_experiments(10, 10, mocked_callback)

        mocked_runner.assert_called_once()  # test initialization of runner
        mocked_runner.run_experiment.assert_called_once_with(1)

    #         searchover
    #         union

    def test_perform_experiments(self):
        pass


if __name__ == "__main__":
    unittest.main()
