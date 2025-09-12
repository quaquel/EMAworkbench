""" """

import pytest

import ema_workbench
from ema_workbench.em_framework import evaluators
from ema_workbench.em_framework.points import Experiment, Policy, Scenario
from ema_workbench.em_framework.experiment_runner import ExperimentRunner

# Created on 14 Mar 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


def test_sequential_evalutor(mocker):
    model = mocker.Mock(spec=ema_workbench.Model)
    model.name = "test"

    mocked_generator = mocker.patch("ema_workbench.em_framework.evaluators.experiment_generator", autospec=True)
    mocked_generator.return_value = iter(Experiment(str(i), "test", Policy(), Scenario(), i ) for i in range(10))

    mocked_runner = mocker.Mock(ExperimentRunner)
    mocker.patch("ema_workbench.em_framework.evaluators.ExperimentRunner", mocker.MagicMock(return_value=mocked_runner))
    mocked_runner.run_experiment.return_value = {}, {}

    mocked_callback = mocker.patch("ema_workbench.em_framework.evaluators.DefaultCallback")

    evaluator = evaluators.SequentialEvaluator(model)
    evaluator.evaluate_experiments(mocked_generator([], [], []), mocked_callback)

    for i, entry in enumerate(mocked_runner.run_experiment.call_args_list):
        assert entry.args[0].name == str(i)


    def test_perform_experiments(self):
        pass



