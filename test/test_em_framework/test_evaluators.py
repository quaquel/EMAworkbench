""" """

import pytest

import ema_workbench
from ema_workbench import RealParameter, ScalarOutcome
from ema_workbench.em_framework import evaluators
from ema_workbench.em_framework.points import Experiment, Policy, Scenario
from ema_workbench.em_framework.experiment_runner import ExperimentRunner
from ema_workbench import EMAError

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

    with  evaluators.SequentialEvaluator(model) as evaluator:
        evaluator.evaluate_experiments(mocked_generator([], [], []), mocked_callback)

    for i, entry in enumerate(mocked_runner.run_experiment.call_args_list):
        assert entry.args[0].name == str(i)

    with pytest.raises(TypeError):
        evaluators.SequentialEvaluator([object()])


def test_perform_experiments(mocker):
    mocked_function = mocker.Mock(return_value={"c":1})
    model = ema_workbench.Model("test", function=mocked_function)
    model.uncertainties = [RealParameter("a", 0,1)]
    model.levers = [RealParameter("b", 0,1)]
    model.outcomes = [ScalarOutcome("c")]

    n_scenarios = 10
    n_policies = 2
    n_experiments = n_scenarios * n_policies

    mocked_callback_class = mocker.patch("ema_workbench.em_framework.evaluators.DefaultCallback")
    mocked_callback_instance = mocker.Mock()
    mocked_callback_class.return_value = mocked_callback_instance
    mocked_callback_instance.i = n_experiments # this is an implicit assert

    with  evaluators.SequentialEvaluator(model) as evaluator:
        evaluator.perform_experiments(n_scenarios, n_policies,)


    mocked_callback_class = mocker.patch("ema_workbench.em_framework.evaluators.DefaultCallback")
    mocked_callback_instance = mocker.Mock()
    mocked_callback_class.return_value = mocked_callback_instance
    mocked_callback_instance.i = n_scenarios # this is an implicit assert

    with  evaluators.SequentialEvaluator(model) as evaluator:
        evaluator.perform_experiments(n_scenarios, n_policies, combine="sample")

    # what to check?
    # fixme, all kinds of permutations of all the possible keyword arguments

    mocked_callback_class = mocker.patch("ema_workbench.em_framework.evaluators.DefaultCallback")
    mocked_callback_instance = mocker.Mock()
    mocked_callback_class.return_value = mocked_callback_instance
    mocked_callback_instance.i = n_experiments
    ret = evaluators.perform_experiments(model,n_scenarios, n_policies, return_callback=True, callback=mocked_callback_class)
    assert ret == mocked_callback_instance

    mocked_callback_class = mocker.patch("ema_workbench.em_framework.evaluators.DefaultCallback")
    mocked_callback_instance = mocker.Mock()
    mocked_callback_class.return_value = mocked_callback_instance
    mocked_callback_instance.i = 1
    with pytest.raises(EMAError):
        evaluators.perform_experiments(model, n_scenarios, n_policies, )

    with pytest.raises(EMAError):
        evaluators.perform_experiments(model )

    with pytest.raises(ValueError):
        evaluators.perform_experiments(model, n_scenarios, n_policies, combine="wrong value" )

def test_optimize(mocker):
    mocked_function = mocker.Mock(return_value={"d":1, "e":1})
    model = ema_workbench.Model("test", function=mocked_function)
    model.uncertainties = [RealParameter("a", 0,1)]
    model.levers = [RealParameter("b", 0,1),
                    RealParameter("c", 0,1)]
    model.outcomes = [ScalarOutcome("d", kind=ScalarOutcome.MAXIMIZE),
                      ScalarOutcome("e", kind=ScalarOutcome.MAXIMIZE)]

    nfe = 100

    with  evaluators.SequentialEvaluator(model) as evaluator:
        evaluator.optimize(nfe=nfe, searchover='levers', epsilons=[0.1, 0.1])

    # fixme what to mock
    # fixme what to assert


def test_robust_optimize(mocker):
    pass


