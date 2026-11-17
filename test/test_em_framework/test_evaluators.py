"""Tests for evaluators.py."""

import numpy as np
import pytest

import ema_workbench
from ema_workbench import EMAError, Model, RealParameter, Sample, ScalarOutcome
from ema_workbench.em_framework import LHSSampler, evaluators
from ema_workbench.em_framework.experiment_runner import ExperimentRunner
from ema_workbench.em_framework.points import Experiment, SampleCollection

# Created on 14 Mar 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


def test_sequential_evalutor(mocker):
    """Test SequentialEvaluator."""
    model = mocker.Mock(spec=ema_workbench.Model)
    model.name = "test"

    mocked_generator = mocker.patch(
        "ema_workbench.em_framework.evaluators.experiment_generator", autospec=True
    )
    mocked_generator.return_value = iter(
        Experiment(str(i), "test", Sample(), Sample(), i) for i in range(10)
    )

    mocked_runner = mocker.Mock(ExperimentRunner)
    mocker.patch(
        "ema_workbench.em_framework.evaluators.ExperimentRunner",
        mocker.MagicMock(return_value=mocked_runner),
    )
    mocked_runner.run_experiment.return_value = {}, {}

    mocked_callback = mocker.patch(
        "ema_workbench.em_framework.evaluators.DefaultCallback"
    )

    with evaluators.SequentialEvaluator(model) as evaluator:
        evaluator.evaluate_experiments(mocked_generator([], [], []), mocked_callback)

    for i, entry in enumerate(mocked_runner.run_experiment.call_args_list):
        assert entry.args[0].name == str(i)

    with pytest.raises(TypeError):
        evaluators.SequentialEvaluator([object()])


def test_perform_experiments(mocker):
    """Tests for perform_experiments."""
    mocked_function = mocker.Mock(return_value={"c": 1})
    model = ema_workbench.Model("test", function=mocked_function)
    model.uncertainties = [RealParameter("a", 0, 1)]
    model.levers = [RealParameter("b", 0, 1)]
    model.outcomes = [ScalarOutcome("c")]

    n_scenarios = 10
    n_policies = 2
    n_experiments = n_scenarios * n_policies

    mocked_callback_class = mocker.patch(
        "ema_workbench.em_framework.evaluators.DefaultCallback"
    )
    mocked_callback_instance = mocker.Mock()
    mocked_callback_class.return_value = mocked_callback_instance
    mocked_callback_instance.i = n_experiments  # this is an implicit assert

    with evaluators.SequentialEvaluator(model) as evaluator:
        evaluator.perform_experiments(
            n_scenarios,
            n_policies,
        )

    mocked_callback_class = mocker.patch(
        "ema_workbench.em_framework.evaluators.DefaultCallback"
    )
    mocked_callback_instance = mocker.Mock()
    mocked_callback_class.return_value = mocked_callback_instance
    mocked_callback_instance.i = n_scenarios  # this is an implicit assert

    with evaluators.SequentialEvaluator(model) as evaluator:
        evaluator.perform_experiments(n_scenarios, n_policies, combine="cycle")

    # what to check?
    # fixme, all kinds of permutations of all the possible keyword arguments

    mocked_callback_class = mocker.patch(
        "ema_workbench.em_framework.evaluators.DefaultCallback"
    )
    mocked_callback_instance = mocker.Mock()
    mocked_callback_class.return_value = mocked_callback_instance
    mocked_callback_instance.i = n_experiments
    ret = evaluators.perform_experiments(
        model,
        n_scenarios,
        n_policies,
        return_callback=True,
        callback=mocked_callback_class,
    )
    assert ret == mocked_callback_instance

    mocked_callback_class = mocker.patch(
        "ema_workbench.em_framework.evaluators.DefaultCallback"
    )
    mocked_callback_instance = mocker.Mock()
    mocked_callback_class.return_value = mocked_callback_instance
    mocked_callback_instance.i = 1
    with pytest.raises(EMAError):
        evaluators.perform_experiments(
            model,
            n_scenarios,
            n_policies,
        )

    with pytest.raises(EMAError):
        evaluators.perform_experiments(model)

    with pytest.raises(ValueError):
        evaluators.perform_experiments(
            model, n_scenarios, n_policies, combine="wrong value"
        )


def test_perform_experiments_mixing_samples_and_numbers():
    """Regression test: combine user specified samples with a number of samples.

    Previously, specifying scenarios as samples and policies as a number, or
    no scenarios and a number of policies, raised a TypeError.
    """

    def some_model(x=0.0, u=0.0):
        return {"y": x + u}

    model = Model("simple", function=some_model)
    model.uncertainties = [RealParameter("u", 0, 1)]
    model.levers = [RealParameter("x", 0, 1)]
    model.outcomes = [ScalarOutcome("y")]

    # reference scenario with sampled policies
    experiments, outcomes = evaluators.perform_experiments(
        model, scenarios=[Sample("s", u=0.5)], policies=3
    )
    assert experiments.shape[0] == 3
    np.testing.assert_allclose(outcomes["y"], experiments["x"] + 0.5)

    # no scenarios with sampled policies
    experiments, outcomes = evaluators.perform_experiments(
        model, scenarios=0, policies=3
    )
    assert experiments.shape[0] == 3
    assert "u" not in experiments.columns
    np.testing.assert_allclose(outcomes["y"], experiments["x"])

    # sampled scenarios with a reference policy
    experiments, outcomes = evaluators.perform_experiments(
        model, scenarios=3, policies=[Sample("p", x=0.5)]
    )
    np.testing.assert_allclose(outcomes["y"], experiments["u"] + 0.5)

    # sampled scenarios without policies
    experiments, outcomes = evaluators.perform_experiments(
        model, scenarios=3, policies=0
    )
    np.testing.assert_allclose(outcomes["y"], experiments["u"])


def test_optimize(mocker):
    """Tests for optimize."""
    mocked_function = mocker.Mock(return_value={"d": 1, "e": 1})
    model = ema_workbench.Model("test", function=mocked_function)
    model.uncertainties = [RealParameter("a", 0, 1)]
    model.levers = [RealParameter("b", 0, 1), RealParameter("c", 0, 1)]
    model.outcomes = [
        ScalarOutcome("d", kind=ScalarOutcome.MAXIMIZE),
        ScalarOutcome("e", kind=ScalarOutcome.MAXIMIZE),
    ]

    nfe = 100

    mocked_optimize = mocker.patch(
        "ema_workbench.em_framework.evaluators._optimize", autospec=True
    )
    with evaluators.SequentialEvaluator(model) as evaluator:
        evaluator.optimize(nfe=nfe, searchover="levers", epsilons=[0.1, 0.1])
    assert mocked_optimize.call_count == 1

    mocked_optimize.reset_mock()
    with evaluators.SequentialEvaluator(model) as evaluator:
        evaluator.optimize(nfe=nfe, searchover="uncertainties", epsilons=[0.1, 0.1])
    assert mocked_optimize.call_count == 1

    mocked_optimize.reset_mock()
    with evaluators.SequentialEvaluator(model) as evaluator:
        evaluator.optimize(
            nfe=nfe, searchover="uncertainties", epsilons=[0.1, 0.1], rng=[1, 2, 3]
        )
    assert mocked_optimize.call_count == 3

    mocked_optimize.reset_mock()
    evaluators.optimize(model, nfe=nfe, searchover="uncertainties", epsilons=[0.1, 0.1])
    assert mocked_optimize.call_count == 1

    with (
        pytest.raises(NotImplementedError),
        evaluators.SequentialEvaluator([model, model]) as evaluator,
    ):
        evaluator.optimize(nfe=nfe, searchover="uncertainties", epsilons=[0.1, 0.1])
    with (
        pytest.raises(ValueError),
        evaluators.SequentialEvaluator(model) as evaluator,
    ):
        evaluator.optimize(nfe=nfe, searchover="unknown value", epsilons=[0.1, 0.1])


@pytest.mark.parametrize("rng", ["some seed", b"some seed", 42, None])
def test_optimize_single_seed(tmp_path, rng):
    """Regression test: a str or bytes seed is a single seed.

    Previously, a str or bytes was treated as a collection of seeds, so the
    optimization was run once for every character.
    """
    import pandas as pd

    def some_model(x=0.0):
        return {"seed_test_y": x}

    model = Model("seedmodel", function=some_model)
    model.levers = [RealParameter("x", 0, 1)]
    model.outcomes = [ScalarOutcome("seed_test_y", kind=ScalarOutcome.MAXIMIZE)]
    scenarios = [Sample("s1"), Sample("s2")]
    robustness_functions = [
        ScalarOutcome(
            "seed_test_mean",
            kind=ScalarOutcome.MAXIMIZE,
            variable_name="seed_test_y",
            function=np.mean,
        )
    ]

    with evaluators.SequentialEvaluator(model) as evaluator:
        results = evaluator.optimize(
            nfe=20,
            searchover="levers",
            epsilons=[0.1],
            population_size=10,
            reference=Sample("reference"),
            filename="optimize.tar.gz",
            directory=tmp_path,
            rng=rng,
        )
        robust_results = evaluator.robust_optimize(
            robustness_functions,
            scenarios,
            nfe=20,
            epsilons=[0.1],
            population_size=10,
            filename="robust.tar.gz",
            directory=tmp_path,
            rng=rng,
        )

    for result in (results, robust_results):
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "optimize.tar.gz",
        "robust.tar.gz",
    ]


def test_optimize_multiple_seeds(tmp_path):
    """A collection of seeds results in one optimization for each seed."""

    def some_model(x=0.0):
        return {"seed_test_y": x}

    model = Model("seedmodel", function=some_model)
    model.levers = [RealParameter("x", 0, 1)]
    model.outcomes = [ScalarOutcome("seed_test_y", kind=ScalarOutcome.MAXIMIZE)]

    with evaluators.SequentialEvaluator(model) as evaluator:
        results = evaluator.optimize(
            nfe=20,
            searchover="levers",
            epsilons=[0.1],
            population_size=10,
            reference=Sample("reference"),
            filename="optimize.tar.gz",
            directory=tmp_path,
            rng=[1, "a"],
        )
    assert isinstance(results, list)
    assert len(results) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "1_optimize.tar.gz",
        "a_optimize.tar.gz",
    ]


def test_robust_optimize(mocker):
    """Tests for robust_optimize."""
    mocked_function = mocker.Mock(return_value={"d": 1, "e": 1})
    model = ema_workbench.Model("test", function=mocked_function)
    model.uncertainties = [RealParameter("a", 0, 1)]
    model.levers = [RealParameter("b", 0, 1), RealParameter("c", 0, 1)]
    model.outcomes = [
        ScalarOutcome("d", kind=ScalarOutcome.MAXIMIZE),
        ScalarOutcome("e", kind=ScalarOutcome.MAXIMIZE),
    ]

    robustness_functions = [
        ScalarOutcome(
            "robustness_d",
            kind=ScalarOutcome.MAXIMIZE,
            variable_name="d",
            function=mocker.Mock(return_value={"r_d": 0.5}),
        ),
        ScalarOutcome(
            "robustness_e",
            kind=ScalarOutcome.MAXIMIZE,
            variable_name="e",
            function=mocker.Mock(return_value={"r_e": 0.5}),
        ),
    ]
    scenarios = 10

    nfe = 100

    mocked_optimize = mocker.patch(
        "ema_workbench.em_framework.evaluators._optimize", autospec=True
    )
    with evaluators.SequentialEvaluator(model) as evaluator:
        evaluator.robust_optimize(
            robustness_functions, scenarios=scenarios, nfe=nfe, epsilons=[0.1, 0.1]
        )
    assert mocked_optimize.call_count == 1

    mocked_optimize.reset_mock()
    with evaluators.SequentialEvaluator(model) as evaluator:
        evaluator.robust_optimize(
            robustness_functions,
            scenarios=scenarios,
            nfe=nfe,
            epsilons=[0.1, 0.1],
            rng=[1, 2, 3],
        )
    assert mocked_optimize.call_count == 3


def test_perform_experiments_samples_with_different_parameters():
    """Regression test: all parameters in the samples end up in the experiments.

    Previously, only the parameters in the first sample were used for the
    columns of the experiments, so values in later samples were used by the
    model but not recorded.
    """
    from ema_workbench import IntegerParameter

    def some_model(x=-1.0, z=-1.0, i=0, u=0.0):
        return {"y": x + z}

    model = Model("simple", function=some_model)
    model.uncertainties = [RealParameter("u", 0, 1)]
    model.levers = [RealParameter("x", 0, 1), RealParameter("z", 0, 1)]
    model.outcomes = [ScalarOutcome("y")]

    policies = [Sample("a", x=0.5), Sample("b", x=0.2, z=0.3)]
    experiments, outcomes = evaluators.perform_experiments(
        model, scenarios=[Sample("s", u=0.5)], policies=policies
    )

    assert "z" in experiments.columns
    np.testing.assert_allclose(experiments["x"], [0.5, 0.2])
    # no value and no default: the model uses its own default
    assert np.isnan(experiments["z"][0])
    assert experiments["z"][1] == 0.3
    np.testing.assert_allclose(outcomes["y"], [-0.5, 0.5])

    # with a default on the parameter, the default is used and stored
    model = Model("simple", function=some_model)
    model.uncertainties = [RealParameter("u", 0, 1)]
    model.levers = [RealParameter("x", 0, 1), RealParameter("z", 0, 1, default=0.7)]
    model.outcomes = [ScalarOutcome("y")]

    experiments, outcomes = evaluators.perform_experiments(
        model, scenarios=[Sample("s", u=0.5)], policies=policies
    )
    np.testing.assert_allclose(experiments["z"], [0.7, 0.3])
    np.testing.assert_allclose(outcomes["y"], [1.2, 0.5])

    # an integer parameter cannot be missing without a default
    model = Model("simple", function=some_model)
    model.uncertainties = [RealParameter("u", 0, 1)]
    model.levers = [RealParameter("x", 0, 1), IntegerParameter("i", 0, 5)]
    model.outcomes = [ScalarOutcome("y")]

    with pytest.raises(EMAError):
        evaluators.perform_experiments(
            model,
            scenarios=[Sample("s", u=0.5)],
            policies=[Sample("a", x=0.5), Sample("b", x=0.2, i=3)],
        )


def test_setup(mocker):
    """Test for _setup helper function."""
    my_model = Model("some_model", function=mocker.Mock(return_value={"d": 1, "e": 1}))
    my_model.uncertainties = [RealParameter("a", 0, 1)]

    samples, parameters, n_samples = evaluators._setup(
        None,
        None,
        {},
        [
            my_model,
        ],
        union=True,
        parameter_type="uncertainties",
    )
    assert n_samples == 1
    for p in parameters:
        assert p in my_model.uncertainties

    n = 10
    samples, parameters, n_samples = evaluators._setup(
        n,
        LHSSampler(),
        None,
        [
            my_model,
        ],
        union=True,
        parameter_type="uncertainties",
    )
    assert n_samples == n
    for p in parameters:
        assert p.name in my_model.uncertainties

    samples = [
        Sample(1, a=0.5),
    ]
    samples, parameters, n_samples = evaluators._setup(
        samples,
        LHSSampler(),
        None,
        [
            my_model,
        ],
        union=True,
        parameter_type="uncertainties",
    )
    assert n_samples == len(samples)
    for p in parameters:
        assert p.name in my_model.uncertainties

    samples = Sample(1, a=0.5)
    samples, parameters, n_samples = evaluators._setup(
        samples,
        LHSSampler(),
        None,
        [
            my_model,
        ],
        union=True,
        parameter_type="uncertainties",
    )
    assert n_samples == 1
    for p in parameters:
        assert p.name in my_model.uncertainties

    rng = np.random.default_rng(42)
    n = 100
    samples = SampleCollection(rng.random((n, 1)), my_model.uncertainties)
    samples, parameters, n_samples = evaluators._setup(
        samples,
        None,
        None,
        [
            my_model,
        ],
        union=True,
        parameter_type="uncertainties",
    )
    assert n_samples == n


def test_perform_experiments_rejects_policies_with_same_name():
    """Regression test: the name of a policy is its unique identifier.

    A model is only initialized again if the name of the policy changes.
    Previously, two different policies with the same name were accepted, and
    the second one was silently run with the levers of the first one.
    """

    def some_model(x=0.0, u=0.0):
        return {"y": x + u}

    model = Model("simple", function=some_model)
    model.uncertainties = [RealParameter("u", 0, 1)]
    model.levers = [RealParameter("x", 0, 1)]
    model.outcomes = [ScalarOutcome("y")]

    scenarios = [Sample("s", u=0.0)]

    with pytest.raises(EMAError, match="same"):
        evaluators.perform_experiments(
            model,
            scenarios=scenarios,
            policies=[
                Sample("same", x=0.2),
                Sample("other", x=0.3),
                Sample("same", x=0.4),
            ],
        )

    # unique names, explicit or generated, are fine
    _, outcomes = evaluators.perform_experiments(
        model,
        scenarios=scenarios,
        policies=[Sample("a", x=0.2), Sample("b", x=0.4), Sample(x=0.6)],
    )
    np.testing.assert_allclose(outcomes["y"], [0.2, 0.4, 0.6])

    experiments, _ = evaluators.perform_experiments(model, scenarios=2, policies=3)
    assert experiments["policy"].nunique() == 3

    # scenarios are not affected
    _, outcomes = evaluators.perform_experiments(
        model,
        scenarios=[Sample("s", u=0.1), Sample("s", u=0.2)],
        policies=[Sample("a", x=0.2)],
    )
    np.testing.assert_allclose(outcomes["y"], [0.3, 0.4])


def test_model_is_initialized_once_for_a_policy():
    """A model is not initialized again as long as the policy has the same name.

    Initializing a model can be expensive, so this should also hold across
    calls to perform_experiments.
    """
    initialized = []

    class CountingModel(Model):
        def model_init(self, policy):
            initialized.append(policy.name)
            super().model_init(policy)

    def some_model(x=0.0, u=0.0):
        return {"y": x + u}

    model = CountingModel("counting", function=some_model)
    model.uncertainties = [RealParameter("u", 0, 1)]
    model.levers = [RealParameter("x", 0, 1)]
    model.outcomes = [ScalarOutcome("y")]

    policies = [Sample("a", x=0.2), Sample("b", x=0.4)]
    evaluators.perform_experiments(model, scenarios=5, policies=policies)
    assert initialized == ["a", "b"]

    evaluators.perform_experiments(model, scenarios=5, policies=[policies[1]])
    evaluators.perform_experiments(model, scenarios=5, policies=[policies[1]])
    assert initialized == ["a", "b"]


def test_perform_experiments_no_uncertainties_multiple_policies():
    """Regression test: model without uncertainties and multiple policies.

    Previously, lever values leaked into the shared (empty) scenario, causing
    an EMAError on the second policy.
    """

    def some_model(x=0.0):
        return {"y": x}

    model = Model("simple", function=some_model)
    model.levers = [RealParameter("x", 0, 1)]
    model.outcomes = [ScalarOutcome("y")]

    policies = [Sample(x=0.1), Sample(x=0.2), Sample(x=0.3)]
    experiments, outcomes = evaluators.perform_experiments(
        model, scenarios=1, policies=policies
    )

    np.testing.assert_allclose(outcomes["y"], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(experiments["x"], [0.1, 0.2, 0.3])


def test_perform_experiments_does_not_modify_user_samples():
    """Regression test: user-supplied scenarios must not get lever values."""

    def some_model(x=0.0, u=0.0):
        return {"y": x + u}

    model = Model("simple", function=some_model)
    model.uncertainties = [RealParameter("u", 0, 1)]
    model.levers = [RealParameter("x", 0, 1)]
    model.outcomes = [ScalarOutcome("y")]

    scenarios = [Sample(u=0.5), Sample(u=0.7)]
    policies = [Sample(x=0.1), Sample(x=0.2)]
    _, outcomes = evaluators.perform_experiments(
        model, scenarios=scenarios, policies=policies
    )

    for scenario in scenarios:
        assert set(scenario.keys()) == {"u"}
    for policy in policies:
        assert set(policy.keys()) == {"x"}
    np.testing.assert_allclose(outcomes["y"], [0.6, 0.8, 0.7, 0.9])
