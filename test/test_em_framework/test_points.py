"""Tests for em_framework/em_framework/points.py."""

import numpy as np
import pytest

from ema_workbench.em_framework.optimization import Problem
from ema_workbench.em_framework.parameters import (
    BooleanParameter,
    CategoricalParameter,
    IntegerParameter,
    RealParameter,
)
from ema_workbench.em_framework.points import (
    Sample,
    SampleCollection,
    experiment_generator,
    from_experiments,
    sample_generator,
)
from ema_workbench.em_framework.util import NamedObject
from test import utilities


def test_experiment_generator():
    """Tests for experiment_generator."""
    scenarios = [NamedObject("scen_1"), NamedObject("scen_2")]
    model = [NamedObject("model")]
    policies = [NamedObject("1"), NamedObject("2"), NamedObject("3")]

    experiments = experiment_generator(
        scenarios, model, policies, combine="full_factorial"
    )
    experiments = list(experiments)

    assert len(experiments) == 6, "wrong number of experiments for factorial"

    experiments = experiment_generator(scenarios, model, policies, combine="cycle")
    experiments = list(experiments)
    assert len(experiments) == 3, "wrong number of experiments for zipover"

    experiments = experiment_generator(scenarios, model, policies, combine="sample")
    experiments = list(experiments)
    assert len(experiments) == 3, "wrong number of experiments for sample"

    rng = np.random.default_rng(42)
    scenarios = SampleCollection(
        rng.uniform(size=(10, 3)), [RealParameter(entry, 0, 1) for entry in "abc"]
    )
    policies = SampleCollection(
        rng.uniform(size=(5, 3)), [RealParameter(entry, 0, 1) for entry in "def"]
    )

    model = [NamedObject("model")]
    experiments = experiment_generator(model, scenarios, policies, combine="sample")
    experiments = list(experiments)
    assert len(experiments) == 10, "wrong number of experiments for sample"

    with pytest.raises(ValueError):
        experiments = experiment_generator(scenarios, model, policies, combine="adf")
        _ = list(experiments)


def test_sample_generator():
    """Tests for sample_generator."""
    rng = np.random.default_rng(42)

    samples = rng.uniform(size=(10, 5))
    samples[:, 1] = np.floor(samples[:, 1] * 10)
    samples[:, 2] = np.floor(samples[:, 2] * 2)
    samples[:, 3] = np.round(samples[:, 3])
    samples[:, 3] = np.round(samples[:, 4])
    parameters = [
        RealParameter("a", 0, 1),
        IntegerParameter("b", 0, 10),
        CategoricalParameter("c", ["a", "b", "c"]),
        BooleanParameter("d", shape=(2,)),
    ]

    generator = sample_generator(samples, parameters)

    for i, sample in enumerate(generator):
        a, b, c, d, e = samples[i]

        assert sample["a"] == a
        assert sample["b"] == int(b)
        assert sample["c"] == parameters[2].cat_for_index(int(c)).value
        assert np.all(sample["d"] == np.asarray([int(d), int(e)], dtype=bool))


def test_parameters_with_shape():
    """Regression test: integer and categorical parameters with a shape.

    Previously, the values for these parameters were a list rather than an
    array, which resulted in an AttributeError when storing the experiment.
    """
    from ema_workbench import Model, ScalarOutcome, perform_experiments
    from ema_workbench.em_framework.points import flatten_sample
    from ema_workbench.em_framework.samplers import LHSSampler

    parameters = [
        RealParameter("a", 0, 1, shape=(2,)),
        IntegerParameter("b", 0, 5, shape=(3,)),
        CategoricalParameter("c", ["x", "y", "z"], shape=(2,)),
    ]

    samples = LHSSampler().generate_samples(parameters, 5, rng=42)
    for sample in samples:
        assert isinstance(sample["a"], np.ndarray)
        assert isinstance(sample["b"], np.ndarray)
        assert isinstance(sample["c"], np.ndarray)
        assert sample["a"].shape == (2,)
        assert sample["b"].shape == (3,)
        assert sample["c"].shape == (2,)
        assert np.issubdtype(sample["b"].dtype, np.integer)
        assert set(sample["c"]) <= {"x", "y", "z"}
        assert len(flatten_sample(sample, parameters)) == 7

    received = []

    def some_model(a=None, b=None, c=None):
        received.append((a, b, c))
        return {"y": float(np.sum(a) + np.sum(b))}

    model = Model("simple", function=some_model)
    model.uncertainties = parameters
    model.outcomes = [ScalarOutcome("y")]

    experiments, outcomes = perform_experiments(model, scenarios=4)

    assert list(experiments.columns[:7]) == [
        "a_0",
        "a_1",
        "b_0",
        "b_1",
        "b_2",
        "c_0",
        "c_1",
    ]
    expected = (
        experiments[["a_0", "a_1", "b_0", "b_1", "b_2"]].astype(float).sum(axis=1)
    )
    np.testing.assert_allclose(outcomes["y"], expected)
    for i, (_, _, c) in enumerate(received):
        assert list(c) == [experiments.loc[i, "c_0"], experiments.loc[i, "c_1"]]


def test_sample_collection():
    """Tests for SampleCollection."""
    rng = np.random.default_rng(42)

    samples = rng.uniform(size=(10, 3))
    parameters = [RealParameter(entry, 0, 1) for entry in "abc"]
    samples_collection = SampleCollection(samples, parameters)

    # basic intit
    assert samples_collection.n == samples.shape[0]

    # iteration
    it = iter(samples_collection)
    for i, entry in enumerate(it):
        values = np.array([entry[k] for k in "abc"])
        assert np.all(values == samples[i])

    # combine
    ## full factorial
    samples1 = np.array([0, 1]).reshape(2, 1)
    samples2 = np.array([0, 1]).reshape(2, 1)

    it1 = SampleCollection(samples1, [RealParameter("a", 0, 1)])
    it2 = SampleCollection(samples2, [RealParameter("b", 0, 1)])

    it3 = it1.combine(it2, "full_factorial")
    samples = it3.samples

    assert np.all(samples == np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))

    ## equal length, so no need to cycle
    samples1 = np.array([0, 1]).reshape(2, 1)
    samples2 = np.array([0, 1]).reshape(2, 1)

    it1 = SampleCollection(samples1, [RealParameter("a", 0, 1)])
    it2 = SampleCollection(samples2, [RealParameter("b", 0, 1)])

    it3 = it1.combine(it2, "cycle")
    samples = it3.samples
    assert np.all(samples == np.array([[0, 0], [1, 1]]))

    ## cycle
    samples1 = np.array([0, 1, 2]).reshape(3, 1)
    samples2 = np.array([0, 1]).reshape(2, 1)

    it1 = SampleCollection(samples1, [RealParameter("a", 0, 1)])
    it2 = SampleCollection(samples2, [RealParameter("b", 0, 1)])

    it3 = it1.combine(it2, "cycle")
    samples = it3.samples
    assert np.all(samples == np.array([[0, 0], [1, 1], [2, 0]]))

    ## sample
    it3 = it1.combine(it2, "sample", rng=42)
    samples = it3.samples
    assert np.all(
        samples == np.array([[0, 0], [1, 1], [2, 1]])
    )  # return has been hard coded for rng 42

    with pytest.raises(ValueError):
        it1.combine(it2, "wrong value")

    with pytest.raises(ValueError):
        it1.combine(
            SampleCollection(samples2, [RealParameter("a", 0, 1)]), "wrong value"
        )

    it1 = SampleCollection(samples1, [RealParameter("a", 0, 1)])
    it2 = SampleCollection(samples2, [RealParameter("a", 0, 1)])
    combined_samples = np.vstack((samples1, samples2))

    it3 = it1 + it2
    for i, sample in enumerate(it3):
        np.all(np.asarray(sample.values) == combined_samples[i])

    with pytest.raises(ValueError):
        it1 = SampleCollection(samples1, [RealParameter("a", 0, 1)])
        it2 = SampleCollection(samples2, [RealParameter("b", 0, 1)])
        it3 = it1 + it2


def test_sample_collection_concat_parameter_order():
    """Regression test: concat with parameters in a different order.

    Previously, the samples were stacked column by column, so values ended
    up under the wrong parameter.
    """
    a = RealParameter("a", 0, 1)
    b = RealParameter("b", 10, 11, shape=(2,))
    c = IntegerParameter("c", 100, 110)

    first = SampleCollection(np.array([[0.1, 10.1, 10.2, 101]]), [a, b, c])
    second = SampleCollection(np.array([[102, 10.3, 10.4, 0.2]]), [c, b, a])

    combined = first.concat(second)
    assert [p.name for p in combined.parameters] == ["a", "b", "c"]
    np.testing.assert_allclose(
        combined.samples, [[0.1, 10.1, 10.2, 101], [0.2, 10.3, 10.4, 102]]
    )
    samples = list(combined)
    assert samples[1]["a"] == 0.2
    assert samples[1]["c"] == 102
    np.testing.assert_allclose(samples[1]["b"], [10.3, 10.4])

    # parameters missing in either of the two collections
    with pytest.raises(ValueError):
        first.concat(SampleCollection(np.array([[0.1]]), [a]))
    with pytest.raises(ValueError):
        SampleCollection(np.array([[0.1]]), [a]).concat(first)


def test_sample_collection_getitem():
    """Test SampleCollection.__getitem__."""
    rng = np.random.default_rng(42)

    samples = rng.uniform(size=(10, 3))
    parameters = [RealParameter(entry, 0, 1) for entry in "abc"]
    sample_collection = SampleCollection(samples, parameters)

    sample = sample_collection[0]
    assert isinstance(sample, Sample)
    assert np.all(np.asarray(list(sample.values())) == samples[0, :])

    sub_samples = sample_collection[0:2]
    for i, sample in enumerate(sub_samples):
        assert isinstance(sample, Sample)
        assert np.all(np.asarray(list(sample.values())) == samples[i, :])

    with pytest.raises(TypeError):
        sample_collection[0:2, :]

    with pytest.raises(TypeError):
        sample_collection[0.5]


def test_from_experiments():
    """Test from_experiments."""
    experiments, _ = utilities.load_scarcity_data()

    samples = from_experiments(experiments)

    assert len(samples) == experiments.shape[0]

    for sample, (_, row) in zip(samples[0:10], experiments.iloc[0:10, :].iterrows()):
        for k, v in sample.items():
            assert row[k] == v


def test_from_experiments_with_parameters():
    """Shaped parameters are turned back into arrays."""
    import pandas as pd

    parameters = [
        RealParameter("a", 0, 1, shape=(2, 2)),
        CategoricalParameter("c", ["lo", "hi"], shape=(2,)),
        IntegerParameter("k", 0, 3),
    ]
    experiments = pd.DataFrame(
        {
            "a_0": [0.1],
            "a_1": [0.2],
            "a_2": [0.3],
            "a_3": [0.4],
            "c_0": ["lo"],
            "c_1": ["hi"],
            "k": [2],
            "y": [1.0],
            "scenario": [0],
            "policy": ["p"],
            "model": ["m"],
        }
    )
    (sample,) = from_experiments(experiments, parameters=parameters)
    assert set(sample) == {"a", "c", "k"}
    np.testing.assert_allclose(sample["a"], [[0.1, 0.2], [0.3, 0.4]])
    assert list(sample["c"]) == ["lo", "hi"]
    assert sample["k"] == 2

    with pytest.raises(ValueError, match="a_3"):
        from_experiments(experiments.drop(columns="a_3"), parameters=parameters)


def test_sample_platypus():
    """Tests for converting samples to platypus solutions and vice versa."""
    parameters = [
        RealParameter("a", 0, 1),
        IntegerParameter("b", 1, 4),
        CategoricalParameter("c", ["a", "b", "c"]),
        CategoricalParameter("d", ["a", "b", "c"], shape=(2,)),
    ]

    problem = Problem("levers", parameters, [], [])

    sample = Sample(
        a=0.5,
        b=2,
        c="b",
        d=np.asarray(
            [
                "a",
                "b",
            ]
        ),
    )
    solution = sample._to_platypus_solution(problem)

    new_sample = Sample._from_platypus_solution(solution)

    assert sample["a"] == new_sample["a"]
    assert sample["b"] == new_sample["b"]
    assert sample["c"] == new_sample["c"]
    assert np.all(sample["d"] == new_sample["d"])


@pytest.mark.parametrize("sampler", ["MC", "LHS", "FF", "SOBOL"])
def test_multidimensional_shape_reaches_model(sampler):
    """Regression test: a parameter with a 2d shape was passed flattened."""
    from ema_workbench import Model, ScalarOutcome, perform_experiments
    from ema_workbench.em_framework.evaluators import Samplers

    seen = []

    def function(a=None, c=None, k=None):
        seen.append((np.shape(a), np.shape(c), np.shape(k)))
        return {"y": 0.0}

    model = Model("model", function=function)
    model.uncertainties = [
        RealParameter("a", 0, 1, shape=(2, 3)),
        CategoricalParameter("c", ["x", "y"], shape=(2, 1)),
        IntegerParameter("k", 0, 5, shape=(1, 2)),
    ]
    model.outcomes = [ScalarOutcome("y")]

    kwargs = {"resolution": [2]} if sampler == "FF" else {}
    n = 2 if sampler == "FF" else 4
    experiments, _ = perform_experiments(
        model,
        n,
        uncertainty_sampling=Samplers[sampler],
        uncertainty_sampling_kwargs=kwargs or None,
    )
    assert set(seen) == {((2, 3), (2, 1), (1, 2))}
    # the experiments still have one column for each element
    assert [f"a_{i}" for i in range(6)] == [
        c for c in experiments.columns if c.startswith("a_")
    ]


def test_multidimensional_shape_from_platypus_solution():
    """Regression test: a decision variable with a 2d shape was flattened."""
    from ema_workbench import ScalarOutcome

    parameters = [
        RealParameter("a", 0, 1, shape=(2, 2)),
        CategoricalParameter("c", ["x", "y"], shape=(1, 2)),
    ]
    problem = Problem(
        "levers", parameters, [ScalarOutcome("y", kind=ScalarOutcome.MAXIMIZE)]
    )
    sample = Sample(a=np.array([[0.1, 0.2], [0.3, 0.4]]), c=np.array([["y", "x"]]))
    result = Sample._from_platypus_solution(sample._to_platypus_solution(problem))
    np.testing.assert_allclose(result["a"], sample["a"])
    assert result["a"].shape == (2, 2)
    assert result["c"].tolist() == [["y", "x"]]
