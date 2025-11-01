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


def test_sample_platypus():
    """Tests for converting samples to platypus solutions and vice versa."""
    parameters = [
        RealParameter("a", 0, 1),
        IntegerParameter("b", 1, 4),
        CategoricalParameter("c", ["a", "b", "c"]),
        CategoricalParameter("d", ["a", "b", "c"], shape=(2,)),
    ]

    problem = Problem("levers", parameters, [], [])

    sample = Sample(a=0.5, b=2, c="b", d=np.asarray(["a", "b",]))
    solution = sample._to_platypus_solution(problem)

    new_sample = Sample._from_platypus_solution(solution)

    assert sample["a"] == new_sample["a"]
    assert sample["b"] == new_sample["b"]
    assert sample["c"] == new_sample["c"]
    assert np.all(sample["d"] == new_sample["d"])

