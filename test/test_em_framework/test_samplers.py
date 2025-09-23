"""Test samplers."""

# Created on 21 jan. 2013
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

import numpy as np
import pytest

from ema_workbench.em_framework import Model
from ema_workbench.em_framework.parameters import (
    CategoricalParameter,
    IntegerParameter,
    RealParameter,
)
from ema_workbench.em_framework.points import Point, Policy, Scenario
from ema_workbench.em_framework.samplers import (
    FullFactorialSampler,
    LHSSampler,
    MonteCarloSampler,
    determine_parameters,
    sample_levers,
    sample_parameters,
    sample_uncertainties,
    DesignIterator,
)


def test_lhs_sampler():
    uncertainties = [
        RealParameter("a", 0, 10),
        IntegerParameter("b", 0, 10),
        CategoricalParameter("c", ["a", "b", "c"]),
    ]

    sampler = LHSSampler()
    samples = sampler.generate_samples(
        uncertainties, 100, rng=np.random.default_rng(42)
    )

    assert samples.shape == (100, 3)

    for i, u in enumerate(uncertainties):
        assert np.all(samples[:, i] >= u.lower_bound)
        assert np.all(samples[:, i] <= u.upper_bound)


def test_mc_sampler():
    uncertainties = [
        RealParameter("a", 0, 10),
        IntegerParameter("b", 0, 10),
        CategoricalParameter("c", ["a", "b", "c"]),
    ]

    sampler = MonteCarloSampler()
    samples = sampler.generate_samples(
        uncertainties, 100, rng=np.random.default_rng(42)
    )

    assert samples.shape == (100, 3)
    for i, u in enumerate(uncertainties):
        assert np.all(samples[:, i] >= u.lower_bound)
        assert np.all(samples[:, i] <= u.upper_bound)


def test_ff_sampler():
    uncertainties = [
        RealParameter("a", 0, 10),
        IntegerParameter("b", 0, 10),
        CategoricalParameter("c", ["a", "b", "c"]),
    ]

    sampler = FullFactorialSampler()
    samples = sampler.generate_samples(uncertainties, 100)

    assert samples.shape == (100 * 11 * 3, 3)
    for i, u in enumerate(uncertainties):
        assert np.all(samples[:, i] >= u.lower_bound)
        assert np.all(samples[:, i] <= u.upper_bound)


def test_determine_parameters(mocker):
    function = mocker.Mock()
    model_a = Model("A", function)
    model_a.uncertainties = [
        RealParameter("a", 0, 1),
        RealParameter("b", 0, 1),
    ]
    function = mocker.Mock()
    model_b = Model("B", function)
    model_b.uncertainties = [
        RealParameter("b", 0, 1),
        RealParameter("c", 0, 1),
    ]

    models = [model_a, model_b]

    parameters = determine_parameters(models, "uncertainties", union=True)
    for model in models:
        for unc in model.uncertainties:
            assert unc.name in parameters.keys()

    parameters = determine_parameters(models, "uncertainties", union=False)
    assert "b" in parameters.keys()
    assert "c" not in parameters.keys()
    assert "a" not in parameters.keys()


def test_sample_parameters():
    parameters = [
        RealParameter("a", 0, 10),
        IntegerParameter("b", 0, 10),
        CategoricalParameter("c", ["a", "b", "c"]),
    ]

    samples = sample_parameters(parameters, n_samples=100)

    assert samples.n == 100
    assert samples.kind is Point

    for sample in samples:
        for parameter in parameters:
            value = sample[parameter.name]

            match parameter.name:
                case "a":
                    assert parameter.lower_bound <= value <= parameter.upper_bound
                    assert isinstance(value, float)
                case "b":
                    assert parameter.lower_bound <= value <= parameter.upper_bound
                    assert isinstance(value, int)
                case "c":
                    assert value in parameter.categories
                    assert isinstance(value, str)


def test_sample_uncertainties(mocker):
    function = mocker.Mock()
    model = Model("A", function)
    model.uncertainties = [
        RealParameter("a", 0, 10),
        IntegerParameter("b", 0, 10),
        CategoricalParameter("c", ["a", "b", "c"]),
    ]

    samples = sample_uncertainties(model, n_samples=100)

    assert samples.n == 100
    assert samples.kind is Scenario


def test_sample_levers(mocker):
    function = mocker.Mock()
    model = Model("A", function)

    model.levers = [
        RealParameter("a", 0, 10),
        IntegerParameter("b", 0, 10),
        CategoricalParameter("c", ["a", "b", "c"]),
    ]

    samples = sample_levers(model, n_samples=100)

    assert samples.n == 100
    assert samples.kind is Policy


def test_design_iterator():
    rng = np.random.default_rng(42)

    samples = rng.uniform(size=(10, 3))
    parameters = [RealParameter(entry, 0, 1) for entry in "abc"]
    iterator = DesignIterator(samples, parameters, Point)

    # basic intit
    assert iterator.n == samples.shape[0]

    # iteration
    it = iter(iterator)
    for i, entry in enumerate(it):
        values = np.array([entry[k] for k in "abc"])
        assert np.all(values == samples[i])

    # combine
    ## full factorial
    samples1 = np.array([0, 1]).reshape(2, 1)
    samples2 = np.array([0, 1]).reshape(2, 1)

    it1 = DesignIterator(samples1, [RealParameter("a", 0, 1)], Point)
    it2 = DesignIterator(samples2, [RealParameter("a", 0, 1)], Point)

    it3 = it1.combine(it2, "full_factorial")
    samples = it3.samples

    assert np.all(samples == np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))

    ## equal length, so no need to cycle
    samples1 = np.array([0, 1]).reshape(2, 1)
    samples2 = np.array([0, 1]).reshape(2, 1)

    it1 = DesignIterator(samples1, [RealParameter("a", 0, 1)], Point)
    it2 = DesignIterator(samples2, [RealParameter("a", 0, 1)], Point)

    it3 = it1.combine(it2, "cycle")
    samples = it3.samples
    assert np.all(samples == np.array([[0, 0], [1, 1]]))

    ## cycle
    samples1 = np.array([0, 1, 2]).reshape(3, 1)
    samples2 = np.array([0, 1]).reshape(2, 1)

    it1 = DesignIterator(samples1, [RealParameter("a", 0, 1)], Point)
    it2 = DesignIterator(samples2, [RealParameter("a", 0, 1)], Point)

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
