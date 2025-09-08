"""Test samplers."""

# Created on 21 jan. 2013
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

import numpy as np

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
)


def test_lhs_sampler():
    uncertainties = [
        RealParameter("a", 0, 10),
        IntegerParameter("b", 0, 10),
        CategoricalParameter("c", ["a", "b", "c"]),
    ]

    sampler = LHSSampler()
    samples = sampler.generate_samples(uncertainties, 100, rng=np.random.default_rng(42))

    assert samples.shape == (100,3)

def test_mc_sampler():
    uncertainties = [
        RealParameter("a", 0, 10),
        IntegerParameter("b", 0, 10),
        CategoricalParameter("c", ["a", "b", "c"]),
    ]

    sampler = MonteCarloSampler()
    samples = sampler.generate_samples(uncertainties, 100, rng=np.random.default_rng(42))

    assert samples.shape == (100, 3)


def test_ff_sampler():
    uncertainties = [
        RealParameter("a", 0, 10),
        IntegerParameter("b", 0, 10),
        CategoricalParameter("c", ["a", "b", "c"]),
    ]

    sampler = FullFactorialSampler()
    samples = sampler.generate_samples(uncertainties, 100)

    assert samples.shape == (100*11*3, 3)

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
