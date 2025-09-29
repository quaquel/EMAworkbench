"""Test samplers."""

# Created on 21 jan. 2013
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

import numpy as np

from ema_workbench.em_framework.parameters import (
    CategoricalParameter,
    IntegerParameter,
    RealParameter,
)
from ema_workbench.em_framework.samplers import (
    FullFactorialSampler,
    LHSSampler,
    MonteCarloSampler,
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
        assert np.all(samples.samples[:, i] >= u.lower_bound)
        assert np.all(samples.samples[:, i] <= u.upper_bound)


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
        assert np.all(samples.samples[:, i] >= u.lower_bound)
        assert np.all(samples.samples[:, i] <= u.upper_bound)


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
        assert np.all(samples.samples[:, i] >= u.lower_bound)
        assert np.all(samples.samples[:, i] <= u.upper_bound)




