"""embedding SALib sampling within the workbench"""


import pytest

from ema_workbench import RealParameter
from ema_workbench.em_framework.parameters import IntegerParameter
from ema_workbench.em_framework.salib_samplers import (
    FASTSampler,
    MorrisSampler,
    SobolSampler,
    get_SALib_problem,
)

# Created on 14 Mar 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = []


def test_sobol():
    parameters = [RealParameter("a", 0, 10), RealParameter("b", 0, 5)]

    sampler = SobolSampler()
    samples = sampler.generate_samples(parameters, 100)

    N = 100 * (2 * 2 + 2)
    assert samples.shape == (N, len(parameters))

    sampler = SobolSampler(second_order=False)
    samples = sampler.generate_samples(parameters, 100)

    N = 100 * (2 + 2)
    assert samples.shape == (N, len(parameters))

    parameters = [
        RealParameter("a", 0, 10),
        RealParameter("b", 0, 5),
        IntegerParameter("c", 0, 2),
    ]

    sampler = SobolSampler()
    samples = sampler.generate_samples(parameters, 100)

    N = 100 * (2 * 3 + 2)
    assert samples.shape == (N, len(parameters))


def test_morris():
    parameters = [RealParameter("a", 0, 10), RealParameter("b", 0, 5)]

    sampler = MorrisSampler()
    samples = sampler.generate_samples(parameters, 100)

    G = 4
    D = len(parameters)
    N = 100

    N = (G / D + 1) * N
    assert samples.shape == (N, len(parameters))

def test_FAST():
    parameters = [RealParameter("a", 0, 10), RealParameter("b", 0, 5)]

    sampler = FASTSampler()
    samples = sampler.generate_samples(parameters, 100)

    N = 100 * 2
    assert samples.shape == (N, len(parameters))

def test_get_salib_problem():
    uncertainties = [
        RealParameter("a", 0, 10),
        RealParameter("b", 0, 5),
        IntegerParameter("c", 0, 4),
    ]

    problem = get_SALib_problem(uncertainties)
    assert len(uncertainties) == problem["num_vars"]
    assert [u.name for u in uncertainties] == problem["names"]

    #         for i, (name, bounds) in enumerate(zip(problem['names'], problem['bounds'])):
    #             self.assertEqual()

    assert (0, 10) == problem["bounds"][0]
    assert (0, 5) == problem["bounds"][1]
    assert (0, 4 + 1) == problem["bounds"][2]  # because flooring


