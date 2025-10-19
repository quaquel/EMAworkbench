"""Tests for convergence metrics."""

import numpy as np
import pandas as pd


from ema_workbench import (
    HypervolumeMetric,
    SpacingMetric,
    GenerationalDistanceMetric,
    Problem,
    InvertedGenerationalDistanceMetric,
    EpsilonIndicatorMetric,
    RealParameter,
    ScalarOutcome, epsilon_nondominated,
)
from ema_workbench.em_framework.points import SampleCollection


# def create_solutions(n, rng):
#     """Helper function to create a set of solutions"""
#     parameters =
#     samples = SampleCollection(rng.random((n, 4)), parameters)
#
#
#
#     solutions = [sample._to_platypus_solution(problem) for sample in samples]
#
#     for solution, value in zip(solutions, rng.random((n, 2))):
#         solution.objectives[:] = value
#     return solutions


def test_metrics():
    """Test convergence metrics."""

    rng = np.random.default_rng(42)

    problem = Problem(
        "uncertainties",
        [RealParameter(entry, 0, 1) for entry in "abcd"],
        [
            ScalarOutcome("x", kind=ScalarOutcome.MAXIMIZE),
            ScalarOutcome("y", kind=ScalarOutcome.MAXIMIZE),
        ],
    )

    solutions = [pd.DataFrame(rng.random((100, 6)), columns=["a", "b", "c", "d", "x", "y"]),
                 pd.DataFrame(rng.random((100, 6)), columns=["a", "b", "c", "d", "x", "y"]),
                 pd.DataFrame(rng.random((100, 6)), columns=["a", "b", "c", "d", "x", "y"])]
    reference_set = epsilon_nondominated(solutions, [0.01, 0.01], problem)

    archive = pd.DataFrame(rng.random((100, 6)), columns=["a", "b", "c", "d", "x", "y"])

    hv = HypervolumeMetric(reference_set, problem)
    hv.calculate(archive)

    sm = SpacingMetric(reference_set, problem)
    sm.calculate(archive)

    gd = GenerationalDistanceMetric(reference_set, problem)
    gd.calculate(archive)

    igd = InvertedGenerationalDistanceMetric(reference_set, problem)
    igd.calculate(archive)

    ei = EpsilonIndicatorMetric(reference_set, problem)
    ei.calculate(archive)

