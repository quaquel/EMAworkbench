"""Tests for convergence metrics."""

import numpy as np
import pandas as pd

from ema_workbench import (
    EpsilonIndicatorMetric,
    GenerationalDistanceMetric,
    HypervolumeMetric,
    InvertedGenerationalDistanceMetric,
    Problem,
    RealParameter,
    ScalarOutcome,
    SpacingMetric,
    epsilon_nondominated,
)


def test_metrics():
    """Test convergence metrics."""
    # fixme currently only ensures it runs, but no further assertions
    rng = np.random.default_rng(42)

    problem = Problem(
        "uncertainties",
        [RealParameter(entry, 0, 1) for entry in "abcd"],
        [
            ScalarOutcome("x", kind=ScalarOutcome.MAXIMIZE),
            ScalarOutcome("y", kind=ScalarOutcome.MAXIMIZE),
        ],
    )
    columns = list("abcdxy")

    solutions = [
        pd.DataFrame(rng.random((100, 6)), columns=columns),
        pd.DataFrame(rng.random((100, 6)), columns=columns),
        pd.DataFrame(rng.random((100, 6)), columns=columns),
    ]
    reference_set = epsilon_nondominated(solutions, [0.01, 0.01], problem)

    archive = pd.DataFrame(rng.random((100, 6)), columns=columns)

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
