"""
An example of using output space exploration on the lake problem

The lake problem itself is adapted from the Rhodium example by Dave Hadka,
see https://gist.github.com/dhadka/a8d7095c98130d8f73bc

"""
import math

import numpy as np
from scipy.optimize import brentq

from ema_workbench import (
    Model,
    RealParameter,
    ScalarOutcome,
    Constant,
    ema_logging,
    MultiprocessingEvaluator,
    Policy,
)

from ema_workbench.em_framework.outputspace_exploration import OutputSpaceExploration


def lake_problem(
    b=0.42,  # decay rate for P in lake (0.42 = irreversible)
    q=2.0,  # recycling exponent
    mean=0.02,  # mean of natural inflows
    stdev=0.001,  # future utility discount rate
    delta=0.98,  # standard deviation of natural inflows
    alpha=0.4,  # utility from pollution
    nsamples=100,  # Monte Carlo sampling of natural inflows
    **kwargs
):
    try:
        decisions = [kwargs[str(i)] for i in range(100)]
    except KeyError:
        decisions = [
            0,
        ] * 100

    Pcrit = brentq(lambda x: x**q / (1 + x**q) - b * x, 0.01, 1.5)
    nvars = len(decisions)
    X = np.zeros((nvars,))
    average_daily_P = np.zeros((nvars,))
    decisions = np.array(decisions)
    reliability = 0.0

    for _ in range(nsamples):
        X[0] = 0.0

        natural_inflows = np.random.lognormal(
            math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
            math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
            size=nvars,
        )

        for t in range(1, nvars):
            X[t] = (
                (1 - b) * X[t - 1]
                + X[t - 1] ** q / (1 + X[t - 1] ** q)
                + decisions[t - 1]
                + natural_inflows[t - 1]
            )
            average_daily_P[t] += X[t] / float(nsamples)

        reliability += np.sum(X < Pcrit) / float(nsamples * nvars)

    max_P = np.max(average_daily_P)
    utility = np.sum(alpha * decisions * np.power(delta, np.arange(nvars)))
    inertia = np.sum(np.absolute(np.diff(decisions)) < 0.02) / float(nvars - 1)

    return max_P, utility, inertia, reliability


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    # instantiate the model
    lake_model = Model("lakeproblem", function=lake_problem)
    lake_model.time_horizon = 100

    # specify uncertainties
    lake_model.uncertainties = [
        RealParameter("b", 0.1, 0.45),
        RealParameter("q", 2.0, 4.5),
        RealParameter("mean", 0.01, 0.05),
        RealParameter("stdev", 0.001, 0.005),
        RealParameter("delta", 0.93, 0.99),
    ]

    # set levers, one for each time step
    lake_model.levers = [RealParameter(str(i), 0, 0.1) for i in range(lake_model.time_horizon)]

    # specify outcomes
    # note that outcomes of kind INFO will be ignored
    lake_model.outcomes = [
        ScalarOutcome("max_P", kind=ScalarOutcome.MAXIMIZE),
        ScalarOutcome("utility", kind=ScalarOutcome.MAXIMIZE),
        ScalarOutcome("inertia", kind=ScalarOutcome.MAXIMIZE),
        ScalarOutcome("reliability", kind=ScalarOutcome.MAXIMIZE),
    ]

    # override some of the defaults of the model
    lake_model.constants = [Constant("alpha", 0.41), Constant("nsamples", 150)]

    # generate a reference policy
    n_scenarios = 1000
    reference = Policy("nopolicy", **{l.name: 0.02 for l in lake_model.levers})

    # we are doing output space exploration given a reference
    # policy, so we are exploring the output space over the uncertainties
    # grid spec specifies the grid structure imposed on the output space
    # each tuple is associated with an outcome. It gives the minimum
    # maximum, and epsilon value.
    with MultiprocessingEvaluator(lake_model) as evaluator:
        res = evaluator.optimize(
            algorithm=OutputSpaceExploration,
            grid_spec=[
                (0, 12, 0.5),
                (0, 1, 0.05),
                (0, 1, 0.1),
                (0, 1, 0.1),
            ],
            nfe=1000,
            searchover="uncertainties",
            reference=reference,
        )
