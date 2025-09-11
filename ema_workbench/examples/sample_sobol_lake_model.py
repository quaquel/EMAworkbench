"""An example of the lake problem using the ema workbench.

The model itself is adapted from the Rhodium example by Dave Hadka,
see https://gist.github.com/dhadka/a8d7095c98130d8f73bc

"""

import math

import numpy as np
import pandas as pd
from SALib.analyze import sobol
from scipy.optimize import brentq

from ema_workbench import (
    Constant,
    Model,
    MultiprocessingEvaluator,
    Policy,
    RealParameter,
    ScalarOutcome,
    ema_logging,
)
from ema_workbench.em_framework import get_SALib_problem
from ema_workbench.em_framework.evaluators import Samplers


def lake_problem(
    b=0.42,  # decay rate for P in lake (0.42 = irreversible)
    q=2.0,  # recycling exponent
    mean=0.02,  # mean of natural inflows
    stdev=0.001,  # future utility discount rate
    delta=0.98,  # standard deviation of natural inflows
    alpha=0.4,  # utility from pollution
    nsamples=100,  # Monte Carlo sampling of natural inflows
    **kwargs,
):
    """Intertemporoal version of the shallow lake problem."""

    try:
        decisions = [kwargs[f"l{i}"] for i in range(100)]
    except KeyError:
        decisions = [0] * 100

    p_crit = brentq(lambda x: x ** q / (1 + x ** q) - b * x, 0.01, 1.5)
    n_vars = len(decisions)
    X = np.zeros((n_vars,))
    average_daily_p = np.zeros((n_vars,))
    decisions = np.array(decisions)
    reliability = 0.0

    for _ in range(nsamples):
        X[0] = 0.0

        natural_inflows = np.random.lognormal(
            math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
            math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
            size=n_vars,
        )

        for t in range(1, n_vars):
            X[t] = (
                (1 - b) * X[t - 1]
                + X[t - 1] ** q / (1 + X[t - 1] ** q)
                + decisions[t - 1]
                + natural_inflows[t - 1]
            )
            average_daily_p[t] += X[t] / float(nsamples)

        reliability += np.sum(p_crit > X) / float(nsamples * n_vars)

    max_p = np.max(average_daily_p)
    utility = np.sum(alpha * decisions * np.power(delta, np.arange(n_vars)))
    inertia = np.sum(np.absolute(np.diff(decisions)) < 0.02) / float(n_vars - 1)

    return max_p, utility, inertia, reliability


def analyze(results, ooi):
    """Analyze results using SALib sobol, returns a dataframe."""
    _, outcomes = results

    problem = get_SALib_problem(lake_model.uncertainties)
    y = outcomes[ooi]
    sobol_indices = sobol.analyze(problem, y)
    sobol_stats = {
        key: sobol_indices[key] for key in ["ST", "ST_conf", "S1", "S1_conf"]
    }
    sobol_stats = pd.DataFrame(sobol_stats, index=problem["names"])
    sobol_stats.sort_values(by="ST", ascending=False)
    s2 = pd.DataFrame(
        sobol_indices["S2"], index=problem["names"], columns=problem["names"]
    )
    s2_conf = pd.DataFrame(
        sobol_indices["S2_conf"], index=problem["names"], columns=problem["names"]
    )

    return sobol_stats, s2, s2_conf


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
    lake_model.levers = [
        RealParameter(f"l{i}", 0, 0.1) for i in range(lake_model.time_horizon)
    ]

    # specify outcomes
    lake_model.outcomes = [
        ScalarOutcome("max_p"),
        ScalarOutcome("utility"),
        ScalarOutcome("inertia"),
        ScalarOutcome("reliability"),
    ]

    # override some of the defaults of the model
    lake_model.constants = [Constant("alpha", 0.41), Constant("nsamples", 150)]

    # generate sa single default no release policy
    policy = Policy("no release", **{f"l{i}": 0.1 for i in range(100)})

    n_scenarios = 1000

    with MultiprocessingEvaluator(lake_model, n_processes=4) as evaluator:
        results = evaluator.perform_experiments(
            n_scenarios, policy, uncertainty_sampling=Samplers.SOBOL, uncertainty_sampling_kwargs={"calc_second_order":True}
        )

    sobol_stats, s2, s2_conf = analyze(results, "max_p")
    print(sobol_stats)
    print(s2)
    print(s2_conf)
