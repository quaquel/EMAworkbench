"""
An example of the lake problem using the ema workbench.

The model itself is adapted from the Rhodium example by Dave Hadka,
see https://gist.github.com/dhadka/a8d7095c98130d8f73bc

"""

import math
import time

import numpy as np
from scipy.optimize import brentq

from ema_workbench import (
    Model,
    RealParameter,
    ScalarOutcome,
    Constant,
    ema_logging,
    MPIEvaluator,
    save_results,
)


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
    try:
        decisions = [kwargs[str(i)] for i in range(100)]
    except KeyError:
        decisions = [0] * 100
        print("No valid decisions found, using 0 water release every year as default")

    nvars = len(decisions)
    decisions = np.array(decisions)

    # Calculate the critical pollution level (Pcrit)
    Pcrit = brentq(lambda x: x**q / (1 + x**q) - b * x, 0.01, 1.5)

    # Generate natural inflows using lognormal distribution
    natural_inflows = np.random.lognormal(
        mean=math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
        sigma=math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
        size=(nsamples, nvars),
    )

    # Initialize the pollution level matrix X
    X = np.zeros((nsamples, nvars))

    # Loop through time to compute the pollution levels
    for t in range(1, nvars):
        X[:, t] = (
            (1 - b) * X[:, t - 1]
            + (X[:, t - 1] ** q / (1 + X[:, t - 1] ** q))
            + decisions[t - 1]
            + natural_inflows[:, t - 1]
        )

    # Calculate the average daily pollution for each time step
    average_daily_P = np.mean(X, axis=0)

    # Calculate the reliability (probability of the pollution level being below Pcrit)
    reliability = np.sum(X < Pcrit) / float(nsamples * nvars)

    # Calculate the maximum pollution level (max_P)
    max_P = np.max(average_daily_P)

    # Calculate the utility by discounting the decisions using the discount factor (delta)
    utility = np.sum(alpha * decisions * np.power(delta, np.arange(nvars)))

    # Calculate the inertia (the fraction of time steps with changes larger than 0.02)
    inertia = np.sum(np.abs(np.diff(decisions)) > 0.02) / float(nvars - 1)

    return max_P, utility, inertia, reliability


if __name__ == "__main__":
    import ema_workbench

    # run with mpiexec -n 1 -usize {ntasks} python example_mpi_lake_model.py
    starttime = time.perf_counter()

    ema_logging.log_to_stderr(ema_logging.INFO, pass_root_logger_level=True)
    ema_logging.get_rootlogger().info(f"{ema_workbench.__version__}")

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
    lake_model.outcomes = [
        ScalarOutcome("max_P"),
        ScalarOutcome("utility"),
        ScalarOutcome("inertia"),
        ScalarOutcome("reliability"),
    ]

    # override some of the defaults of the model
    lake_model.constants = [Constant("alpha", 0.41), Constant("nsamples", 150)]

    # generate some random policies by sampling over levers
    n_scenarios = 10000
    n_policies = 4

    with MPIEvaluator(lake_model) as evaluator:
        res = evaluator.perform_experiments(n_scenarios, n_policies, chunksize=250)

    save_results(res, "test.tar.gz")

    print(time.perf_counter() - starttime)
