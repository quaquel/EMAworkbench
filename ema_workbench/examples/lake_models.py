"""dps and intertemporal version of the shallow lake problem."""

import math

import numpy as np
from scipy.optimize import brentq

__all__ = [
    "lake_problem_dps",
    "lake_problem_intertemporal",
]


def lake_problem_intertemporal(
    b: float = 0.42,  # decay rate for P in lake (0.42 = irreversible)
    q: float = 2.0,  # recycling exponent
    mean: float = 0.02,  # mean of natural inflows
    stdev: float = 0.001,  # future utility discount rate
    delta: float = 0.98,  # standard deviation of natural inflows
    alpha: float = 0.4,  # utility from pollution
    n_samples: int = 100,  # Monte Carlo sampling of natural inflows
    rng: int | None = 42,
    decisions: np.ndarray | None = None,
):
    """Run the intertemporal version of the shallow lake model."""
    if decisions is None:
        decisions = [0] * 100
        print("No valid decisions found, using 0 pollution release per year as default")

    rng = np.random.default_rng(rng)

    nvars = len(decisions)
    decisions = np.array(decisions)

    # Calculate the critical pollution level (p_crit)
    p_crit = brentq(lambda x: x**q / (1 + x**q) - b * x, 0.01, 1.5)

    # Generate natural inflows using lognormal distribution
    natural_inflows = rng.lognormal(
        mean=math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
        sigma=math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
        size=(n_samples, nvars),
    )

    # Initialize the pollution level matrix X
    X = np.zeros((n_samples, nvars))  # noqa: N806

    # Loop through time to compute the pollution levels
    for t in range(1, nvars):
        X[:, t] = (
            (1 - b) * X[:, t - 1]
            + (X[:, t - 1] ** q / (1 + X[:, t - 1] ** q))
            + decisions[t - 1]
            + natural_inflows[:, t - 1]
        )

    # Calculate the average daily pollution for each time step
    average_daily_p = np.mean(X, axis=0)

    # Calculate the reliability (probability of the pollution level being below Pcrit)
    reliability = np.sum(p_crit > X) / float(n_samples * nvars)

    # Calculate the maximum pollution level (max_P)
    max_p = np.max(average_daily_p)

    # Calculate the utility by discounting the decisions using the discount factor (delta)
    utility = np.sum(alpha * decisions * np.power(delta, np.arange(nvars)))

    # Calculate the inertia (the fraction of time steps with changes larger than 0.02)
    inertia = np.sum(np.abs(np.diff(decisions)) > 0.02) / float(nvars - 1)

    return max_p, utility, inertia, reliability


def get_antropogenic_release(
    xt: float, c1: float, c2: float, r1: float, r2: float, w1: float
):
    """Return anthropogenic release at xt.

    Parameters
    ----------
    xt : float
         pollution in lake at time t
    c1 : float
         center rbf 1
    c2 : float
         center rbf 2
    r1 : float
         radius rbf 1
    r2 : float
         radius rbf 2
    w1 : float
         weight of rbf 1

    note:: w2 = 1 - w1

    """
    rule = w1 * (abs(xt - c1) / r1) ** 3 + (1 - w1) * (abs(xt - c2) / r2) ** 3
    at1 = max(rule, 0.01)
    at = min(at1, 0.1)

    return at


def lake_problem_dps(
    b=0.42,  # decay rate for P in lake (0.42 = irreversible)
    q=2.0,  # recycling exponent
    mean=0.02,  # mean of natural inflows
    stdev=0.001,  # future utility discount rate
    delta=0.98,  # standard deviation of natural inflows
    alpha=0.4,  # utility from pollution
    n_samples=100,  # Monte Carlo sampling of natural inflows
    myears=1,  # the runtime of the simulation model
    c1=0.25,
    c2=0.25,
    r1=0.5,
    r2=0.5,
    w1=0.5,
    rng=42,
):
    """DPS version of the lake problem."""
    p_crit = brentq(lambda x: x**q / (1 + x**q) - b * x, 0.01, 1.5)

    rng = np.random.default_rng(rng)

    X = np.zeros(myears)  # noqa: N806
    average_daily_p = np.zeros(myears)
    reliability = 0.0
    inertia = 0
    utility = 0

    for _ in range(n_samples):
        X[0] = 0.0
        decision = 0.1

        decisions = np.zeros(myears)
        decisions[0] = decision

        natural_inflows = rng.lognormal(
            math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
            math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
            size=myears,
        )

        for t in range(1, myears):
            # here we use the decision rule
            decision = get_antropogenic_release(X[t - 1], c1, c2, r1, r2, w1)
            decisions[t] = decision

            X[t] = (
                (1 - b) * X[t - 1]
                + X[t - 1] ** q / (1 + X[t - 1] ** q)
                + decision
                + natural_inflows[t - 1]
            )
            average_daily_p[t] += X[t] / n_samples

        reliability += np.sum(p_crit > X) / (n_samples * myears)
        inertia += np.sum(np.absolute(np.diff(decisions) < 0.02)) / (n_samples * myears)
        utility += (
            np.sum(alpha * decisions * np.power(delta, np.arange(myears))) / n_samples
        )
    max_p = np.max(average_daily_p)

    return max_p, utility, inertia, reliability
