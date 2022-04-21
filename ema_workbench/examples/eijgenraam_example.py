"""

"""

# Created on 12 Mar 2020
#
# .. codeauthor::  jhkwakkel

import bisect
import functools
import math
import operator

import numpy as np
import scipy as sp

from ema_workbench import (
    Model,
    RealParameter,
    ScalarOutcome,
    ema_logging,
    MultiprocessingEvaluator,
)

##==============================================================================
## Implement the model described by Eijgenraam et al. (2012)
# code taken from Rhodium Eijgenraam example
##------------------------------------------------------------------------------

# Parameters pulled from the paper describing each dike ring
params = ("c", "b", "lam", "alpha", "eta", "zeta", "V0", "P0", "max_Pf")
raw_data = {
    10: (16.6939, 0.6258, 0.0014, 0.033027, 0.320, 0.003774, 1564.9, 0.00044, 1 / 2000),
    11: (42.6200, 1.7068, 0.0000, 0.032000, 0.320, 0.003469, 1700.1, 0.00117, 1 / 2000),
    15: (
        125.6422,
        1.1268,
        0.0098,
        0.050200,
        0.760,
        0.003764,
        11810.4,
        0.00137,
        1 / 2000,
    ),
    16: (
        324.6287,
        2.1304,
        0.0100,
        0.057400,
        0.760,
        0.002032,
        22656.5,
        0.00110,
        1 / 2000,
    ),
    22: (
        154.4388,
        0.9325,
        0.0066,
        0.070000,
        0.620,
        0.002893,
        9641.1,
        0.00055,
        1 / 2000,
    ),
    23: (26.4653, 0.5250, 0.0034, 0.053400, 0.800, 0.002031, 61.6, 0.00137, 1 / 2000),
    24: (71.6923, 1.0750, 0.0059, 0.043900, 1.060, 0.003733, 2706.4, 0.00188, 1 / 2000),
    35: (49.7384, 0.6888, 0.0088, 0.036000, 1.060, 0.004105, 4534.7, 0.00196, 1 / 2000),
    38: (24.3404, 0.7000, 0.0040, 0.025321, 0.412, 0.004153, 3062.6, 0.00171, 1 / 1250),
    41: (
        58.8110,
        0.9250,
        0.0033,
        0.025321,
        0.422,
        0.002749,
        10013.1,
        0.00171,
        1 / 1250,
    ),
    42: (21.8254, 0.4625, 0.0019, 0.026194, 0.442, 0.001241, 1090.8, 0.00171, 1 / 1250),
    43: (
        340.5081,
        4.2975,
        0.0043,
        0.025321,
        0.448,
        0.002043,
        19767.6,
        0.00171,
        1 / 1250,
    ),
    44: (
        24.0977,
        0.7300,
        0.0054,
        0.031651,
        0.316,
        0.003485,
        37596.3,
        0.00033,
        1 / 1250,
    ),
    45: (3.4375, 0.1375, 0.0069, 0.033027, 0.320, 0.002397, 10421.2, 0.00016, 1 / 1250),
    47: (8.7813, 0.3513, 0.0026, 0.029000, 0.358, 0.003257, 1369.0, 0.00171, 1 / 1250),
    48: (35.6250, 1.4250, 0.0063, 0.023019, 0.496, 0.003076, 7046.4, 0.00171, 1 / 1250),
    49: (20.0000, 0.8000, 0.0046, 0.034529, 0.304, 0.003744, 823.3, 0.00171, 1 / 1250),
    50: (8.1250, 0.3250, 0.0000, 0.033027, 0.320, 0.004033, 2118.5, 0.00171, 1 / 1250),
    51: (15.0000, 0.6000, 0.0071, 0.036173, 0.294, 0.004315, 570.4, 0.00171, 1 / 1250),
    52: (49.2200, 1.6075, 0.0047, 0.036173, 0.304, 0.001716, 4025.6, 0.00171, 1 / 1250),
    53: (69.4565, 1.1625, 0.0028, 0.031651, 0.336, 0.002700, 9819.5, 0.00171, 1 / 1250),
}
data = {i: {k: v for k, v in zip(params, raw_data[i])} for i in raw_data.keys()}

# Set the ring we are analyzing
ring = 15
max_failure_probability = data[ring]["max_Pf"]


# Compute the investment cost to increase the dike height
def exponential_investment_cost(
    u,  # increase in dike height
    h0,  # original height of the dike
    c,  # constant from Table 1
    b,  # constant from Table 1
    lam,
):  # constant from Table 1
    if u == 0:
        return 0
    else:
        return (c + b * u) * math.exp(lam * (h0 + u))


def eijgenraam_model(
    X1,
    X2,
    X3,
    X4,
    X5,
    X6,
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    T=300,
    P0=data[ring]["P0"],
    V0=data[ring]["V0"],
    alpha=data[ring]["alpha"],
    delta=0.04,
    eta=data[ring]["eta"],
    gamma=0.035,
    rho=0.015,
    zeta=data[ring]["zeta"],
    c=data[ring]["c"],
    b=data[ring]["b"],
    lam=data[ring]["lam"],
):
    """Python implementation of the Eijgenraam model

    Params
    ------
    Xs : list
         list of dike heightenings
    Ts : list
         time of dike heightenings
    T : int, optional
        planning horizon
    P0 : <>, optional
         constant from Table 1
    V0 : <>, optional
         constant from Table 1
    alpha : <>, optional
            constant from Table 1
    delta : float, optional
            discount rate, mentioned in Section 2.2
    eta : <>, optional
          constant from Table 1
    gamma : float, optional
            paper says this is taken from government report, but no indication
            of actual value
    rho : float, optional
          risk-free rate, mentioned in Section 2.2
    zeta : <>, optional
           constant from Table 1
    c : <>, optional
        constant from Table 1
    b : <>, optional
        constant from Table 1
    lam : <>, optional
         constant from Table 1

    """
    Ts = [T1, T2, T3, T4, T5, T6]
    Xs = [X1, X2, X3, X4, X5, X6]

    Ts = [int(Ts[i] + sum(Ts[:i])) for i in range(len(Ts)) if Ts[i] + sum(Ts[:i]) < T]
    Xs = Xs[: len(Ts)]

    if len(Ts) == 0:
        Ts = [0]
        Xs = [0]

    if Ts[0] > 0:
        Ts.insert(0, 0)
        Xs.insert(0, 0)

    S0 = P0 * V0
    beta = alpha * eta + gamma - rho
    theta = alpha - zeta

    # calculate investment
    investment = 0

    for i in range(len(Xs)):
        step_cost = exponential_investment_cost(
            Xs[i], 0 if i == 0 else sum(Xs[:i]), c, b, lam
        )
        step_discount = math.exp(-delta * Ts[i])
        investment += step_cost * step_discount

    # calculate expected losses
    losses = 0

    for i in range(len(Xs) - 1):
        losses += math.exp(-theta * sum(Xs[: (i + 1)])) * (
            math.exp((beta - delta) * Ts[i + 1]) - math.exp((beta - delta) * Ts[i])
        )

    if Ts[-1] < T:
        losses += math.exp(-theta * sum(Xs)) * (
            math.exp((beta - delta) * T) - math.exp((beta - delta) * Ts[-1])
        )

    losses = losses * S0 / (beta - delta)

    # salvage term
    losses += (
        S0
        * math.exp(beta * T)
        * math.exp(-theta * sum(Xs))
        * math.exp(-delta * T)
        / delta
    )

    def find_height(t):
        if t < Ts[0]:
            return 0
        elif t > Ts[-1]:
            return sum(Xs)
        else:
            return sum(Xs[: bisect.bisect_right(Ts, t)])

    failure_probability = [
        P0 * np.exp(alpha * eta * t) * np.exp(-alpha * find_height(t))
        for t in range(T + 1)
    ]
    total_failure = 1 - functools.reduce(
        operator.mul, [1 - p for p in failure_probability], 1
    )
    mean_failure = sum(failure_probability) / (T + 1)
    max_failure = max(failure_probability)

    return (
        investment,
        losses,
        investment + losses,
        total_failure,
        mean_failure,
        max_failure,
    )


if __name__ == "__main__":
    model = Model("eijgenraam", eijgenraam_model)

    model.responses = [
        ScalarOutcome("TotalInvestment", ScalarOutcome.INFO),
        ScalarOutcome("TotalLoss", ScalarOutcome.INFO),
        ScalarOutcome("TotalCost", ScalarOutcome.MINIMIZE),
        ScalarOutcome("TotalFailureProb", ScalarOutcome.INFO),
        ScalarOutcome("AvgFailureProb", ScalarOutcome.MINIMIZE),
        ScalarOutcome("MaxFailureProb", ScalarOutcome.MINIMIZE),
    ]

    # Set uncertainties
    model.uncertainties = [
        RealParameter.from_dist("P0", sp.stats.lognorm(scale=0.00137, s=0.25)),
        # @UndefinedVariable
        RealParameter.from_dist("alpha", sp.stats.norm(loc=0.0502, scale=0.01)),
        # @UndefinedVariable
        RealParameter.from_dist("eta", sp.stats.lognorm(scale=0.76, s=0.1)),
    ]  # @UndefinedVariable

    # having a list like parameter were values are automagically wrappen
    # into a list can be quite usefull.....
    model.levers = [RealParameter(f"X{i}", 0, 500) for i in range(1, 7)] + [
        RealParameter(f"T{i}", 0, 300) for i in range(1, 7)
    ]

    ema_logging.log_to_stderr(ema_logging.INFO)

    with MultiprocessingEvaluator(model, n_processes=4) as evaluator:
        results = evaluator.perform_experiments(1000, 4)
