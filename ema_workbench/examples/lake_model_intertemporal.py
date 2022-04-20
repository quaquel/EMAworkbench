"""
An example of the lake problem using the ema workbench.

The model itself is adapted from the Rhodium example by Dave Hadka,
see https://gist.github.com/dhadka/a8d7095c98130d8f73bc

"""
import math

import numpy as np
from scipy.optimize import brentq

from ema_workbench import (
    Model,
    RealParameter,
    ScalarOutcome,
    ema_logging,
    MultiprocessingEvaluator,
    Constraint,
)
from ema_workbench.em_framework.optimization import HyperVolume, EpsilonProgress


def lake_problem(
    b=0.42,
    q=2.0,
    mean=0.02,
    stdev=0.0017,
    delta=0.98,
    alpha=0.4,
    nsamples=100,
    l0=0,
    l1=0,
    l2=0,
    l3=0,
    l4=0,
    l5=0,
    l6=0,
    l7=0,
    l8=0,
    l9=0,
    l10=0,
    l11=0,
    l12=0,
    l13=0,
    l14=0,
    l15=0,
    l16=0,
    l17=0,
    l18=0,
    l19=0,
    l20=0,
    l21=0,
    l22=0,
    l23=0,
    l24=0,
    l25=0,
    l26=0,
    l27=0,
    l28=0,
    l29=0,
    l30=0,
    l31=0,
    l32=0,
    l33=0,
    l34=0,
    l35=0,
    l36=0,
    l37=0,
    l38=0,
    l39=0,
    l40=0,
    l41=0,
    l42=0,
    l43=0,
    l44=0,
    l45=0,
    l46=0,
    l47=0,
    l48=0,
    l49=0,
    l50=0,
    l51=0,
    l52=0,
    l53=0,
    l54=0,
    l55=0,
    l56=0,
    l57=0,
    l58=0,
    l59=0,
    l60=0,
    l61=0,
    l62=0,
    l63=0,
    l64=0,
    l65=0,
    l66=0,
    l67=0,
    l68=0,
    l69=0,
    l70=0,
    l71=0,
    l72=0,
    l73=0,
    l74=0,
    l75=0,
    l76=0,
    l77=0,
    l78=0,
    l79=0,
    l80=0,
    l81=0,
    l82=0,
    l83=0,
    l84=0,
    l85=0,
    l86=0,
    l87=0,
    l88=0,
    l89=0,
    l90=0,
    l91=0,
    l92=0,
    l93=0,
    l94=0,
    l95=0,
    l96=0,
    l97=0,
    l98=0,
    l99=0,
):
    decisions = np.array(
        [
            l0,
            l1,
            l2,
            l3,
            l4,
            l5,
            l6,
            l7,
            l8,
            l9,
            l10,
            l11,
            l12,
            l13,
            l14,
            l15,
            l16,
            l17,
            l18,
            l19,
            l20,
            l21,
            l22,
            l23,
            l24,
            l25,
            l26,
            l27,
            l28,
            l29,
            l30,
            l31,
            l32,
            l33,
            l34,
            l35,
            l36,
            l37,
            l38,
            l39,
            l40,
            l41,
            l42,
            l43,
            l44,
            l45,
            l46,
            l47,
            l48,
            l49,
            l50,
            l51,
            l52,
            l53,
            l54,
            l55,
            l56,
            l57,
            l58,
            l59,
            l60,
            l61,
            l62,
            l63,
            l64,
            l65,
            l66,
            l67,
            l68,
            l69,
            l70,
            l71,
            l72,
            l73,
            l74,
            l75,
            l76,
            l77,
            l78,
            l79,
            l80,
            l81,
            l82,
            l83,
            l84,
            l85,
            l86,
            l87,
            l88,
            l89,
            l90,
            l91,
            l92,
            l93,
            l94,
            l95,
            l96,
            l97,
            l98,
            l99,
        ]
    )

    Pcrit = brentq(lambda x: x ** q / (1 + x ** q) - b * x, 0.01, 1.5)
    nvars = len(decisions)
    X = np.zeros((nvars,))
    average_daily_P = np.zeros((nvars,))
    decisions = np.array(decisions)
    reliability = 0.0

    for _ in range(nsamples):
        X[0] = 0.0

        natural_inflows = np.random.lognormal(
            math.log(mean ** 2 / math.sqrt(stdev ** 2 + mean ** 2)),
            math.sqrt(math.log(1.0 + stdev ** 2 / mean ** 2)),
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
    inertia = np.sum(np.abs(np.diff(decisions)) > 0.02) / float(nvars - 1)

    return max_P, utility, inertia, reliability


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    # instantiate the model
    lake_model = Model("lakeproblem", function=lake_problem)
    lake_model.time_horizon = 100  # used to specify the number of timesteps

    # specify uncertainties
    lake_model.uncertainties = [
        RealParameter("mean", 0.01, 0.05),
        RealParameter("stdev", 0.001, 0.005),
        RealParameter("b", 0.1, 0.45),
        RealParameter("q", 2.0, 4.5),
        RealParameter("delta", 0.93, 0.99),
    ]

    # set levers, one for each time step
    lake_model.levers = [
        RealParameter(f"l{i}", 0, 0.1) for i in range(lake_model.time_horizon)
    ]

    # specify outcomes
    # specify outcomes
    lake_model.outcomes = [
        ScalarOutcome("max_P", kind=ScalarOutcome.MINIMIZE, expected_range=(0, 5)),
        ScalarOutcome("utility", kind=ScalarOutcome.MAXIMIZE, expected_range=(0, 2)),
        ScalarOutcome("inertia", kind=ScalarOutcome.MAXIMIZE, expected_range=(0, 1)),
        ScalarOutcome(
            "reliability", kind=ScalarOutcome.MAXIMIZE, expected_range=(0, 1)
        ),
    ]

    convergence_metrics = [
        HyperVolume.from_outcomes(lake_model.outcomes),
        EpsilonProgress(),
    ]

    constraints = [
        Constraint(
            "max pollution", outcome_names="max_P", function=lambda x: max(0, x - 5)
        )
    ]

    with MultiprocessingEvaluator(lake_model) as evaluator:
        results, convergence = evaluator.optimize(
            nfe=5000,
            searchover="levers",
            epsilons=[0.125, 0.05, 0.01, 0.01],
            convergence=convergence_metrics,
            constraints=constraints,
        )
