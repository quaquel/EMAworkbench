"""An example of performing robust many objective optimization with the workbench.

This example takes the direct policy search formulation of the lake problem as
found in Quinn et al (2017), but embeds in in a robust optimization.

Quinn, J.D., Reed, P.M., Keller, K. (2017)
Direct policy search for robust multi-objective management of deeply
uncertain socio-ecological tipping points. Environmental Modelling &
Software 92, 125-141.


"""

import math

import numpy as np
from scipy.optimize import brentq

from ema_workbench import (
    Constant,
    Model,
    MultiprocessingEvaluator,
    RealParameter,
    ScalarOutcome,
    ema_logging,
)
from ema_workbench.em_framework.samplers import LHSSampler

from lake_models import lake_problem_dps

# Created on 1 Jun 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    # instantiate the model
    lake_model = Model("lakeproblem", function=lake_problem_dps)
    # specify uncertainties
    lake_model.uncertainties = [
        RealParameter("b", 0.1, 0.45),
        RealParameter("q", 2.0, 4.5),
        RealParameter("mean", 0.01, 0.05),
        RealParameter("stdev", 0.001, 0.005),
        RealParameter("delta", 0.93, 0.99),
    ]

    # set levers
    lake_model.levers = [
        RealParameter("c1", -2, 2),
        RealParameter("c2", -2, 2),
        RealParameter("r1", 0, 2),
        RealParameter("r2", 0, 2),
        RealParameter("w1", 0, 1),
    ]

    # specify outcomes
    lake_model.outcomes = [
        ScalarOutcome("max_p"),
        ScalarOutcome("utility"),
        ScalarOutcome("inertia"),
        ScalarOutcome("reliability"),
    ]

    # override some of the defaults of the model
    lake_model.constants = [
        Constant("alpha", 0.41),
        Constant("n_samples", 100),
        Constant("myears", 100),
    ]

    # setup and execute the robust optimization
    def signal_to_noise(data):
        mean = np.mean(data)
        std = np.std(data)
        sn = mean / std
        return sn

    MAXIMIZE = ScalarOutcome.MAXIMIZE  # @UndefinedVariable
    MINIMIZE = ScalarOutcome.MINIMIZE  # @UndefinedVariable
    robustness_functions = [
        ScalarOutcome("mean_p", kind=MINIMIZE, variable_name="max_p", function=np.mean),
        ScalarOutcome("std_p", kind=MINIMIZE, variable_name="max_p", function=np.std),
        ScalarOutcome(
            "sn_reliability",
            kind=MAXIMIZE,
            variable_name="reliability",
            function=signal_to_noise,
        ),
    ]
    n_scenarios = 10
    scenarios = LHSSampler().generate_samples(lake_model.uncertainties, n_scenarios, rng=42)
    nfe = 1000

    with MultiprocessingEvaluator(lake_model) as evaluator:
        evaluator.robust_optimize(
            robustness_functions,
            scenarios,
            nfe=nfe,
            epsilons=[0.1] * len(robustness_functions),
            population_size=5,
        )
