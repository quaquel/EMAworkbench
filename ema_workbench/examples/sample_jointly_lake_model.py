"""An example of the lake problem using the ema workbench. This example
illustrated how you can control more finely how samples are being generated.
In this particular case, we want to apply Sobol analysis over both the
uncertainties and levers at the same time.

"""

import pandas as pd
from lake_models import lake_problem_dps
from SALib.analyze import sobol

from ema_workbench import (
    Constant,
    Model,
    MultiprocessingEvaluator,
    RealParameter,
    ScalarOutcome,
    Scenario,
    ema_logging,
)
from ema_workbench.em_framework import (
    SobolSampler,
    get_SALib_problem,
)


def analyze(results, ooi):
    """Analyze results using SALib sobol, returns a dataframe"""
    _, outcomes = results

    parameters = lake_model.uncertainties.copy() + lake_model.levers.copy()

    problem = get_SALib_problem(parameters)
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
        ScalarOutcome("max_P", kind=ScalarOutcome.MINIMIZE),
        # @UndefinedVariable
        ScalarOutcome("utility", kind=ScalarOutcome.MAXIMIZE),
        # @UndefinedVariable
        ScalarOutcome("inertia", kind=ScalarOutcome.MAXIMIZE),
        # @UndefinedVariable
        ScalarOutcome("reliability", kind=ScalarOutcome.MAXIMIZE),
    ]  # @UndefinedVariable

    # override some of the defaults of the model
    lake_model.constants = [
        Constant("alpha", 0.41),
        Constant("n_samples", 100),
        Constant("myears", 100),
    ]

    # combine parameters and uncertainties prior to sampling
    n_scenarios = 1000
    parameters = lake_model.uncertainties + lake_model.levers
    scenarios = sample_parameters(parameters, n_scenarios, SobolSampler(), Scenario)

    with MultiprocessingEvaluator(lake_model) as evaluator:
        results = evaluator.perform_experiments(scenarios)

    sobol_stats, s2, s2_conf = analyze(results, "max_P")
    print(sobol_stats)
    print(s2)
    print(s2_conf)
