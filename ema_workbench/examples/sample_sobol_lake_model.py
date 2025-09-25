"""An example of the lake problem using the ema workbench.

The model itself is adapted from the Rhodium example by Dave Hadka,
see https://gist.github.com/dhadka/a8d7095c98130d8f73bc

"""

import pandas as pd
from lake_models import lake_problem_intertemporal
from SALib.analyze import sobol

from ema_workbench import (
    Constant,
    Model,
    MultiprocessingEvaluator,
    Sample,
    RealParameter,
    ScalarOutcome,
    ema_logging,
)
from ema_workbench.em_framework import get_SALib_problem
from ema_workbench.em_framework.evaluators import Samplers


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
    lake_model = Model("lakeproblem", function=lake_problem_intertemporal)
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
    lake_model.constants = [Constant("alpha", 0.41), Constant("n_samples", 150)]

    # generate sa single default no release policy
    policy = Sample("no release", **{f"l{i}": 0.1 for i in range(100)})

    n_scenarios = 1000

    with MultiprocessingEvaluator(lake_model, n_processes=4) as evaluator:
        results = evaluator.perform_experiments(
            n_scenarios,
            policy,
            uncertainty_sampling=Samplers.SOBOL,
            uncertainty_sampling_kwargs={"calc_second_order": True},
        )

    sobol_stats, s2, s2_conf = analyze(results, "max_p")
    print(sobol_stats)
    print(s2)
    print(s2_conf)
