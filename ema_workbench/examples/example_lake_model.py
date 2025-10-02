"""An example of the lake problem using the ema workbench.

The model itself is adapted from the Rhodium example by Dave Hadka,
see https://gist.github.com/dhadka/a8d7095c98130d8f73bc

"""

from lake_models import lake_problem_intertemporal

from ema_workbench import (
    Constant,
    Model,
    MultiprocessingEvaluator,
    RealParameter,
    ScalarOutcome,
    ema_logging,
)
from ema_workbench.em_framework.evaluators import Samplers

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

    # generate some random policies by sampling over levers
    n_scenarios = 100
    n_policies = 4

    with MultiprocessingEvaluator(lake_model) as evaluator:
        res = evaluator.perform_experiments(
            n_scenarios,
            n_policies,
            lever_sampling=Samplers.MC,
            lever_sampling_kwargs={"rng": 42},
            uncertainty_sampling_kwargs={"rng": 42},
        )
