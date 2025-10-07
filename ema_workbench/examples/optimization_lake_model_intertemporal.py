"""An example of the lake problem using the ema workbench.

The model itself is adapted from the Rhodium example by Dave Hadka,
see https://gist.github.com/dhadka/a8d7095c98130d8f73bc

"""

from lake_models import lake_problem_intertemporal

from ema_workbench import (
    Constraint,
    Model,
    MultiprocessingEvaluator,
    RealParameter,
    ScalarOutcome,
    ema_logging,
)

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    # instantiate the model
    lake_model = Model("lakeproblem", function=lake_problem_intertemporal)
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
        ScalarOutcome("max_p", kind=ScalarOutcome.MINIMIZE, expected_range=(0, 5)),
        ScalarOutcome("utility", kind=ScalarOutcome.MAXIMIZE, expected_range=(0, 2)),
        ScalarOutcome("inertia", kind=ScalarOutcome.MAXIMIZE, expected_range=(0, 1)),
        ScalarOutcome(
            "reliability", kind=ScalarOutcome.MAXIMIZE, expected_range=(0, 1)
        ),
    ]

    constraints = [
        Constraint(
            "max_pollution", outcome_names="max_p", function=lambda x: max(0, x - 5)
        )
    ]

    with MultiprocessingEvaluator(lake_model) as evaluator:
        results = evaluator.optimize(
            nfe=5000,
            searchover="levers",
            epsilons=[0.125, 0.05, 0.01, 0.01],
            constraints=constraints,
            rng=42,
            filename="lake_model_intertemporal_archive.tar.gz",
            directory="./data",
        )
