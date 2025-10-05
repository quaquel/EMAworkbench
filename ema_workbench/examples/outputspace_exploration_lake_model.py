"""An example of using output space exploration on the lake problem.

The lake problem itself is adapted from the Rhodium example by Dave Hadka,
see https://gist.github.com/dhadka/a8d7095c98130d8f73bc

"""

from lake_models import lake_problem_intertemporal

from ema_workbench import (
    Constant,
    Model,
    MultiprocessingEvaluator,
    RealParameter,
    Sample,
    ScalarOutcome,
    ema_logging,
)
from ema_workbench.em_framework.outputspace_exploration import OutputSpaceExploration

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
    # note that outcomes of kind INFO will be ignored
    lake_model.outcomes = [
        ScalarOutcome("max_P", kind=ScalarOutcome.MAXIMIZE),
        ScalarOutcome("utility", kind=ScalarOutcome.MAXIMIZE),
        ScalarOutcome("inertia", kind=ScalarOutcome.MAXIMIZE),
        ScalarOutcome("reliability", kind=ScalarOutcome.MAXIMIZE),
    ]

    # override some of the defaults of the model
    lake_model.constants = [Constant("alpha", 0.41), Constant("n_samples", 150)]

    # generate a reference policy
    n_scenarios = 1000
    reference = Sample("no_policy", **{l.name: 0.02 for l in lake_model.levers})  # noqa: E741

    # we are doing output space exploration given a reference
    # policy, so we are exploring the output space over the uncertainties
    # grid spec specifies the grid structure imposed on the output space
    # each tuple is associated with an outcome. It gives the minimum
    # maximum, and epsilon value.
    with MultiprocessingEvaluator(lake_model) as evaluator:
        res = evaluator.optimize(
            algorithm=OutputSpaceExploration,
            grid_spec=[
                (0, 12, 0.5),
                (0, 1, 0.05),
                (0, 1, 0.1),
                (0, 1, 0.1),
            ],
            nfe=1000,
            searchover="uncertainties",
            reference=reference,
        )
