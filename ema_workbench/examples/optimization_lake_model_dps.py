"""An example of using the workbench for many objective optimization.

This example replicates Quinn, J.D., Reed, P.M., Keller, K. (2017)
Direct policy search for robust multi-objective management of deeply
uncertain socio-ecological tipping points. Environmental Modelling &
Software 92, 125-141.

It also showcases how the workbench can be used to apply the MORDM extension
suggested by Watson, A.A., Kasprzyk, J.R. (2017) Incorporating deeply uncertain
factors into the many objective search process. Environmental Modelling &
Software 89, 159-171.

"""

import time

from lake_models import lake_problem_dps

from ema_workbench import (
    Constant,
    Model,
    MultiprocessingEvaluator,
    RealParameter,
    Sample,
    ScalarOutcome,
    ema_logging,
)
from ema_workbench.em_framework import LHSSampler
from ema_workbench.em_framework.optimization import GenerationalBorg

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
        ScalarOutcome("max_p", kind=ScalarOutcome.MINIMIZE),
        ScalarOutcome("utility", kind=ScalarOutcome.MAXIMIZE),
        ScalarOutcome("inertia", kind=ScalarOutcome.MAXIMIZE),
        ScalarOutcome("reliability", kind=ScalarOutcome.MAXIMIZE),
    ]

    # override some of the defaults of the model
    lake_model.constants = [
        Constant("alpha", 0.41),
        Constant("n_samples", 100),
        Constant("myears", 100),
    ]

    # reference is optional, but can be used to implement search for
    # various user specified scenarios along the lines suggested by
    # Watson and Kasprzyk (2017)
    reference = Sample("reference", b=0.4, q=2, mean=0.02, stdev=0.01)

    population = LHSSampler().generate_samples(lake_model.levers, 20)

    with MultiprocessingEvaluator(lake_model) as evaluator:
        for _ in range(5):
            seed = time.time()
            results, convergence_info = evaluator.optimize(
                searchover="levers",
                algorithm=GenerationalBorg,
                nfe=10000,
                convergence_freq=250,
                epsilons=[0.1] * len(lake_model.outcomes),
                reference=reference,
                filename=f"lake_model_dps_archive_{seed}.tar.gz",
                directory="./data/convergences",
                rng=seed,
                initial_population=population,
            )
            results.to_csv(f"./data/convergences/final_archive_{seed}.csv")
            convergence_info.to_csv(
                f"./data/convergences/runtime_convergence_info_{seed}.csv"
            )
