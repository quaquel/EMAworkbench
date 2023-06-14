"""

This example is a proof of principle for how NetLogo models can be
controlled using pyNetLogo and the ema_workbench. Note that this
example uses the NetLogo 6 version of the predator prey model that
comes with NetLogo. If you are using NetLogo 5, replace the model file
with the one that comes with NetLogo.

"""
import numpy as np

from ema_workbench import (
    RealParameter,
    ema_logging,
    ScalarOutcome,
    TimeSeriesOutcome,
    MultiprocessingEvaluator,
)
from ema_workbench.connectors.netlogo import NetLogoModel

# Created on 20 mrt. 2013
#
# .. codeauthor::  jhkwakkel


if __name__ == "__main__":
    # turn on logging
    ema_logging.log_to_stderr(ema_logging.INFO)

    model = NetLogoModel(
        "predprey", wd="./models/predatorPreyNetlogo", model_file="Wolf Sheep Predation.nlogo"
    )
    model.run_length = 100
    model.replications = 10

    model.uncertainties = [
        RealParameter("grass-regrowth-time", 1, 99),
        RealParameter("initial-number-sheep", 50, 100),
        RealParameter("initial-number-wolves", 50, 100),
        RealParameter("sheep-reproduce", 5, 10),
        RealParameter("wolf-reproduce", 5, 10),
    ]

    model.outcomes = [
        ScalarOutcome("sheep", variable_name="count sheep", function=np.mean),
        TimeSeriesOutcome("wolves"),
        TimeSeriesOutcome("grass"),
    ]

    # perform experiments
    n = 10

    with MultiprocessingEvaluator(model, n_processes=-1, maxtasksperchild=4) as evaluator:
        results = evaluator.perform_experiments(n)

    print()
