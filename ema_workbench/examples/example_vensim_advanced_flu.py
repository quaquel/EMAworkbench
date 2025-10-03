"""A more advanced illustration of using the workbench with vensim.

The underlying case is the same as used in example_flu.py

"""

import numpy as np

from ema_workbench import (
    MultiprocessingEvaluator,
    Sample,
    ScalarOutcome,
    TimeSeriesOutcome,
    ema_logging,
    save_results,
)
from ema_workbench.connectors.vensim import VensimModel
from ema_workbench.em_framework.parameters import parameters_from_csv

# Created on 20 May, 2011
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
#                 epruyt <e.pruyt (at) tudelft (dot) nl>


def time_of_max(infected_fraction, time):
    """Returns the time of the maximum infected fraction."""
    index = np.where(infected_fraction == np.max(infected_fraction))
    timing = time[index][0]
    return timing


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    model = VensimModel(
        "flu", wd="./models/flu", model_file="FLUvensimV1basecase.vpm"
    )

    # outcomes
    model.outcomes = [
        TimeSeriesOutcome(
            "deceased_population_region_1", variable_name="deceased population region 1"
        ),
        TimeSeriesOutcome("infected_fraction_R1", variable_name="infected fraction R1"),
        ScalarOutcome(
            "max_infection_fraction",
            variable_name="infected fraction R1",
            function=np.max,
        ),
        ScalarOutcome(
            "time_of_max",
            variable_name=["infected fraction R1", "TIME"],
            function=time_of_max,
        ),
    ]

    # create uncertainties based on csv
    # FIXME csv is missing
    model.uncertainties = parameters_from_csv("./models/flu/flu_uncertainties.csv")

    # add policies
    policies = [
        Sample("no policy", model_file="FLUvensimV1basecase.vpm"),
        Sample("static policy", model_file="FLUvensimV1static.vpm"),
        Sample("adaptive policy", model_file="FLUvensimV1dynamic.vpm"),
    ]

    with MultiprocessingEvaluator(model, n_processes=-1) as evaluator:
        results = evaluator.perform_experiments(1000, policies=policies)

    save_results(results, "./data/1000 flu cases with policies.tar.gz")
