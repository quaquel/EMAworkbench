"""A showcase of a mexican flu model with Vensim.

This module shows how you can use vensim models directly
instead of coding the model in Python. The underlying case
is the same as used in example_flu.py


"""

# Created on 20 May, 2011
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
#                 epruyt <e.pruyt (at) tudelft (dot) nl>

import numpy as np

from ema_workbench import (
    Sample,
    RealParameter,
    ScalarOutcome,
    TimeSeriesOutcome,
    ema_logging,
    perform_experiments,
    save_results,
)
from ema_workbench.connectors.vensim import VensimModel

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    model = VensimModel(
        "fluCase", wd=r"./models/flu", model_file=r"FLUvensimV1basecase.vpm"
    )

    # outcomes
    model.outcomes = [
        TimeSeriesOutcome("deceased_population_region_1"),
        TimeSeriesOutcome("infected_fraction_R1"),
        ScalarOutcome(
            "max_infection_fraction",
            function=np.max,
            variable_name="infected_fraction_R1",
        ),
    ]

    # Plain Parametric Uncertainties
    model.uncertainties = [
        RealParameter("additional_seasonal_immune_population_fraction_R1", 0, 0.5),
        RealParameter(
            "additional_seasonal_immune_population_fraction_R2",
            0,
            0.5,
        ),
        RealParameter(
            "fatality_ratio_region_1",
            0.0001,
            0.1,
        ),
        RealParameter(
            "fatality_rate_region_2",
            0.0001,
            0.1,
        ),
        RealParameter(
            "initial_immune_fraction_of_the_population_of_region_1",
            0,
            0.5,
        ),
        RealParameter(
            "initial_immune_fraction_of_the_population_of_region_2",
            0,
            0.5,
        ),
        RealParameter(
            "normal_interregional_contact_rate",
            0,
            0.9,
        ),
        RealParameter(
            "permanent_immune_population_fraction_R1",
            0,
            0.5,
        ),
        RealParameter(
            "permanent_immune_population_fraction_R2",
            0,
            0.5,
        ),
        RealParameter("recovery_time_region_1", 0.1, 0.75),
        RealParameter("recovery_time_region_2", 0.1, 0.75),
        RealParameter("susceptible_to_immune_population_delay_time_region_1", 0.5, 2),
        RealParameter("susceptible_to_immune_population_delay_time_region_2", 0.5, 2),
        RealParameter("root_contact_rate_region_1", 0.01, 5),
        RealParameter("root_contact_ratio_region_2", 0.01, 5),
        RealParameter(
            "infection_ratio_region_1",
            0,
            0.15,
        ),
        RealParameter("infection_rate_region_2", 0, 0.15),
        RealParameter(
            "normal_contact_rate_region_1",
            10,
            100,
        ),
        RealParameter(
            "normal_contact_rate_region_2",
            10,
            200,
        ),
    ]

    # add policies
    policies = [
        Sample("no policy", model_file=r"FLUvensimV1basecase.vpm"),
        Sample("static policy", model_file=r"FLUvensimV1static.vpm"),
        Sample("adaptive policy", model_file=r"FLUvensimV1dynamic.vpm"),
    ]

    results = perform_experiments(model, 1000, policies=policies)
    save_results(results, "./data/1000 flu cases with policies.tar.gz")
