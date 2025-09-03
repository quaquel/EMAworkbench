"""
Created on 20 May, 2011

This module shows how you can use vensim models directly
instead of coding the model in Python. The underlying case
is the same as used in fluExample

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
                epruyt <e.pruyt (at) tudelft (dot) nl>
"""

import numpy as np

from ema_workbench import (
    RealParameter,
    TimeSeriesOutcome,
    ema_logging,
    ScalarOutcome,
    perform_experiments,
    Policy,
    save_results,
)
from ema_workbench.connectors.vensim import VensimModel

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    model = VensimModel("fluCase", wd=r"./models/flu", model_file=r"FLUvensimV1basecase.vpm")

    # outcomes
    model.outcomes = [
        TimeSeriesOutcome(
            "deceased_population_region_1", variable_name="deceased population region 1"
        ),
        TimeSeriesOutcome("infected_fraction_R1", variable_name="infected fraction R1"),
        ScalarOutcome(
            "max_infection_fraction", variable_name="infected fraction R1", function=np.max
        ),
    ]

    # Plain Parametric Uncertainties
    model.uncertainties = [
        RealParameter(
            "additional_seasonal_immune_population_fraction_R1",
            0,
            0.5,
            variable_name="additional seasonal immune population fraction R1",
        ),
        RealParameter(
            "additional_seasonal_immune_population_fraction_R2",
            0,
            0.5,
            variable_name="additional seasonal immune population fraction R2",
        ),
        RealParameter(
            "fatality_ratio_region_1", 0.0001, 0.1, variable_name="fatality ratio region 1"
        ),
        RealParameter(
            "fatality_rate_region_2", 0.0001, 0.1, variable_name="fatality rate region 2"
        ),
        RealParameter(
            "initial_immune_fraction_of_the_population_of_region_1",
            0,
            0.5,
            variable_name="initial immune fraction of the population of region 1",
        ),
        RealParameter(
            "initial_immune_fraction_of_the_population_of_region_2",
            0,
            0.5,
            variable_name="initial immune fraction of the population of region 2",
        ),
        RealParameter(
            "normal_interregional_contact_rate",
            0,
            0.9,
            variable_name="normal interregional contact rate",
        ),
        RealParameter(
            "permanent_immune_population_fraction_R1",
            0,
            0.5,
            variable_name="permanent immune population fraction R1",
        ),
        RealParameter(
            "permanent_immune_population_fraction_R2",
            0,
            0.5,
            variable_name="permanent immune population fraction R2",
        ),
        RealParameter("recovery_time_region_1", 0.1, 0.75, variable_name="recovery time region 1"),
        RealParameter("recovery_time_region_2", 0.1, 0.75, variable_name="recovery time region 2"),
        RealParameter(
            "susceptible_to_immune_population_delay_time_region_1",
            0.5,
            2,
            variable_name="susceptible to immune population delay time region 1",
        ),
        RealParameter(
            "susceptible_to_immune_population_delay_time_region_2",
            0.5,
            2,
            variable_name="susceptible to immune population delay time region 2",
        ),
        RealParameter(
            "root_contact_rate_region_1", 0.01, 5, variable_name="root contact rate region 1"
        ),
        RealParameter(
            "root_contact_ratio_region_2", 0.01, 5, variable_name="root contact ratio region 2"
        ),
        RealParameter(
            "infection_ratio_region_1", 0, 0.15, variable_name="infection ratio region 1"
        ),
        RealParameter("infection_rate_region_2", 0, 0.15, variable_name="infection rate region 2"),
        RealParameter(
            "normal_contact_rate_region_1", 10, 100, variable_name="normal contact rate region 1"
        ),
        RealParameter(
            "normal_contact_rate_region_2", 10, 200, variable_name="normal contact rate region 2"
        ),
    ]

    # add policies
    policies = [
        Policy("no policy", model_file=r"FLUvensimV1basecase.vpm"),
        Policy("static policy", model_file=r"FLUvensimV1static.vpm"),
        Policy("adaptive policy", model_file=r"FLUvensimV1dynamic.vpm"),
    ]

    results = perform_experiments(model, 1000, policies=policies)
    save_results(results, "./data/1000 flu cases with policies.tar.gz")
