"""
Created on 27 Jan 2014

@author: jhkwakkel
"""

from ema_workbench import (
    RealParameter,
    TimeSeriesOutcome,
    ema_logging,
    MultiprocessingEvaluator,
    ScalarOutcome,
    perform_experiments,
    CategoricalParameter,
    save_results,
    Policy,
)
from ema_workbench.connectors.vensim import VensimModel
from ema_workbench.em_framework.evaluators import SequentialEvaluator


def get_energy_model():
    model = VensimModel(
        "energyTransition",
        wd="./models",
        model_file="RB_V25_ets_1_policy_modified_adaptive_extended_outcomes.vpm",
    )

    model.outcomes = [
        TimeSeriesOutcome(
            "cumulative_carbon_emissions", variable_name="cumulative carbon emissions"
        ),
        TimeSeriesOutcome(
            "carbon_emissions_reduction_fraction",
            variable_name="carbon emissions reduction fraction",
        ),
        TimeSeriesOutcome("fraction_renewables", variable_name="fraction renewables"),
        TimeSeriesOutcome("average_total_costs", variable_name="average total costs"),
        TimeSeriesOutcome("total_costs_of_electricity", variable_name="total costs of electricity"),
    ]

    model.uncertainties = [
        RealParameter(
            "demand_fuel_price_elasticity_factor",
            0,
            0.5,
            variable_name="demand fuel price elasticity factor",
        ),
        RealParameter(
            "economic_lifetime_biomass", 30, 50, variable_name="economic lifetime biomass"
        ),
        RealParameter("economic_lifetime_coal", 30, 50, variable_name="economic lifetime coal"),
        RealParameter("economic_lifetime_gas", 25, 40, variable_name="economic lifetime gas"),
        RealParameter("economic_lifetime_igcc", 30, 50, variable_name="economic lifetime igcc"),
        RealParameter("economic_lifetime_ngcc", 25, 40, variable_name="economic lifetime ngcc"),
        RealParameter(
            "economic_lifetime_nuclear", 50, 70, variable_name="economic lifetime nuclear"
        ),
        RealParameter("economic_lifetime_pv", 20, 30, variable_name="economic lifetime pv"),
        RealParameter("economic_lifetime_wind", 20, 30, variable_name="economic lifetime wind"),
        RealParameter("economic_lifetime_hydro", 50, 70, variable_name="economic lifetime hydro"),
        RealParameter(
            "uncertainty_initial_gross_fuel_costs",
            0.5,
            1.5,
            variable_name="uncertainty initial gross fuel costs",
        ),
        RealParameter(
            "investment_proportionality_constant",
            0.5,
            4,
            variable_name="investment proportionality constant",
        ),
        RealParameter(
            "investors_desired_excess_capacity_investment",
            0.2,
            2,
            variable_name="investors desired excess capacity investment",
        ),
        RealParameter(
            "price_demand_elasticity_factor",
            -0.07,
            -0.001,
            variable_name="price demand elasticity factor",
        ),
        RealParameter(
            "price_volatility_global_resource_markets",
            0.1,
            0.2,
            variable_name="price volatility global resource markets",
        ),
        RealParameter("progress_ratio_biomass", 0.85, 1, variable_name="progress ratio biomass"),
        RealParameter("progress_ratio_coal", 0.9, 1.05, variable_name="progress ratio coal"),
        RealParameter("progress_ratio_gas", 0.85, 1, variable_name="progress ratio gas"),
        RealParameter("progress_ratio_igcc", 0.9, 1.05, variable_name="progress ratio igcc"),
        RealParameter("progress_ratio_ngcc", 0.85, 1, variable_name="progress ratio ngcc"),
        RealParameter("progress_ratio_nuclear", 0.9, 1.05, variable_name="progress ratio nuclear"),
        RealParameter("progress_ratio_pv", 0.75, 0.9, variable_name="progress ratio pv"),
        RealParameter("progress_ratio_wind", 0.85, 1, variable_name="progress ratio wind"),
        RealParameter("progress_ratio_hydro", 0.9, 1.05, variable_name="progress ratio hydro"),
        RealParameter(
            "starting_construction_time", 0.1, 3, variable_name="starting construction time"
        ),
        RealParameter(
            "time_of_nuclear_power_plant_ban",
            2013,
            2100,
            variable_name="time of nuclear power plant ban",
        ),
        RealParameter(
            "weight_factor_carbon_abatement", 1, 10, variable_name="weight factor carbon abatement"
        ),
        RealParameter(
            "weight_factor_marginal_investment_costs",
            1,
            10,
            variable_name="weight factor marginal investment costs",
        ),
        RealParameter(
            "weight_factor_technological_familiarity",
            1,
            10,
            variable_name="weight factor technological familiarity",
        ),
        RealParameter(
            "weight_factor_technological_growth_potential",
            1,
            10,
            variable_name="weight factor technological growth potential",
        ),
        RealParameter(
            "maximum_battery_storage_uncertainty_constant",
            0.2,
            3,
            variable_name="maximum battery storage uncertainty constant",
        ),
        RealParameter(
            "maximum_no_storage_penetration_rate_wind",
            0.2,
            0.6,
            variable_name="maximum no storage penetration rate wind",
        ),
        RealParameter(
            "maximum_no_storage_penetration_rate_pv",
            0.1,
            0.4,
            variable_name="maximum no storage penetration rate pv",
        ),
        CategoricalParameter(
            "SWITCH_lookup_curve_TGC", (1, 2, 3, 4), variable_name="SWITCH lookup curve TGC"
        ),
        CategoricalParameter(
            "SWTICH_preference_carbon_curve", (1, 2), variable_name="SWTICH preference carbon curve"
        ),
        CategoricalParameter(
            "SWITCH_economic_growth", (1, 2, 3, 4, 5, 6), variable_name="SWITCH economic growth"
        ),
        CategoricalParameter(
            "SWITCH_electrification_rate",
            (1, 2, 3, 4, 5, 6),
            variable_name="SWITCH electrification rate",
        ),
        CategoricalParameter(
            "SWITCH_Market_price_determination",
            (1, 2),
            variable_name="SWITCH Market price determination",
        ),
        CategoricalParameter(
            "SWITCH_physical_limits", (1, 2), variable_name="SWITCH physical limits"
        ),
        CategoricalParameter(
            "SWITCH_low_reserve_margin_price_markup",
            (1, 2, 3, 4),
            variable_name="SWITCH low reserve margin price markup",
        ),
        CategoricalParameter(
            "SWITCH_interconnection_capacity_expansion",
            (1, 2, 3, 4),
            variable_name="SWITCH interconnection capacity expansion",
        ),
        CategoricalParameter(
            "SWITCH_storage_for_intermittent_supply",
            (1, 2, 3, 4, 5, 6, 7),
            variable_name="SWITCH storage for intermittent supply",
        ),
        CategoricalParameter("SWITCH_carbon_cap", (1, 2, 3), variable_name="SWITCH carbon cap"),
        CategoricalParameter(
            "SWITCH_TGC_obligation_curve", (1, 2, 3), variable_name="SWITCH TGC obligation curve"
        ),
        CategoricalParameter(
            "SWITCH_carbon_price_determination",
            (1, 2, 3),
            variable_name="SWITCH carbon price determination",
        ),
    ]
    return model


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)
    model = get_energy_model()

    policies = [
        Policy("no policy", model_file="RB_V25_ets_1_extended_outcomes.vpm"),
        Policy("static policy", model_file="ETSPolicy.vpm"),
        Policy(
            "adaptive policy",
            model_file="RB_V25_ets_1_policy_modified_adaptive_extended_outcomes.vpm",
        ),
    ]

    n = 100000
    with MultiprocessingEvaluator(model) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(n, policies=policies)
    #
    # outcomes.pop("TIME")
    # results = experiments, outcomes
    # save_results(results, f"./data/{n}_lhs.tar.gz")
