"""
Created on 8 mrt. 2011

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
                epruyt <e.pruyt (at) tudelft (dot) nl>
"""

from math import exp

from ema_workbench.connectors.vensim import VensimModel
from ema_workbench.em_framework import (
    RealParameter,
    CategoricalParameter,
    TimeSeriesOutcome,
    perform_experiments,
)
from ema_workbench.util import ema_logging


class ScarcityModel(VensimModel):
    def returnsToScale(self, x, speed, scale):
        return (x * 1000, scale * 1 / (1 + exp(-1 * speed * (x - 50))))

    def approxLearning(self, x, speed, scale, start):
        x = x - start
        loc = 1 - scale
        a = (x * 10000, scale * 1 / (1 + exp(speed * x)) + loc)
        return a

    def f(self, x, speed, loc):
        return (x / 10, loc * 1 / (1 + exp(speed * x)))

    def priceSubstite(self, x, speed, begin, end):
        scale = 2 * end
        start = begin - scale / 2

        return (x + 2000, scale * 1 / (1 + exp(-1 * speed * x)) + start)

    def run_model(self, scenario, policy):
        """Method for running an instantiated model structure"""
        kwargs = scenario
        loc = kwargs.pop("lookup_shortage_loc")
        speed = kwargs.pop("lookup_shortage_speed")
        lookup = [self.f(x / 10, speed, loc) for x in range(0, 100)]
        kwargs["shortage price effect lookup"] = lookup

        speed = kwargs.pop("lookup_price_substitute_speed")
        begin = kwargs.pop("lookup_price_substitute_begin")
        end = kwargs.pop("lookup_price_substitute_end")
        lookup = [self.priceSubstite(x, speed, begin, end) for x in range(0, 100, 10)]
        kwargs["relative price substitute lookup"] = lookup

        scale = kwargs.pop("lookup_returns_to_scale_speed")
        speed = kwargs.pop("lookup_returns_to_scale_scale")
        lookup = [self.returnsToScale(x, speed, scale) for x in range(0, 101, 10)]
        kwargs["returns to scale lookup"] = lookup

        scale = kwargs.pop("lookup_approximated_learning_speed")
        speed = kwargs.pop("lookup_approximated_learning_scale")
        start = kwargs.pop("lookup_approximated_learning_start")
        lookup = [self.approxLearning(x, speed, scale, start) for x in range(0, 101, 10)]
        kwargs["approximated learning effect lookup"] = lookup

        super().run_model(kwargs, policy)


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.DEBUG)

    model = ScarcityModel("scarcity", wd="./models/scarcity", model_file="MetalsEMA.vpm")

    model.outcomes = [
        TimeSeriesOutcome("relative_market_price", variable_name="relative market price"),
        TimeSeriesOutcome("supply_demand_ratio", variable_name="supply demand ratio"),
        TimeSeriesOutcome("real_annual_demand", variable_name="real annual demand"),
        TimeSeriesOutcome(
            "produced_of_intrinsically_demanded", variable_name="produced of intrinsically demanded"
        ),
        TimeSeriesOutcome("supply", variable_name="supply"),
        TimeSeriesOutcome(
            "Installed_Recycling_Capacity", variable_name="Installed Recycling Capacity"
        ),
        TimeSeriesOutcome(
            "Installed_Extraction_Capacity", variable_name="Installed Extraction Capacity"
        ),
    ]

    model.uncertainties = [
        RealParameter(
            "price_elasticity_of_demand", 0, 0.5, variable_name="price elasticity of demand"
        ),
        RealParameter(
            "fraction_of_maximum_extraction_capacity_used",
            0.6,
            1.2,
            variable_name="fraction of maximum extraction capacity used",
        ),
        RealParameter(
            "initial_average_recycling_cost", 1, 4, variable_name="initial average recycling cost"
        ),
        RealParameter(
            "exogenously_planned_extraction_capacity",
            0,
            15000,
            variable_name="exogenously planned extraction capacity",
        ),
        RealParameter(
            "absolute_recycling_loss_fraction",
            0.1,
            0.5,
            variable_name="absolute recycling loss fraction",
        ),
        RealParameter("normal_profit_margin", 0, 0.4, variable_name="normal profit margin"),
        RealParameter(
            "initial_annual_supply", 100000, 120000, variable_name="initial annual supply"
        ),
        RealParameter("initial_in_goods", 1500000, 2500000, variable_name="initial in goods"),
        RealParameter(
            "average_construction_time_extraction_capacity",
            1,
            10,
            variable_name="average construction time extraction capacity",
        ),
        RealParameter(
            "average_lifetime_extraction_capacity",
            20,
            40,
            variable_name="average lifetime extraction capacity",
        ),
        RealParameter(
            "average_lifetime_recycling_capacity",
            20,
            40,
            variable_name="average lifetime recycling capacity",
        ),
        RealParameter(
            "initial_extraction_capacity_under_construction",
            5000,
            20000,
            variable_name="initial extraction capacity under construction",
        ),
        RealParameter(
            "initial_recycling_capacity_under_construction",
            5000,
            20000,
            variable_name="initial recycling capacity under construction",
        ),
        RealParameter(
            "initial_recycling_infrastructure",
            5000,
            20000,
            variable_name="initial recycling infrastructure",
        ),
        # order of delay
        CategoricalParameter(
            "order_in_goods_delay", (1, 4, 10, 1000), variable_name="order in goods delay"
        ),
        CategoricalParameter(
            "order_recycling_capacity_delay",
            (1, 4, 10),
            variable_name="order recycling capacity delay",
        ),
        CategoricalParameter(
            "order_extraction_capacity_delay",
            (1, 4, 10),
            variable_name="order extraction capacity delay",
        ),
        # uncertainties associated with lookups
        RealParameter("lookup_shortage_loc", 20, 50, variable_name="lookup shortage loc"),
        RealParameter("lookup_shortage_speed", 1, 5, variable_name="lookup shortage speed"),
        RealParameter(
            "lookup_price_substitute_speed", 0.1, 0.5, variable_name="lookup price substitute speed"
        ),
        RealParameter(
            "lookup_price_substitute_begin", 3, 7, variable_name="lookup price substitute begin"
        ),
        RealParameter(
            "lookup_price_substitute_end", 15, 25, variable_name="lookup price substitute end"
        ),
        RealParameter(
            "lookup_returns_to_scale_speed",
            0.01,
            0.2,
            variable_name="lookup returns to scale speed",
        ),
        RealParameter(
            "lookup_returns_to_scale_scale", 0.3, 0.7, variable_name="lookup returns to scale scale"
        ),
        RealParameter(
            "lookup_approximated_learning_speed",
            0.01,
            0.2,
            variable_name="lookup approximated learning speed",
        ),
        RealParameter(
            "lookup_approximated_learning_scale",
            0.3,
            0.6,
            variable_name="lookup approximated learning scale",
        ),
        RealParameter(
            "lookup_approximated_learning_start",
            30,
            60,
            variable_name="lookup approximated learning start",
        ),
    ]

    results = perform_experiments(model, 1000)
