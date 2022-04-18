"""
Created on 27 Jun 2019

@author: jhkwakkel
"""
from ema_workbench import (
    ema_logging,
    CategoricalParameter,
    MultiprocessingEvaluator,
    ScalarOutcome,
)

from ema_workbench.connectors.simio_connector import SimioModel

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    model = SimioModel(
        "simioDemo",
        wd="./model_bahareh",
        model_file="SupplyChainV3.spfx",
        main_model="Model",
    )

    model.uncertainties = [
        CategoricalParameter("DemandDistributionParameter", (20, 30, 40, 50, 60)),
        CategoricalParameter(
            "DemandInterarrivalTime", (0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2)
        ),
    ]

    model.levers = [
        CategoricalParameter("InitialInventory", (500, 600, 700, 800, 900)),
        CategoricalParameter("ReorderPoint", (100, 200, 300, 400, 500)),
        CategoricalParameter("OrderUpToQuantity", (500, 600, 700, 800, 900)),
        CategoricalParameter("ReviewPeriod", (3, 4, 5, 6, 7)),
    ]

    model.outcomes = [
        ScalarOutcome("AverageInventory"),
        ScalarOutcome("AverageServiceLevel"),
    ]

    n_scenarios = 10
    n_policies = 2

    with MultiprocessingEvaluator(model) as evaluator:
        results = evaluator.perform_experiments(n_scenarios, n_policies)
