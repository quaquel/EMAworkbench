"""Tests for logistic_regression.py."""

import matplotlib.pyplot as plt

from ema_workbench.analysis import logistic_regression as lr
from test import utilities

# Created on 16 Mar 2019
#
# @author: jhkwakkel


def flu_classify(data):
    """Helper function."""
    # get the output for deceased population
    result = data["deceased_population_region_1"][:, -1]
    return result > 1000000


def test_logit():
    """Test logit."""
    experiments, outcomes = utilities.load_flu_data()
    y = flu_classify(outcomes)

    logitmodel = lr.Logit(experiments, y)

    columns = set(
        experiments.drop(
            ["scenario", "policy", "model"], axis=1
        ).columns.values.tolist()
    )

    # check init
    for entry in logitmodel.feature_names:
        assert entry in columns

    logitmodel.run(method="newton", maxiter=2)

    logitmodel.show_tradeoff()
    logitmodel.show_threshold_tradeoff(1)
    logitmodel.plot_pairwise_scatter(1)
    logitmodel.inspect(1)

    logitmodel.threshold = 0.8

    plt.draw()
    plt.close("all")
