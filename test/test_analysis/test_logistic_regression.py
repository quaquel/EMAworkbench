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


def test_logit_coverage_density_res_dim():
    """Regression test: the coverage, density, and res_dim attributes.

    Previously, accessing these attributes raised an AttributeError.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    n = 500
    x = pd.DataFrame({"a": rng.random(n), "b": rng.random(n), "c": rng.random(n)})
    y = ((x.a > 0.6) & (x.b < 0.4)).values

    logitmodel = lr.Logit(x, y, threshold=0.5)
    logitmodel.run(disp=0)

    last = logitmodel.peeling_trajectory.iloc[-1]
    assert logitmodel.coverage == last["coverage"]
    assert logitmodel.density == last["density"]
    assert logitmodel.res_dim == last["res_dim"]
    assert logitmodel.res_dim == len(logitmodel.models)

    # changing the threshold changes coverage and density
    logitmodel.threshold = 0.9
    assert logitmodel.coverage == logitmodel.peeling_trajectory.iloc[-1]["coverage"]
    assert logitmodel.coverage < last["coverage"]


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


def test_logit_non_numeric_columns():
    """Regression test: categorical, boolean, and string columns.

    Previously, this raised a TypeError when normalizing the data, because
    the dummy variables for these columns are boolean.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    n = 500
    x = pd.DataFrame(
        {
            "a": rng.random(n),
            "b": rng.random(n),
            "category": pd.Categorical(rng.choice(["p", "q", "r"], n)),
            "flag": rng.random(n) > 0.5,
            "text": rng.choice(["u", "v"], n),
        }
    )
    y = ((x.a > 0.5) & x.flag).values

    logitmodel = lr.Logit(x, y)
    assert set(logitmodel.feature_names) == {
        "a",
        "b",
        "flag",
        "category!?!p",
        "category!?!q",
        "category!?!r",
        "text!?!u",
        "text!?!v",
    }
    normalized = logitmodel._normalized.drop(columns="Intercept")
    assert all(dtype == float for dtype in normalized.dtypes)
    assert normalized.min().min() == 0
    assert normalized.max().max() == 1

    logitmodel.run(disp=0)
    assert len(logitmodel.models) > 0
    selected = set(logitmodel.models[-1].params.index)
    assert {"a", "flag"} <= selected

    logitmodel.show_tradeoff()
    logitmodel.show_threshold_tradeoff(len(logitmodel.models) - 1)
    plt.close("all")
