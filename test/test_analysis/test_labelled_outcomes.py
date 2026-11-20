"""Tests for using outcomes that are DataArrays in the analysis package."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from ema_workbench.analysis import (
    cart,
    clusterer,
    dimensional_stacking,
    feature_scoring,
    logistic_regression,
    pairs_plotting,
    plotting,
    prim,
    regional_sa,
)
from ema_workbench.analysis.plotting_util import (
    Density,
    determine_time_indices,
    prepare_data,
)
from ema_workbench.util import EMAError
from test import utilities

OUTCOME = "deceased_population_region_1"
OTHER = "infected_fraction_R1"


@pytest.fixture(scope="module")
def flu():
    """The flu results, with the outcomes as numpy arrays and as DataArrays."""
    experiments, outcomes = utilities.load_flu_data()
    time = outcomes["TIME"][0]
    labelled = {
        key: xr.DataArray(
            value, dims=("experiment", "time"), coords={"time": time}, name=key
        )
        for key, value in outcomes.items()
        if key != "TIME"
    }
    x = experiments.drop(columns=["model", "policy", "scenario"])
    return experiments, x, outcomes, labelled, time


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# scenario discovery and sensitivity analysis


def test_prim_with_dataarray(flu):
    _, x, outcomes, labelled, _ = flu
    y_array = outcomes[OUTCOME][:, -1] > 1e6
    y_labelled = labelled[OUTCOME][:, -1] > 1e6
    assert isinstance(y_labelled, xr.DataArray)

    box_array = prim.Prim(x, y_array).find_box()
    box_labelled = prim.Prim(x, y_labelled).find_box()
    assert box_labelled.peeling_trajectory.equals(box_array.peeling_trajectory)

    box = prim.RegressionPrim(x, labelled[OUTCOME].isel(time=-1)).find_box()
    assert len(box.peeling_trajectory) > 1


def test_pca_preprocess_with_dataarray(flu):
    _, x, outcomes, labelled, _ = flu
    rotated_array, _ = prim.pca_preprocess(x, outcomes[OUTCOME][:, -1] > 1e6)
    rotated_labelled, _ = prim.pca_preprocess(x, labelled[OUTCOME][:, -1] > 1e6)
    assert rotated_labelled.equals(rotated_array)


def test_other_scenario_discovery_with_dataarray(flu):
    _, x, _, labelled, _ = flu
    y_class = labelled[OUTCOME][:, -1] > 1e6
    y_value = labelled[OUTCOME].sel(time=48)

    tree = cart.CART(x, y_class)
    tree.build_tree()
    assert type(tree.y) is np.ndarray
    cart.CART(x, y_value, mode=cart.sdutil.RuleInductionType.REGRESSION).build_tree()
    assert type(logistic_regression.Logit(x, y_class).y) is np.ndarray

    scores = feature_scoring.get_feature_scores_all(x, {"end": y_value})
    assert list(scores.columns) == ["end"]
    feature_scoring.get_ex_feature_scores(
        x, y_class, mode=feature_scoring.RuleInductionType.CLASSIFICATION
    )
    feature_scoring.get_univariate_feature_scores(x, y_class)

    regional_sa.plot_cdfs(x, y_class)
    dimensional_stacking.create_pivot_plot(x, y_class)


# time index used for plotting


def test_determine_time_indices(flu):
    _, _, outcomes, labelled, time = flu

    # the coordinates of a DataArray
    shifted = labelled[OTHER].assign_coords(time=time + 100)
    without_coords = xr.DataArray(outcomes[OTHER], dims=("experiment", "time"))
    times, arrays = determine_time_indices(
        {
            "labelled": labelled[OUTCOME],
            "shifted": shifted,
            "positional": without_coords,
            "scalar": np.zeros(len(outcomes[OTHER])),
        }
    )
    np.testing.assert_array_equal(times["labelled"], time)
    np.testing.assert_array_equal(times["shifted"], time + 100)
    np.testing.assert_array_equal(times["positional"], np.arange(time.size))
    assert times["scalar"] is None
    assert all(type(value) is np.ndarray for value in arrays.values())

    # an outcome named TIME, for numpy arrays of the same length
    times, arrays = determine_time_indices(
        {
            "array": outcomes[OUTCOME],
            "shorter": outcomes[OUTCOME][:, :10],
            "labelled": shifted,
            "TIME": outcomes["TIME"],
        }
    )
    assert "TIME" not in arrays
    assert "TIME" not in times
    np.testing.assert_array_equal(times["array"], time)
    np.testing.assert_array_equal(times["shorter"], np.arange(10))
    np.testing.assert_array_equal(times["labelled"], time + 100)

    # without any time index, the positions are used
    times, _ = determine_time_indices({"array": outcomes[OUTCOME]})
    np.testing.assert_array_equal(times["array"], np.arange(time.size))


def test_lines_use_time_index_of_each_outcome(flu):
    experiments, _, outcomes, labelled, time = flu
    selection = np.arange(20)

    data = {
        OUTCOME: labelled[OUTCOME],
        OTHER: labelled[OTHER][:, ::2].assign_coords(time=time[::2] + 100),
    }
    _, axes = plotting.lines(experiments, data, experiments_to_show=selection)
    np.testing.assert_array_equal(axes[OUTCOME].lines[0].get_xdata(), time)
    np.testing.assert_array_equal(axes[OTHER].lines[0].get_xdata(), time[::2] + 100)

    # the same plot as with numpy arrays and an outcome named TIME
    _, axes_array = plotting.lines(
        experiments,
        {OUTCOME: outcomes[OUTCOME], "TIME": outcomes["TIME"]},
        experiments_to_show=selection,
    )
    assert len(axes[OUTCOME].lines) == len(axes_array[OUTCOME].lines) == 20
    for line, line_array in zip(axes[OUTCOME].lines, axes_array[OUTCOME].lines):
        np.testing.assert_array_equal(line.get_xdata(), line_array.get_xdata())
        np.testing.assert_array_equal(line.get_ydata(), line_array.get_ydata())


def test_plotting_functions_with_dataarrays(flu):
    experiments, _, _, labelled, time = flu
    experiments = experiments.copy()
    experiments["group"] = np.where(np.arange(len(experiments)) % 2, "a", "b")

    plotting.lines(
        experiments,
        labelled,
        outcomes_to_show=OTHER,
        group_by="group",
        density=Density.KDE,
    )
    plotting.envelopes(
        experiments, labelled, group_by="group", density=Density.HIST, fill=True
    )
    plotting.kde_over_time(experiments, labelled, outcomes_to_show=OTHER)
    plotting.multiple_densities(
        experiments,
        labelled,
        outcomes_to_show=OTHER,
        points_in_time=[time[10], time[20], time[30]],
    )
    # without outcomes_to_show, and without an outcome named TIME
    plotting.multiple_densities(experiments, labelled, points_in_time=[time[10]])

    pairs_plotting.pairs_scatter(experiments, labelled)
    pairs_plotting.pairs_lines(experiments, labelled)
    pairs_plotting.pairs_density(experiments, labelled)


def test_prepare_data_selects_experiments(flu):
    experiments, _, _, labelled, time = flu
    mask = np.zeros(len(experiments), dtype=bool)
    mask[:7] = True

    _, outcomes, outcomes_to_show, times, _ = prepare_data(
        experiments, mask, labelled, OUTCOME
    )
    assert outcomes_to_show == [OUTCOME]
    assert type(outcomes[OUTCOME]) is np.ndarray
    assert outcomes[OUTCOME].shape == (7, time.size)
    np.testing.assert_array_equal(times[OUTCOME], time)


# replications


def test_replications_need_to_be_reduced(flu):
    experiments, _, _, labelled, _ = flu
    series = labelled[OUTCOME]
    replicated = xr.concat([series, series * 1.1], dim="replication").transpose(
        "experiment", "replication", "time"
    )

    with pytest.raises(EMAError, match="'replicated' has 3 dimensions"):
        plotting.lines(experiments, {"replicated": replicated})
    with pytest.raises(EMAError, match="'replicated' has 3 dimensions"):
        plotting.envelopes(experiments, {"replicated": replicated})

    # an outcome that is not shown does not matter
    plotting.lines(
        experiments, {"replicated": replicated, OTHER: labelled[OTHER]}, OTHER
    )

    # reduced over the replications, it can be shown
    _, axes = plotting.lines(experiments, {"mean": replicated.mean("replication")})
    np.testing.assert_allclose(axes["mean"].lines[0].get_ydata(), series[0] * 1.05)

    with pytest.raises(ValueError, match="should be 2d"):
        clusterer.calculate_cid(replicated[:10])


def test_clusterer_with_dataarray(flu):
    _, _, outcomes, labelled, _ = flu
    distances_array = clusterer.calculate_cid(outcomes[OTHER][:40])
    distances_labelled = clusterer.calculate_cid(labelled[OTHER][:40])
    np.testing.assert_array_equal(distances_labelled, distances_array)
