"""Tests for outcomes with declared dimensions, which are returned as DataArrays."""

import os
import pickle
import subprocess
import sys
import warnings

import numpy as np
import pytest
import xarray as xr

from ema_workbench import (
    ArrayOutcome,
    Model,
    RealParameter,
    ReplicatorModel,
    ScalarOutcome,
    SequentialEvaluator,
    TimeSeriesOutcome,
    perform_experiments,
)
from ema_workbench.em_framework.outcomes import Outcome, register
from ema_workbench.util import EMAError, ExperimentError

TIME = np.arange(0, 5, 0.5)


@pytest.fixture(autouse=True)
def restore_register():
    """Outcomes are registered globally by name, so isolate each test."""
    registered = dict(register.outcomes)
    register.outcomes.clear()
    yield
    register.outcomes.clear()
    register.outcomes.update(registered)


def model_function(a=1.0, seed=None):
    """A model that returns time series, the time index, and a grid."""
    noise = 0.0 if seed is None else np.random.default_rng(seed).random()
    return {
        "x": a * TIME + noise,
        "TIME": TIME,
        "grid": np.full((2, 3), a),
        "scalar": a,
    }


def make_model(outcomes, klass=Model, function=model_function, name="model"):
    """Helper function for making a model."""
    model = klass(name, function=function)
    model.uncertainties = [RealParameter("a", 0, 1)]
    model.outcomes = outcomes
    return model


# declaring outcomes


def test_time_series_outcome_declaration():
    outcome = TimeSeriesOutcome("x", time=TIME)
    assert outcome.dims == ("time",)
    np.testing.assert_array_equal(outcome.coords["time"], TIME)
    assert outcome.time_variable is None

    outcome = TimeSeriesOutcome("x", time_variable="TIME")
    assert outcome.dims == ("time",)
    assert outcome.coords == {}
    assert outcome.time_variable == "TIME"

    # without a time index, nothing changes
    outcome = TimeSeriesOutcome("x")
    assert outcome.dims is None
    assert outcome.time_variable is None

    with pytest.raises(ValueError, match="either time or time_variable"):
        TimeSeriesOutcome("x", time=TIME, time_variable="TIME")
    with pytest.raises(ValueError, match="must be a string"):
        TimeSeriesOutcome("x", time_variable=TIME)
    with pytest.raises(ValueError, match="one dimensional"):
        TimeSeriesOutcome("x", time=np.zeros((2, 2)))


def test_array_outcome_declaration():
    outcome = ArrayOutcome("grid", dims=("row", "col"), coords={"col": [10, 20, 30]})
    assert outcome.dims == ("row", "col")
    assert list(outcome.coords) == ["col"]

    assert ArrayOutcome("grid", dims="row").dims == ("row",)
    assert ArrayOutcome("grid").dims is None

    with pytest.raises(ValueError, match="together with dims"):
        ArrayOutcome("grid", coords={"col": [1, 2]})
    with pytest.raises(ValueError, match="not in dims"):
        ArrayOutcome("grid", dims=("row",), coords={"col": [1, 2]})
    with pytest.raises(ValueError, match="must be unique"):
        ArrayOutcome("grid", dims=("row", "row"))
    with pytest.raises(ValueError, match="added by the workbench"):
        ArrayOutcome("grid", dims=("experiment", "row"))
    with pytest.raises(ValueError, match="added by the workbench"):
        ArrayOutcome("grid", dims=("replication", "row"))
    with pytest.raises(ValueError, match="string or a tuple of strings"):
        ArrayOutcome("grid", dims=(1, 2))


def test_outcome_equality_with_coords():
    assert TimeSeriesOutcome("x", time=TIME) == TimeSeriesOutcome("x", time=TIME)
    assert TimeSeriesOutcome("x", time=TIME) != TimeSeriesOutcome("x", time=TIME + 1)
    assert TimeSeriesOutcome("x", time=TIME) != TimeSeriesOutcome("x")
    assert TimeSeriesOutcome("x", time_variable="TIME") != TimeSeriesOutcome("x")


def test_output_variables_include_time_variable():
    model = make_model(
        [
            TimeSeriesOutcome("x", time_variable="TIME"),
            TimeSeriesOutcome("y", variable_name="x", time_variable="TIME"),
        ]
    )
    assert model.output_variables == ["x", "x", "TIME"]


# results of perform_experiments


def test_time_series_results():
    model = make_model(
        [
            TimeSeriesOutcome("declared", variable_name="x", time=TIME),
            TimeSeriesOutcome("from_model", variable_name="x", time_variable="TIME"),
            TimeSeriesOutcome("plain", variable_name="x"),
            ScalarOutcome("scalar"),
        ]
    )
    experiments, outcomes = perform_experiments(model, 4)

    for name in ("declared", "from_model"):
        outcome = outcomes[name]
        assert isinstance(outcome, xr.DataArray)
        assert outcome.dims == ("experiment", "time")
        assert outcome.shape == (4, TIME.size)
        assert outcome.name == name
        np.testing.assert_array_equal(outcome.time.values, TIME)
        # experiments are positional
        assert "experiment" not in outcome.coords
        np.testing.assert_allclose(
            outcome.values, experiments["a"].values[:, np.newaxis] * TIME
        )

    # the time index is not an outcome
    assert set(outcomes) == {"declared", "from_model", "plain", "scalar"}

    # outcomes that are not declared with a time index are numpy arrays
    assert type(outcomes["plain"]) is np.ndarray
    assert type(outcomes["scalar"]) is np.ndarray
    np.testing.assert_array_equal(outcomes["plain"], outcomes["declared"].values)

    # numpy style indexing still works, and selecting on time is possible
    declared = outcomes["declared"]
    np.testing.assert_array_equal(declared[:, -1], outcomes["plain"][:, -1])
    np.testing.assert_array_equal(declared.sel(time=2.5), outcomes["plain"][:, 5])


def test_array_outcome_results():
    model = make_model(
        [ArrayOutcome("grid", dims=("row", "col"), coords={"col": [10, 20, 30]})]
    )
    _, outcomes = perform_experiments(model, 3)

    grid = outcomes["grid"]
    assert grid.dims == ("experiment", "row", "col")
    assert grid.shape == (3, 2, 3)
    assert list(grid.coords) == ["col"]
    np.testing.assert_array_equal(grid.col.values, [10, 20, 30])


def test_time_series_with_replications():
    """Time series for each replication have a replication dimension."""
    model = make_model(
        [
            TimeSeriesOutcome("declared", variable_name="x", time=TIME),
            TimeSeriesOutcome("from_model", variable_name="x", time_variable="TIME"),
            TimeSeriesOutcome(
                "mean",
                variable_name="x",
                time_variable="TIME",
                function=lambda values: np.mean(values, axis=0),
            ),
            TimeSeriesOutcome("plain", variable_name="x"),
        ],
        klass=ReplicatorModel,
    )
    model.replications = [{"seed": 1}, {"seed": 2}, {"seed": 3}]
    _, outcomes = perform_experiments(model, 4)

    for name in ("declared", "from_model"):
        outcome = outcomes[name]
        assert outcome.dims == ("experiment", "replication", "time")
        assert outcome.shape == (4, 3, TIME.size)
        assert list(outcome.coords) == ["time"]
        np.testing.assert_array_equal(outcome.time.values, TIME)
        np.testing.assert_array_equal(outcome.values, outcomes["plain"])

    # the replications are different, and can be aggregated by name
    declared = outcomes["declared"]
    assert not np.allclose(declared[:, 0], declared[:, 1])
    np.testing.assert_allclose(declared.mean("replication"), outcomes["mean"])

    # aggregated by the function of the outcome
    assert outcomes["mean"].dims == ("experiment", "time")
    assert type(outcomes["plain"]) is np.ndarray


def test_failed_experiments_are_nan():
    def function(a=1.0):
        if a > 0.5:
            raise ExperimentError("failed", {"a": a})
        return {"x": a * TIME}

    model = make_model([TimeSeriesOutcome("x", time=TIME)], function=function)
    experiments, outcomes = perform_experiments(model, 20)

    failed = (experiments["a"] > 0.5).values
    assert 0 < failed.sum() < 20
    assert isinstance(outcomes["x"], xr.DataArray)
    assert np.isnan(outcomes["x"].values[failed]).all()
    assert not np.isnan(outcomes["x"].values[~failed]).any()


def test_outcome_of_experiment_can_be_pickled():
    """The outcomes of an experiment are sent from the workers to the main process."""
    outcome = TimeSeriesOutcome("x", time_variable="TIME")
    value = outcome.attach_time(TIME * 2, TIME)
    restored = pickle.loads(pickle.dumps(value))  # noqa: S301
    xr.testing.assert_identical(value, restored)


_PARALLEL_MODEL = """
import numpy as np

TIME = np.arange(0, 5, 0.5)


def model(a=1.0):
    return {"x": a * TIME, "TIME": TIME}
"""

_PARALLEL_SCRIPT = """
import multiprocessing
import sys
import warnings

import numpy as np

from ema_workbench import Model, MultiprocessingEvaluator, RealParameter, TimeSeriesOutcome

import parallel_model

if __name__ == "__main__":
    multiprocessing.set_start_method(sys.argv[1])
    model = Model("model", function=parallel_model.model)
    model.uncertainties = [RealParameter("a", 0, 1)]
    model.outcomes = [TimeSeriesOutcome("x", time_variable="TIME")]
    with MultiprocessingEvaluator(model, n_processes=2) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(6)
    x = outcomes["x"]
    assert x.dims == ("experiment", "time"), x.dims
    assert (x.time.values == parallel_model.TIME).all()
    expected = experiments["a"].values[:, np.newaxis] * parallel_model.TIME
    assert np.allclose(x.values, expected)
    print("RESULT OK")
"""


@pytest.mark.parametrize("start_method", ["fork", "spawn"])
def test_time_variable_with_multiprocessing(tmp_path, start_method):
    import multiprocessing

    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"start method {start_method} not available")

    (tmp_path / "parallel_model.py").write_text(_PARALLEL_MODEL)
    (tmp_path / "script.py").write_text(_PARALLEL_SCRIPT)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), *sys.path])}
    result = subprocess.run(
        [sys.executable, "script.py", start_method],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "RESULT OK" in result.stdout


# problems are reported at the first experiment


def test_time_index_of_wrong_length():
    calls = []

    def function(a=1.0):
        calls.append(a)
        return {"x": a * TIME}

    model = make_model([TimeSeriesOutcome("x", time=np.arange(7))], function=function)
    with pytest.raises(EMAError, match=r"10 values along 'time'.*have 7 values"):
        perform_experiments(model, 5)
    # reported at the first experiment, not after running all of them
    assert len(calls) == 1


def test_time_index_differs_between_experiments():
    def varying_index(a=1.0):
        return {"x": np.zeros(5), "TIME": np.arange(5) + a}

    def varying_length(a=1.0):
        n = 5 if a < 0.5 else 6
        return {"x": np.zeros(n), "TIME": np.arange(n)}

    for function in (varying_index, varying_length):
        model = make_model(
            [TimeSeriesOutcome("x", time_variable="TIME")], function=function
        )
        with pytest.raises(EMAError, match="time index of outcome 'x' differs"):
            perform_experiments(model, 10)


def test_time_index_differs_between_replications():
    def function(a=1.0, shift=0):
        return {"x": np.zeros(5), "TIME": np.arange(5) + shift}

    model = make_model(
        [TimeSeriesOutcome("x", time_variable="TIME")],
        klass=ReplicatorModel,
        function=function,
    )
    model.replications = [{"shift": 0}, {"shift": 1}]
    with pytest.raises(EMAError, match="differs between replications"):
        perform_experiments(model, 2)


def test_time_index_does_not_match_time_series():
    def function(a=1.0):
        return {"x": np.zeros(5), "TIME": np.arange(6)}

    model = make_model(
        [TimeSeriesOutcome("x", time_variable="TIME")], function=function
    )
    with pytest.raises(EMAError, match=r"has 5 time steps.*has 6 values"):
        perform_experiments(model, 2)


def test_extra_dimension_without_replications():
    """Only a model with replications adds a dimension."""
    model = make_model([TimeSeriesOutcome("grid", time=np.arange(3))])
    with pytest.raises(EMAError, match="should be 1, but is 2"):
        perform_experiments(model, 2)


def test_same_outcome_in_several_models():
    def make(name, time):
        return make_model([TimeSeriesOutcome("x", time=time)], name=name)

    _, outcomes = perform_experiments([make("first", TIME), make("second", TIME)], 3)
    assert outcomes["x"].shape == (6, TIME.size)

    with pytest.raises(EMAError, match="different dimensions or coordinates"):
        perform_experiments([make("first", TIME), make("second", TIME + 1)], 3)

    plain = make_model([TimeSeriesOutcome("x")], name="plain")
    with pytest.raises(EMAError, match="different dimensions or coordinates"):
        perform_experiments([make("first", TIME), plain], 3)


# optimization


def test_robustness_functions_get_numpy_arrays(tmp_path):
    """Robustness functions keep working with numpy arrays."""
    received = []

    def robustness(values):
        received.append(type(values))
        return np.mean(values[:, -1])

    def function(a=1.0, b=1.0):
        return {"x": a * b * TIME}

    model = make_model([TimeSeriesOutcome("x", time=TIME)], function=function)
    model.levers = [RealParameter("b", 0, 1)]

    robustness_functions = [
        ScalarOutcome(
            "mean_end",
            kind=Outcome.MAXIMIZE,
            variable_name="x",
            function=robustness,
        )
    ]
    with SequentialEvaluator(model) as evaluator:
        archive, _ = evaluator.robust_optimize(
            robustness_functions,
            3,
            nfe=20,
            epsilons=[0.1],
            population_size=5,
            filename="archive.tar.gz",
            directory=str(tmp_path),
        )
    assert set(received) == {np.ndarray}
    assert archive["mean_end"].dtype == float


# saving, loading and merging


def _replicated_results(n_experiments=4):
    """Results with DataArrays, including replications, and numpy arrays."""
    model = make_model(
        [
            TimeSeriesOutcome("x", time_variable="TIME"),
            TimeSeriesOutcome("plain", variable_name="x"),
            ArrayOutcome(
                "mean_grid",
                variable_name="grid",
                dims=("row", "col"),
                coords={"col": [10, 20, 30]},
                function=lambda values: np.mean(values, axis=0),
            ),
            ScalarOutcome("scalar", function=np.mean),
        ],
        klass=ReplicatorModel,
    )
    model.replications = [{"seed": 1}, {"seed": 2}, {"seed": 3}]
    return perform_experiments(model, n_experiments)


def test_save_and_load_results(tmp_path):
    """DataArrays keep their dimensions and coordinates when saved."""
    import tarfile

    from ema_workbench import load_results, save_results

    experiments, outcomes = _replicated_results()
    file_name = tmp_path / "results.tar.gz"
    save_results((experiments, outcomes), file_name)

    with tarfile.open(file_name) as archive:
        names = set(archive.getnames())
    assert {"x.nc", "mean_grid.nc", "plain.npy", "scalar.cls"} <= names

    loaded_experiments, loaded = load_results(file_name)
    assert set(loaded) == set(outcomes)
    assert len(loaded_experiments) == len(experiments)

    # identical includes name, dimensions, coordinates, and values
    xr.testing.assert_identical(loaded["x"], outcomes["x"])
    xr.testing.assert_identical(loaded["mean_grid"], outcomes["mean_grid"])
    assert loaded["x"].dims == ("experiment", "replication", "time")
    assert loaded["x"].dtype == outcomes["x"].dtype

    assert type(loaded["plain"]) is np.ndarray
    np.testing.assert_array_equal(loaded["plain"], outcomes["plain"])
    np.testing.assert_allclose(loaded["scalar"], outcomes["scalar"])

    # loaded results can be saved again
    save_results((loaded_experiments, loaded), tmp_path / "again.tar.gz")
    _, again = load_results(tmp_path / "again.tar.gz")
    xr.testing.assert_identical(again["x"], outcomes["x"])


def test_save_and_load_time_series_without_replications(tmp_path):
    from ema_workbench import load_results, save_results

    model = make_model([TimeSeriesOutcome("x", time=TIME)])
    experiments, outcomes = perform_experiments(model, 3)
    save_results((experiments, outcomes), tmp_path / "results.tar.gz")
    _, loaded = load_results(tmp_path / "results.tar.gz")
    xr.testing.assert_identical(loaded["x"], outcomes["x"])


def test_merge_results():
    from ema_workbench.util.utilities import merge_results

    results1 = _replicated_results(4)
    results2 = _replicated_results(2)
    experiments, outcomes = merge_results(results1, results2)

    assert len(experiments) == 6
    x = outcomes["x"]
    assert isinstance(x, xr.DataArray)
    assert x.dims == ("experiment", "replication", "time")
    assert x.shape == (6, 3, TIME.size)
    np.testing.assert_array_equal(x.time.values, TIME)
    np.testing.assert_array_equal(x.values[:4], results1[1]["x"].values)
    np.testing.assert_array_equal(x.values[4:], results2[1]["x"].values)
    # experiments stay positional
    assert "experiment" not in x.coords

    assert type(outcomes["plain"]) is np.ndarray
    assert outcomes["plain"].shape == (6, 3, TIME.size)


def test_merge_results_with_different_index():
    from ema_workbench.util.utilities import merge_results

    def results(outcome):
        return perform_experiments(make_model([outcome]), 2)

    declared = results(TimeSeriesOutcome("x", time=TIME))

    with pytest.raises(EMAError, match="different time index"):
        merge_results(declared, results(TimeSeriesOutcome("x", time=TIME + 1)))
    with pytest.raises(EMAError, match="DataArray in one of the results"):
        merge_results(declared, results(TimeSeriesOutcome("x")))

    replicated = make_model([TimeSeriesOutcome("x", time=TIME)], klass=ReplicatorModel)
    replicated.replications = 2
    with pytest.raises(EMAError, match="has dimensions"):
        merge_results(declared, perform_experiments(replicated, 2))


# deprecation of time series outcomes without a time index


def test_time_series_outcome_without_time_index_is_deprecated():
    with pytest.warns(DeprecationWarning, match="'x' has no time index") as record:
        TimeSeriesOutcome("x")
    # the warning points to the code that declares the outcome
    assert record[0].filename == __file__

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        TimeSeriesOutcome("x", time=TIME)
        TimeSeriesOutcome("x", time_variable="TIME")
        # other outcomes are not affected
        ArrayOutcome("grid")
        ScalarOutcome("scalar")


def test_no_deprecation_warning_from_the_workbench_itself(tmp_path):
    """The user can only act on the warning for outcomes they declare."""
    import pandas as pd

    from ema_workbench import load_results, save_results
    from ema_workbench.em_framework.outcomes import create_outcomes

    with pytest.warns(DeprecationWarning, match="has no time index"):
        model = make_model([TimeSeriesOutcome("plain", variable_name="x")])
    results = perform_experiments(model, 2)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.warns(UserWarning, match="Timeseries outcome seralization"):
            save_results(results, tmp_path / "results.tar.gz")
        register.outcomes.clear()
        _, outcomes = load_results(tmp_path / "results.tar.gz")
        assert type(outcomes["plain"]) is np.ndarray

        create_outcomes(
            pd.DataFrame({"name": ["a", "b"], "type": ["scalar", "timeseries"]})
        )
