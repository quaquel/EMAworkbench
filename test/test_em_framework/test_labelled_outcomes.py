"""Tests for outcomes with declared dimensions, which are returned as DataArrays."""

import os
import pickle
import subprocess
import sys

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
    import multiprocessing  # noqa: PLC0415

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
    with pytest.raises(EMAError, match="10 values along 'time'.*have 7 values"):
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
    with pytest.raises(EMAError, match="has 5 time steps.*has 6 values"):
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
