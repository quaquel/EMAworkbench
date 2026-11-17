"""Tests for handling optional dependencies.

Missing optional dependencies should not result in warnings when importing
the workbench, because every worker process repeats them when running
experiments in parallel. Instead, using the functionality that needs the
dependency raises an ImportError.
"""

import os
import subprocess
import sys

import pytest


def run_python(code, tmp_path, *args):
    """Run code in a fresh interpreter with all warnings shown."""
    script = tmp_path / "script.py"
    script.write_text(code)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), *sys.path])}
    return subprocess.run(
        [sys.executable, "-W", "always", str(script), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_import_without_warnings(tmp_path):
    """Regression test: importing the workbench and its connectors.

    Previously, importing the connectors warned for every connector of which
    the dependencies are not installed, and it set a global filter that showed
    every ImportWarning once.
    """
    code = """
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import ema_workbench
    import ema_workbench.connectors
    import ema_workbench.analysis

relevant = [w for w in caught if "ema_workbench" in str(w.filename) or "not available" in str(w.message)]
print("WARNINGS", [str(w.message) for w in relevant])

# the global warning filters are not changed
print("FILTERS", [f for f in warnings.filters if f[2] is ImportWarning and f[0] == "once"])
"""
    result = run_python(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "WARNINGS []" in result.stdout, result.stdout
    assert "FILTERS []" in result.stdout, result.stdout


def test_connectors_are_imported_on_first_use():
    """Connectors can still be accessed as attributes of the package."""
    from ema_workbench import connectors

    assert connectors.excel.__name__ == "ema_workbench.connectors.excel"
    with pytest.raises(AttributeError):
        _ = connectors.does_not_exist


def test_ipyparallel_evaluator_without_ipyparallel(tmp_path):
    """Regression test: without ipyparallel, using the evaluator raises.

    Previously, importing the workbench warned if ipyparallel was not
    installed, and IpyparallelEvaluator was None.
    """
    code = """
import sys
import warnings

sys.modules["ipyparallel"] = None  # simulate that ipyparallel is not installed

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    from ema_workbench import IpyparallelEvaluator
print("WARNINGS", [str(w.message) for w in caught if "ipyparallel" in str(w.message)])

try:
    IpyparallelEvaluator(None)
except ImportError as e:
    print("IMPORTERROR", e)
"""
    result = run_python(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "WARNINGS []" in result.stdout, result.stdout
    assert "IMPORTERROR IpyparallelEvaluator requires ipyparallel" in result.stdout


_MODEL = """
import ema_workbench.connectors  # noqa: F401, as a model based on a connector


def model(x=0.0):
    return {"y": x}
"""

_SCRIPT = """
import multiprocessing
import sys

from ema_workbench import Model, MultiprocessingEvaluator, RealParameter, ScalarOutcome

import connector_model

if __name__ == "__main__":
    multiprocessing.set_start_method(sys.argv[1])
    model = Model("connector", function=connector_model.model)
    model.uncertainties = [RealParameter("x", 0, 1)]
    model.outcomes = [ScalarOutcome("y")]
    with MultiprocessingEvaluator(model, n_processes=2) as evaluator:
        evaluator.perform_experiments(4)
"""


@pytest.mark.parametrize("start_method", ["fork", "spawn"])
def test_no_warnings_from_workers(tmp_path, start_method):
    """Regression test for #449: workers repeated the import warnings."""
    import multiprocessing

    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"start method {start_method} not available")

    (tmp_path / "connector_model.py").write_text(_MODEL)
    result = run_python(_SCRIPT, tmp_path, start_method)
    assert result.returncode == 0, result.stderr
    assert "not available" not in result.stderr, result.stderr
    assert "ImportWarning" not in result.stderr, result.stderr
