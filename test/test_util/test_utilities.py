"""Created on 22 nov. 2012

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
"""

import os
import tarfile
import unittest

import numpy as np
import pandas as pd

from ema_workbench.em_framework.outcomes import (
    ArrayOutcome,
    ScalarOutcome,
    register,
)
from ema_workbench.util.utilities import (
    get_ema_project_home_dir,
    load_results,
    merge_results,
    save_results,
)


def setUpModule():
    global cwd
    cwd = os.getcwd()
    dir_of_module = os.path.dirname(os.path.abspath(__file__))
    register.outcomes = {}  # reset internal dict
    os.chdir(dir_of_module)


def tearDownModule():
    os.chdir(cwd)


class SaveResultsTestCase(unittest.TestCase):
    def test_save_results(self):
        fn = "../data/test.tar.gz"

        # test for 1d
        nr_experiments = 10000
        cases = np.empty(
            nr_experiments, dtype=[("x", float), ("y", int), ("z", bool), ("q", object)]
        )
        experiments = pd.DataFrame.from_records(cases)
        experiments["q"] = experiments["q"].astype("category")
        outcome_q = np.random.rand(nr_experiments, 1)

        outcomes = {ScalarOutcome("q").name: outcome_q}
        results = (experiments, outcomes)

        # test for 2d
        save_results(results, fn)
        os.remove(fn)

        nr_experiments = 10000
        nr_timesteps = 100
        experiments = pd.DataFrame(
            index=np.arange(nr_experiments), columns={"x": float, "y": float}
        )
        outcome_r = np.zeros((nr_experiments, nr_timesteps))

        outcomes = {ArrayOutcome("r").name: outcome_r}
        results = (experiments, outcomes)

        save_results(results, fn)
        os.remove(fn)

        # test for 3d
        # test for very large
        nr_experiments = 10000
        nr_timesteps = 100
        nr_replications = 10
        experiments = pd.DataFrame(
            index=np.arange(nr_experiments), columns={"x": float, "y": float}
        )
        outcome_s = np.zeros((nr_experiments, nr_timesteps, nr_replications))

        outcomes = {ArrayOutcome("s").name: outcome_s}
        results = (experiments, outcomes)

        save_results(results, fn)
        os.remove(fn)


class LoadResultsTestCase(unittest.TestCase):
    def test_load_results(self):
        nr_experiments = 10000

        # test for 2d
        cases = np.empty(
            nr_experiments, dtype=[("x", float), ("y", int), ("z", bool), ("q", object)]
        )
        experiments = pd.DataFrame.from_records(cases)

        experiments["x"] = np.random.rand(nr_experiments)
        experiments["y"] = np.random.randint(0, 10, size=nr_experiments)
        experiments["z"] = np.random.randint(0, 1, size=nr_experiments, dtype=bool)
        experiments["q"] = np.random.randint(0, 10, size=nr_experiments).astype(object)
        experiments["q"] = experiments["q"].astype("category")

        outcome_a = np.zeros((nr_experiments, 1))

        outcomes = {ArrayOutcome("a").name: outcome_a}
        results = (experiments, outcomes)

        save_results(results, "../data/test.tar.gz")
        loaded_experiments, outcomes = load_results("../data/test.tar.gz")

        self.assertTrue(np.all(np.allclose(outcomes["a"], outcome_a)))
        self.assertTrue(np.all(np.allclose(experiments["x"], loaded_experiments["x"])))
        self.assertTrue(np.all(np.allclose(experiments["y"], loaded_experiments["y"])))

        for name, dtype in experiments.dtypes.items():
            self.assertTrue(
                dtype == loaded_experiments[name].dtype,
                msg=f"{name}, {dtype}, {loaded_experiments[name].dtype}",
            )

        os.remove("../data/test.tar.gz")

        # test 3d
        nr_experiments = 1000
        nr_timesteps = 100
        nr_replications = 10
        experiments = pd.DataFrame(
            index=np.arange(nr_experiments), columns={"x": float, "y": float}
        )
        experiments["x"] = np.random.rand(nr_experiments)
        experiments["y"] = np.random.rand(nr_experiments)

        outcome_b = np.zeros((nr_experiments, nr_timesteps, nr_replications))

        outcomes = {ArrayOutcome("b").name: outcome_b}
        results = (experiments, outcomes)

        save_results(results, "../data/test.tar.gz")
        loaded_experiments, outcomes = load_results("../data/test.tar.gz")

        os.remove("../data/test.tar.gz")

        self.assertTrue(np.all(np.allclose(outcomes["b"], outcome_b)))
        self.assertTrue(np.all(np.allclose(experiments["x"], loaded_experiments["x"])))
        self.assertTrue(np.all(np.allclose(experiments["y"], loaded_experiments["y"])))


def test_save_results_compression(tmp_path):
    """Regression test: the compression follows from the extension.

    Previously, results were always saved with gzip, and load_results could
    only read gzip compressed tarballs.
    """
    import pytest  # noqa: PLC0415

    experiments = pd.DataFrame(
        {
            "x": [0.1, 0.2, 0.3],
            "c": pd.Categorical(["a", "b", "a"]),
            "i": [1, 2, 3],
        }
    )
    ScalarOutcome("compression_scalar")
    ArrayOutcome("compression_array")
    outcomes = {
        "compression_scalar": np.array([1.0, 2.0, 3.0]),
        "compression_array": np.arange(24.0).reshape(3, 2, 4),
    }

    for name, magic in [
        ("results.tar.gz", b"\x1f\x8b"),
        ("results.tgz", b"\x1f\x8b"),
        ("results.tar.bz2", b"BZh"),
        ("results.tbz2", b"BZh"),
        ("results.tar.xz", b"\xfd7zXZ"),
        ("results.txz", b"\xfd7zXZ"),
        ("RESULTS.TAR.XZ", b"\xfd7zXZ"),
        ("results.tar", None),
        # an extension that is not recognized uses gzip, as before
        ("results", b"\x1f\x8b"),
        ("results.dat", b"\x1f\x8b"),
    ]:
        file_name = tmp_path / name
        save_results((experiments, outcomes), file_name)

        with open(file_name, "rb") as fh:
            start = fh.read(6)
        if magic is None:
            assert not start.startswith((b"\x1f\x8b", b"BZh", b"\xfd7zXZ"))
            with tarfile.open(file_name, "r:") as archive:
                assert "metadata.json" in archive.getnames()
        else:
            assert start.startswith(magic), name

        loaded_experiments, loaded_outcomes = load_results(file_name)
        pd.testing.assert_frame_equal(experiments, loaded_experiments)
        assert set(loaded_outcomes) == set(outcomes)
        for key, value in outcomes.items():
            np.testing.assert_array_equal(loaded_outcomes[key], value)

        file_name.unlink()

    # load_results detects the compression, regardless of the extension
    save_results((experiments, outcomes), tmp_path / "results.tar.bz2")
    os.rename(tmp_path / "results.tar.bz2", tmp_path / "misnamed.tar.gz")
    loaded_experiments, _ = load_results(tmp_path / "misnamed.tar.gz")
    pd.testing.assert_frame_equal(experiments, loaded_experiments)

    with pytest.raises(FileNotFoundError):
        load_results(tmp_path / "does_not_exist.tar.gz")


def test_determine_compression():
    """Tests for determine_compression."""
    from pathlib import Path  # noqa: PLC0415

    from ema_workbench.util.utilities import determine_compression  # noqa: PLC0415

    assert determine_compression("a.tar.gz") == "gz"
    assert determine_compression("a.TGZ") == "gz"
    assert determine_compression(Path("dir") / "a.tar.bz2") == "bz2"
    assert determine_compression("a.tbz2") == "bz2"
    assert determine_compression("a.tar.xz") == "xz"
    assert determine_compression("a.txz") == "xz"
    assert determine_compression("a.tar") == ""
    assert determine_compression("a.gz") is None
    assert determine_compression("a.zip") is None
    assert determine_compression("a") is None


class ExperimentsToScenariosTestCase(unittest.TestCase):
    pass


class MergeResultsTestCase(unittest.TestCase):
    def test_merge_results(self):
        results1 = load_results("../data/1000 runs scarcity.tar.gz")
        results2 = load_results("../data/1000 runs scarcity.tar.gz")

        n1 = results1[0].shape[0]
        n2 = results2[0].shape[0]

        merged = merge_results(results1, results2)

        self.assertEqual(merged[0].shape[0], n1 + n2)


class ConfigTestCase(unittest.TestCase):
    def test_get_home_dir(self):
        _ = get_ema_project_home_dir()


if __name__ == "__main__":
    unittest.main()
#     testsuite = unittest.TestSuite()
#     testsuite.addTest(LoadResultsTestCase("test_load_results"))
#     unittest.TextTestRunner().run(testsuite)
