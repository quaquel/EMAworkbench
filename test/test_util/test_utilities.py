"""Created on 22 nov. 2012

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
"""

import os
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


def test_save_results_undeclared_outcome(tmp_path):
    """Regression test: saving outcomes that were never declared as an Outcome.

    Previously, this raised a KeyError, for example when a derived outcome
    was added to the outcomes dict before saving.
    """
    experiments = pd.DataFrame({"x": [0.1, 0.2, 0.3]})
    outcomes = {
        "undeclared_scalar": np.array([1.0, 2.0, 3.0]),
        "undeclared_timeseries": np.arange(12.0).reshape(3, 4),
        "undeclared_array": np.arange(24.0).reshape(3, 2, 4),
    }
    for key in outcomes:
        assert key not in register.outcomes

    file_name = tmp_path / "results.tar.gz"
    save_results((experiments, outcomes), file_name)
    loaded_experiments, loaded_outcomes = load_results(file_name)

    pd.testing.assert_frame_equal(experiments, loaded_experiments)
    assert set(loaded_outcomes) == set(outcomes)
    for key, value in outcomes.items():
        assert loaded_outcomes[key].shape == value.shape
        np.testing.assert_allclose(loaded_outcomes[key], value)


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
