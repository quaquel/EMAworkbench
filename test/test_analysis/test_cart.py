"""Created on May 22, 2015

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
"""

import unittest

import numpy as np
import pandas as pd

from ema_workbench.analysis import cart
from ema_workbench.analysis.scenario_discovery_util import RuleInductionType
from test import utilities


def flu_classify(data):
    # get the output for deceased population
    result = data["deceased_population_region_1"]

    # make an empty array of length equal to number of cases
    classes = np.zeros(result.shape[0])

    # if deceased population is higher then 1.000.000 people, classify as 1
    classes[result[:, -1] > 1000000] = 1

    return classes


def scarcity_classify(outcomes):
    outcome = outcomes["relative_market_price"]
    change = np.abs(outcome[:, 1::] - outcome[:, 0:-1])

    neg_change = np.min(change, axis=1)
    pos_change = np.max(change, axis=1)

    logical = (neg_change > -0.6) & (pos_change > 0.6)

    classes = np.zeros(outcome.shape[0])
    classes[logical] = 1

    return classes


class CartTestCase(unittest.TestCase):
    def test_boxes(self):
        np.random.seed(42)
        x = pd.DataFrame(np.random.rand(1000, 2), columns=["a", "b"])
        y = (x.a > 0.5) & (x.b < 0.5)
        alg = cart.CART(x, y, mode=RuleInductionType.BINARY)
        alg.build_tree()

        boxes = alg.boxes

        self.assertEqual(len(boxes), 3)

    def test_boxes_mixed_dtypes(self):
        # integer and categorical inputs: boxes must partition the data like the
        # leaves of the tree (previously integers raised and categorical splits
        # were only applied on one side, so boxes overlapped)
        rng = np.random.default_rng(0)
        n = 1000
        x = pd.DataFrame(
            {
                "a": rng.random(n),
                "i": rng.integers(0, 10, n),
                "c": pd.Categorical(rng.choice(["p", "q", "r"], n)),
            }
        )
        y = (((x.c == "q") & (x.a > 0.5)) | (x.i > 6)).astype(int).values
        alg = cart.CART(x, y)
        alg.build_tree()

        from ema_workbench.analysis import scenario_discovery_util as sdutil

        covered = np.zeros(n, dtype=int)
        for box in alg.boxes:
            covered += np.asarray(sdutil._in_box(x, box))
        np.testing.assert_array_equal(covered, 1)
        self.assertEqual(len(alg.boxes), (alg.clf.tree_.children_left == -1).sum())

    def test_stats(self):
        x = pd.DataFrame([(0, 1, 2), (2, 5, 6), (3, 2, 1)], columns=["a", "b", "c"])

        box = pd.DataFrame([(0, 1, 1), (3, 5, 3)], columns=["a", "b", "c"])

        y = np.array([0, 1, 1])
        alg = cart.CART(x, y, mode=RuleInductionType.BINARY)
        alg._boxes = [box]
        alg.clf = "something"
        stats = alg.stats[0]

        self.assertEqual(stats["coverage"], 0.5)
        self.assertEqual(stats["density"], 0.5)
        self.assertEqual(stats["res_dim"], 1)
        self.assertEqual(stats["mass"], 2 / 3)

        y = np.array([0, 1, 2])
        alg = cart.CART(x, y, mode=RuleInductionType.REGRESSION)
        alg._boxes = [box]
        alg.clf = "something"
        stats = alg.stats[0]

        self.assertEqual(stats["mean"], 1)
        self.assertEqual(stats["res_dim"], 1)
        self.assertEqual(stats["mass"], 2 / 3)

        y = np.array([0, 1, 2])
        alg = cart.CART(x, y, mode=RuleInductionType.CLASSIFICATION)
        alg._boxes = [box]
        alg.clf = "something"
        stats = alg.stats[0]

        self.assertEqual(stats["gini"], 0.5)
        self.assertAlmostEqual(stats["purity"], 1 - np.log(2) / np.log(3))
        self.assertEqual(stats["box_composition"], [1, 0, 1])
        self.assertEqual(stats["res_dim"], 1)
        self.assertEqual(stats["mass"], 2 / 3)

        self.assertEqual(stats, alg.stats[0])

    def test_stats_to_dataframe(self):
        x, outcomes = utilities.load_flu_data()

        y = flu_classify(outcomes)
        alg = cart.CART(x, y, mode=RuleInductionType.BINARY)
        alg.build_tree()
        stats = alg.stats_to_dataframe()

        y = outcomes["deceased_population_region_1"][:, -1]
        alg = cart.CART(x, y, mode=RuleInductionType.REGRESSION)
        alg.build_tree()
        stats = alg.stats_to_dataframe()

        y = np.random.randint(1, 5, y.shape[0])
        alg = cart.CART(x, y, mode=RuleInductionType.CLASSIFICATION)
        alg.build_tree()
        stats = alg.stats_to_dataframe()
        print(stats)

    def test_extract_rules(self):
        rng = np.random.default_rng(0)
        n = 2000
        x = pd.DataFrame(
            {
                "a": rng.random(n),
                "i": rng.integers(0, 10, n),
                "c": pd.Categorical(rng.choice(["p", "q", "r"], n)),
            }
        )
        y = (((x.c == "q") & (x.a > 0.5)) | (x.i > 6)).astype(int).values
        alg = cart.CART(x, y)
        alg.build_tree()

        rules = alg.extract_rules()
        self.assertEqual(set(rules), {0, 1})
        predicted = alg.clf.predict(alg._x)
        for klass, rule in rules.items():
            np.testing.assert_array_equal(rule.evaluate(x), predicted == klass)

        # classification mode: one rule per class, the rules partition the data
        y = rng.integers(0, 3, n)
        alg = cart.CART(x, y, mode=RuleInductionType.CLASSIFICATION)
        alg.build_tree()
        rules = alg.extract_rules(tolerance=0.01)
        self.assertEqual(set(rules), {0, 1, 2})

        # regression: no classes, no rules
        alg = cart.CART(x, rng.random(n), mode=RuleInductionType.REGRESSION)
        alg.build_tree()
        with self.assertRaises(cart.EMAError):
            alg.extract_rules()

    def test_build_tree(self):
        x, outcomes = utilities.load_flu_data()

        y = flu_classify(outcomes)
        alg = cart.CART(x, y, mass_min=0.05)
        alg.build_tree()
        self.assertTrue(isinstance(alg.clf, cart.tree.DecisionTreeClassifier))

        y = outcomes["deceased_population_region_1"][:, -1]
        alg = cart.CART(x, y, mode=RuleInductionType.REGRESSION, mass_min=0.05)
        alg.build_tree()
        self.assertTrue(isinstance(alg.clf, cart.tree.DecisionTreeRegressor))

    def test_show_pairs_scatter(self):
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(1)
        n = 1000
        x = pd.DataFrame(
            {
                "a": rng.random(n),
                "b": rng.random(n),
                "c": pd.Categorical(rng.choice(["p", "q", "r"], n)),
            }
        )
        y = ((x.a > 0.5) & (x.c == "q")).astype(int).values
        alg = cart.CART(x, y)
        alg.build_tree()

        # default: the region where the tree predicts the cases of interest
        grid = alg.show_pairs_scatter()
        self.assertEqual(list(grid.x_vars), ["a", "c"])
        grid = alg.show_pairs_scatter(i=0, diag="hist", upper=None, lower="scatter")
        grid = alg.show_pairs_scatter(klass=1, dims=["a", "b"])
        self.assertEqual(list(grid.x_vars), ["a", "b"])

        with self.assertRaises(ValueError):
            alg.show_pairs_scatter(i=0, klass=1)
        with self.assertRaises(ValueError):
            alg.show_pairs_scatter(klass=5)
        with self.assertRaises(ValueError):
            alg.show_pairs_scatter(diag="wrong value")

        # classification: a class or a box index is required
        y = np.where(x.a > 0.5, 2, np.where(x.c == "q", 1, 0))
        alg = cart.CART(x, y, mode=RuleInductionType.CLASSIFICATION)
        alg.build_tree()
        grid = alg.show_pairs_scatter(klass=2, diag="hist", lower="scatter")
        self.assertEqual(list(grid.x_vars), ["a"])
        with self.assertRaises(ValueError):
            alg.show_pairs_scatter()

        # regression: a box index is required
        alg = cart.CART(x, x.a.values, mode=RuleInductionType.REGRESSION)
        alg.build_tree()
        with self.assertRaises(cart.EMAError):
            alg.show_pairs_scatter()
        plt.close("all")

    #     def test_show_tree(self):
    #         results = utilities.load_flu_data()
    #
    #         alg = cart.setup_cart(results, flu_classify,
    #                               mass_min=0.05)
    #         alg.build_tree()
    #
    #         fig = alg.show_tree(mplfig=True)
    #         bytestream = alg.show_tree(mplfig=False)
    #
    #         self.assertTrue(isinstance(fig, mpl.figure.Figure))
    #         self.assertTrue(isinstance(bytestream, bytes))

    def test_uninformative_columns_dropped(self):
        rng = np.random.default_rng(0)
        x = pd.DataFrame(
            {
                "scenario": np.arange(100),
                "a": rng.random(100),
                "constant": 2.0,
                "model": pd.Categorical(["m"] * 100),
            }
        )
        alg = cart.CART(x, (x.a > 0.5).astype(int).values)
        self.assertEqual(alg.x.columns.tolist(), ["a"])
        self.assertEqual(alg.feature_names, ["a"])


if __name__ == "__main__":
    unittest.main()

#     suite = unittest.TestSuite()
#     suite.addTest(PrimTestCase("test_write_boxes_to_stdout"))
#     unittest.TextTestRunner().run(suite)
