"""Created on May 22, 2015

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
"""

import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ema_workbench.analysis import scenario_discovery_util as sdutil
from ema_workbench.util import EMAError


class ScenarioDiscoveryUtilTestCase(unittest.TestCase):
    def test_get_sorted_box_lims(self):
        x = pd.DataFrame([(0, 1, 2), (2, 5, 6), (3, 2, 1)], columns=["a", "b", "c"])

        box_init = sdutil._make_box(x)

        box_lim = pd.DataFrame([(0, 1, 1), (2, 5, 2)], columns=["a", "b", "c"])

        _, uncs = sdutil._get_sorted_box_lims([box_lim], box_init)

        self.assertEqual(uncs, ["c", "a"])

    def test_in_box(self):
        x = pd.DataFrame(
            [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)], columns=["a"]
        )
        boxlim = pd.DataFrame([(1,), (8,)], columns=["a"])
        correct_result = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=int).T
        logical = sdutil._in_box(x, boxlim)
        result = x.loc[logical]
        self.assertTrue(np.all(correct_result == result.values))

        x = pd.DataFrame(
            [
                (0, 0),
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 5),
                (6, 6),
                (7, 7),
                (8, 8),
                (9, 9),
            ],
            columns=["a", "b"],
        )
        boxlim = pd.DataFrame([(1, 0), (8, 7)], columns=["a", "b"])
        correct_result = np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=int).T
        logical = sdutil._in_box(x, boxlim)
        result = x.loc[logical]
        self.assertTrue(np.all(correct_result == result))

        x = pd.DataFrame(
            [
                (0.1, 0, "a"),
                (1.1, 1, "a"),
                (2.1, 2, "b"),
                (3.1, 3, "b"),
                (4.1, 4, "c"),
                (5.1, 5, "c"),
                (6.1, 6, "d"),
                (7.1, 7, "d"),
                (8.1, 8, "e"),
                (9.1, 9, "e"),
            ],
            columns=["a", "b", "c"],
        )
        boxlim = pd.DataFrame(
            [(1.2, 0, {"a", "b"}), (8.0, 7, {"a", "b"})], columns=["a", "b", "c"]
        )
        x["c"] = x["c"].astype("category")

        correct_result = x.loc[[2, 3], :]
        logical = sdutil._in_box(x, boxlim)
        result = x.loc[logical]
        self.assertTrue(np.all(correct_result == result))

        boxlim = pd.DataFrame(
            [(0.1, 0, {"a", "b", "c", "d", "e"}), (9.1, 9, {"a", "b", "c", "d", "e"})],
            columns=["a", "b", "c"],
        )
        correct_result = x.loc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], :]
        logical = sdutil._in_box(x, boxlim)
        result = x.loc[logical]
        self.assertTrue(np.all(correct_result == result))

        x = pd.DataFrame(
            [
                (0.1, 0, "a", True),
                (1.1, 1, "a", True),
                (2.1, 2, "b", True),
                (3.1, 3, "b", True),
                (4.1, 4, "c", False),
                (5.1, 5, "c", False),
                (6.1, 6, "d", False),
                (7.1, 7, "d", False),
                (8.1, 8, "e", False),
                (9.1, 9, "e", False),
            ],
            columns=["a", "b", "c", "d"],
        )
        boxlim = pd.DataFrame(
            [(1.2, 0, {"a", "b", "c"}, True), (8.0, 7, {"a", "b", "c"}, True)],
            columns=["a", "b", "c", "d"],
        )
        x["c"] = x["c"].astype("category")

        correct_result = x.loc[[2, 3], :]
        logical = sdutil._in_box(x, boxlim)
        result = x.loc[logical]
        self.assertTrue(np.all(correct_result == result))

    def test_make_box(self):
        x = pd.DataFrame([(0, 1, 2), (2, 5, 6), (3, 2, 1)], columns=["a", "b", "c"])

        box_lims = sdutil._make_box(x)

        # some test on the box
        self.assertEqual(np.min(box_lims["a"]), 0, "min a fails")
        self.assertEqual(np.max(box_lims["a"]), 3, "max a fails")

        self.assertEqual(np.min(box_lims["b"]), 1, "min b fails")
        self.assertEqual(np.max(box_lims["b"]), 5, "max c fails")

        self.assertEqual(np.min(box_lims["c"]), 1, "min c fails")
        self.assertEqual(np.max(box_lims["c"]), 6, "max c fails")

    def test_normalize(self):
        x = pd.DataFrame([(0, 1, 2), (2, 5, 6), (3, 2, 1)], columns=["a", "b", "c"])

        box_init = sdutil._make_box(x)

        box_lim = pd.DataFrame([(0, 1, 1), (2, 5, 2)], columns=["a", "b", "c"])
        uncs = box_lim.columns.values.tolist()
        normalized = sdutil._normalize(box_lim, box_init, uncs)

        for i, lims in enumerate([(0, 2 / 3), (0, 1), (0, 0.2)]):
            lower, upper = lims
            self.assertAlmostEqual(
                normalized[i, 0], lower, msg="lower unequal for " + uncs[i]
            )
            self.assertAlmostEqual(
                normalized[i, 1], upper, msg="upper unequal for " + uncs[i]
            )

    def test_determine_restricted_dims(self):
        x = np.random.rand(5, 2)
        x = pd.DataFrame(x, columns=["a", "b"])

        # all dimensions the same
        box_init = sdutil._make_box(x)
        u = sdutil._determine_restricted_dims(box_init, box_init)

        self.assertEqual(list(u), [])

        # dimensions 1 different and dimension 2 the same
        b = pd.DataFrame([(1, 1), (0, 1)], columns=["a", "b"])
        u = sdutil._determine_restricted_dims(b, box_init)

        self.assertEqual(list(u), ["a", "b"])

    def test_determine_nr_restricted_dims(self):
        x = np.random.rand(5, 2)
        x = pd.DataFrame(x, columns=["a", "b"])

        # all dimensions the same
        box_init = sdutil._make_box(x)
        n = sdutil._determine_nr_restricted_dims(box_init, box_init)

        self.assertEqual(n, 0)

        # dimensions 1 different and dimension 2 the same
        b = pd.DataFrame([(1, 1), (0, 1)], columns=["a", "b"])
        n = sdutil._determine_nr_restricted_dims(b, box_init)
        self.assertEqual(n, 2)

    def test_compare(self):
        # all dimensions the same
        a = np.array([(0, 1), (0, 1)], dtype=[("a", float), ("b", float)])
        b = np.array([(0, 1), (0, 1)], dtype=[("a", float), ("b", float)])

        self.assertTrue(np.all(sdutil._compare(a, b)))

        # all dimensions different
        a = np.array([(0, 1), (0, 1)], dtype=[("a", float), ("b", float)])
        b = np.array([(1, 1), (0, 0)], dtype=[("a", float), ("b", float)])
        test = sdutil._compare(a, b) == False
        self.assertTrue(np.all(test))

        # dimensions 1 different and dimension 2 the same
        a = np.array([(0, 1), (0, 1)], dtype=[("a", float), ("b", float)])
        b = np.array([(1, 1), (0, 1)], dtype=[("a", float), ("b", float)])
        test = sdutil._compare(a, b)
        test = (test[0] == False) & (test[1] == True)
        self.assertTrue(test)

    def test_plot_box(self):
        x = pd.DataFrame(
            [
                [0.1, 0, "a"],
                [0.2, 1, "b"],
                [0.3, 2, "a"],
                [0.4, 3, "b"],
                [0.5, 4, "a"],
                [0.6, 5, "a"],
                [0.7, 6, "b"],
                [0.8, 7, "a"],
                [0.9, 8, "b"],
                [1.0, 9, "a"],
            ],
            columns=["a", "b", "c"],
        )
        y = np.zeros((x.shape[0],), dtype=int)
        y[(x.a > 0.5) & (x.c != "a")] = 1

        x["c"] = x["c"].astype("category")

        box_init = sdutil._make_box(x)
        boxlim = box_init.copy()
        boxlim.a = [0.5, 1.0]
        boxlim.c = [set("b")] * 2
        restricted_dims = ["a", "c"]

        qp_values = {"a": [0.05, 0.9], "c": [0.05, -1]}

        fig, ax = plt.subplots()
        sdutil.plot_box(
            boxlim,
            qp_values,
            box_init,
            restricted_dims,
            {"coverage": 1, "density": 1},
            ax,
        )
        plt.draw()
        plt.close("all")

    def test_plot_pairwise_scatter(self):
        x = pd.DataFrame(
            [
                [0.1, 0, "a"],
                [0.2, 1, "b"],
                [0.3, 2, "a"],
                [0.4, 3, "b"],
                [0.5, 4, "a"],
                [0.6, 5, "a"],
                [0.7, 6, "b"],
                [0.8, 7, "a"],
                [0.9, 8, "b"],
                [1.0, 9, "a"],
            ],
            columns=["a", "b", "c"],
        )
        y = np.zeros((x.shape[0],), dtype=int)
        y[(x.a > 0.5) & (x.c != "a")] = 1

        x.loc[6, "c"] = "a"

        x["c"] = x["c"].astype("category")

        box_init = sdutil._make_box(x)
        boxlim = box_init.copy()
        boxlim.a = [0.5, 1.0]
        boxlim.c = [set("b")] * 2
        restricted_dims = ["a", "c"]

        sdutil.plot_pair_wise_scatter(x, y, boxlim, box_init, restricted_dims)
        plt.draw()
        plt.close("all")

    def test_plot_pair_wise_scatter_multiple_boxes(self):
        from matplotlib.collections import LineCollection

        rng = np.random.default_rng(0)
        n = 300
        x = pd.DataFrame(
            {
                "a": rng.random(n),
                "b": rng.random(n),
                "c": pd.Categorical(rng.choice(["p", "q", "r"], n)),
            }
        )
        box_init = sdutil._make_box(x)

        def make(**limits):
            box = box_init.copy()
            for column, value in limits.items():
                box[column] = value
            return box

        # an L-shaped region from two boxes, plus a secondary region
        boxes = [make(a=[0, 0.5], c=[{"p"}] * 2), make(a=[0.5, 1], b=[0, 0.4], c=[{"r"}] * 2)]
        secondary = [make(a=[0.5, 1], b=[0.4, 1])]
        labels = pd.Categorical(
            np.where(x.a < 0.5, "inside", "other"), categories=["other", "inside"]
        )
        palette = {"other": "lightgrey", "inside": "tab:blue"}

        grid = sdutil.plot_pair_wise_scatter(
            x,
            labels,
            boxes,
            box_init,
            ["a", "b", "c"],
            diag="hist",
            upper=None,
            lower="scatter",
            secondary_boxlims=secondary,
            box_color="tab:blue",
            palette=palette,
            plot_kws={"s": 5},
        )
        ax = grid.axes[1, 0]  # b against a
        outlines = [c for c in ax.collections if isinstance(c, LineCollection)]
        self.assertEqual(len(outlines), 2)  # the union, and the dashed secondary region
        # the union of the two boxes is a single L-shaped outline: six edges,
        # drawn as more segments where the grid splits an edge
        self.assertGreaterEqual(len(outlines[0].get_segments()), 6)
        self.assertEqual(outlines[1].get_linestyle()[0][1] is not None, True)
        self.assertFalse(grid.axes[0, 1].axison)  # empty upper triangle switched off

        # categorical axis: categories of the boxes come first, p and r are
        # not adjacent in the box, so each box covers its own interval
        ax = grid.axes[2, 0]  # c against a
        self.assertEqual(
            [t.get_text() for t in ax.get_yticklabels()][1:3], ["p", "r"]
        )

        grid = sdutil.plot_pair_wise_scatter(
            x, labels, boxes, box_init, ["a", "b"], diag="hist", outline="boxes"
        )
        rectangles = [p for p in grid.axes[1, 0].patches if p.get_facecolor()[3] == 0]
        self.assertEqual(len(rectangles), 2)

        with self.assertRaises(ValueError):
            sdutil.plot_pair_wise_scatter(
                x, labels, boxes, box_init, ["a", "b"], diag="hist", outline="other"
            )
        plt.close("all")

    def test_plot_pair_wise_scatter_kde_diagonal(self):
        # diag="kde" draws a kernel density estimate per class on the diagonal,
        # not a scatter of the uncertainty against y
        from matplotlib.collections import PathCollection

        rng = np.random.default_rng(1)
        x = pd.DataFrame({"a": rng.random(200), "b": rng.random(200)})
        y = (x.a > 0.5).astype(int).values
        box_init = sdutil._make_box(x)
        boxlim = box_init.copy()
        boxlim["a"] = [0.5, 1.0]

        grid = sdutil.plot_pair_wise_scatter(
            x, y, boxlim, box_init, ["a", "b"], diag="kde", lower="scatter"
        )
        for ax in grid.diag_axes:
            self.assertGreaterEqual(len(ax.get_lines()), 2)  # one density per class
            self.assertFalse(any(isinstance(c, PathCollection) for c in ax.collections))
        plt.close("all")

    def test_draw_union_outline(self):
        fig, ax = plt.subplots()
        # two touching squares form one rectangle: 4 outer edges, each split once
        sdutil._draw_union_outline(ax, [(0, 1, 0, 1), (1, 2, 0, 1)], colors=["k"])
        segments = ax.collections[0].get_segments()
        self.assertEqual(len(segments), 6)
        self.assertFalse(
            any(np.allclose(seg, [[1, 0], [1, 1]]) for seg in segments)
        )  # the shared edge is not drawn
        plt.close(fig)

    def test_merge_code_runs(self):
        self.assertEqual(sdutil._find_contiguous_ranges([3, 0, 1, 5]), [(0, 1), (3, 3), (5, 5)])
        self.assertEqual(sdutil._find_contiguous_ranges([]), [])

    def test_plot_boxes(self):
        x = pd.DataFrame(
            [
                [0.1, 0, "a"],
                [0.2, 1, "b"],
                [0.3, 2, "a"],
                [0.4, 3, "b"],
                [0.5, 4, "a"],
                [0.6, 5, "a"],
                [0.7, 6, "b"],
                [0.8, 7, "a"],
                [0.9, 8, "b"],
                [1.0, 9, "a"],
            ],
            columns=["a", "b", "c"],
        )
        y = np.zeros((x.shape[0],), dtype=int)
        logical = (x.a > 0.5) & (x.c != "a")
        y[logical] = 1

        logical = (x.a < 0.5) & (x.c != "b")
        y[logical] = 1

        x["c"] = x["c"].astype("category")

        box_init = sdutil._make_box(x)
        boxlim1 = box_init.copy()
        boxlim1.a = [0.5, 1]
        boxlim1.c = [set("b")] * 2

        boxlim2 = box_init.copy()
        boxlim2.a = [0.1, 0.5]
        boxlim2.c = [set("a")] * 2

        sdutil.plot_boxes(x, [boxlim1, boxlim2], together=True)
        sdutil.plot_boxes(x, [boxlim1, boxlim2], together=False)
        plt.draw()
        plt.close("all")

    def _make_mixed_data(self, n=1000, seed=0):
        rng = np.random.default_rng(seed)
        x = pd.DataFrame(
            {
                "a": rng.random(n),
                "i": rng.integers(0, 10, n),
                "b": rng.random(n) > 0.5,
                "c": pd.Categorical(rng.choice(["p", "q", "r"], n)),
            }
        )
        y = (((x.c == "q") & (x.a > 0.5)) | ((x.i > 6) & x.b)).astype(int).values
        return x, y

    def _fit_tree(self, x, y, sep="!?!"):
        from sklearn.tree import DecisionTreeClassifier

        dummies = pd.get_dummies(x, prefix_sep=sep)
        dummies_map = {
            c: {str(v): v for v in x[c].unique()}
            for c in x.select_dtypes(exclude=np.number).columns
        }
        clf = DecisionTreeClassifier(min_samples_leaf=20, random_state=0)
        clf.fit(dummies.values, y)
        return clf, dummies.columns.tolist(), dummies_map

    def test_extract_tree_boxes(self):
        x, y = self._make_mixed_data()
        box_init_before = sdutil._make_box(x)
        clf, names, dummies_map = self._fit_tree(x, y)

        boxes = sdutil._extract_tree_boxes(clf, x, names, dummies_map)
        leaves = np.flatnonzero(clf.tree_.children_left == -1)
        self.assertEqual([node for node, _ in boxes], leaves.tolist())

        # the boxes partition the data exactly as the leaves of the tree do
        leaf_of_point = clf.apply(pd.get_dummies(x, prefix_sep="!?!").values)
        for node, box in boxes:
            inside = np.asarray(sdutil._in_box(x, box))
            np.testing.assert_array_equal(inside, leaf_of_point == node)

        # integer and boolean limits stay integer and boolean
        for _, box in boxes:
            self.assertTrue(pd.api.types.is_integer_dtype(box["i"].dtype))
            self.assertTrue(pd.api.types.is_bool_dtype(box["b"].dtype))

        # a categorical split restricts both children; nothing is mutated
        categories = [box.at[0, "c"] for _, box in boxes]
        self.assertTrue(any(c == {"q"} for c in categories))
        pd.testing.assert_frame_equal(sdutil._make_box(x), box_init_before)

    def test_extract_tree_boxes_merge_key(self):
        x, y = self._make_mixed_data()
        clf, names, dummies_map = self._fit_tree(x, y)
        tree_ = clf.tree_
        leaf_of_point = clf.apply(pd.get_dummies(x, prefix_sep="!?!").values)

        def leaves_below(node):
            if tree_.children_left[node] == -1:
                return {node}
            return leaves_below(tree_.children_left[node]) | leaves_below(
                tree_.children_right[node]
            )

        # key = side of the root split: exactly the two children of the root
        left_leaves = leaves_below(tree_.children_left[0])
        merged = sdutil._extract_tree_boxes(
            clf, x, names, dummies_map, merge_key=lambda n: n in left_leaves
        )
        self.assertEqual(
            [node for node, _ in merged],
            [tree_.children_left[0], tree_.children_right[0]],
        )
        for node, box in merged:
            inside = np.asarray(sdutil._in_box(x, box))
            np.testing.assert_array_equal(
                inside, np.isin(leaf_of_point, list(leaves_below(node)))
            )

        # one key for all leaves: a single box, the initial box
        merged = sdutil._extract_tree_boxes(
            clf, x, names, dummies_map, merge_key=lambda n: 0
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0][0], 0)
        self.assertTrue(np.all(sdutil._in_box(x, merged[0][1])))

        # key None never merges
        merged = sdutil._extract_tree_boxes(
            clf, x, names, dummies_map, merge_key=lambda n: None
        )
        self.assertEqual(len(merged), len(leaves_below(0)))

    def test_compute_purity(self):
        self.assertEqual(sdutil._compute_purity([10, 0, 0]), 1)
        self.assertAlmostEqual(sdutil._compute_purity([5, 5, 5]), 0)
        self.assertAlmostEqual(sdutil._compute_purity([5, 5]), 0)
        self.assertEqual(sdutil._compute_purity([7]), 1)
        self.assertEqual(sdutil._compute_purity([0, 0]), 1)
        p = sdutil._compute_purity([8, 1, 1])
        self.assertTrue(0 < p < 1)

    def test_determine_effective_restricted_dims(self):
        rng = np.random.default_rng(2)
        x = pd.DataFrame(rng.random((2000, 3)), columns=["a", "b", "c"])
        box_init = sdutil._make_box(x)

        def make(**limits):
            box = box_init.copy()
            for column, (lower, upper) in limits.items():
                box[column] = [lower, upper]
            return box

        # L-shape: both a and b matter jointly
        l_shape = [make(a=(0, 0.5)), make(a=(0.5, 1), b=(0, 0.4))]
        self.assertEqual(
            sdutil._determine_effective_restricted_dims(x, l_shape, box_init),
            ["a", "b"],
        )

        # split on b on both sides with identical a limits: b is irrelevant
        redundant = [make(a=(0, 0.3), b=(0, 0.5)), make(a=(0, 0.3), b=(0.5, 1))]
        self.assertEqual(
            sdutil._determine_effective_restricted_dims(x, redundant, box_init), ["a"]
        )

        # near vacuous limit: only dropped with a tolerance
        nearly = [make(a=(0, 0.5), c=(0, 0.995))]
        self.assertEqual(
            sdutil._determine_effective_restricted_dims(x, nearly, box_init),
            ["a", "c"],
        )
        self.assertEqual(
            sdutil._determine_effective_restricted_dims(
                x, nearly, box_init, tolerance=0.02
            ),
            ["a"],
        )
        self.assertEqual(
            sdutil._determine_effective_restricted_dims(x, [], box_init), []
        )

    def test_OutputFormatterMixin(self):
        x = pd.DataFrame(
            [
                [0.1, 0, "a"],
                [0.2, 1, "b"],
                [0.3, 2, "a"],
                [0.4, 3, "b"],
                [0.5, 4, "a"],
                [0.6, 5, "a"],
                [0.7, 6, "b"],
                [0.8, 7, "a"],
                [0.9, 8, "b"],
                [1.0, 9, "a"],
            ],
            columns=["a", "b", "c"],
        )
        y = np.zeros((x.shape[0],), dtype=int)
        logical = (x.a > 0.5) & (x.c != "a")
        y[logical] = 1

        logical = (x.a < 0.5) & (x.c != "b")
        y[logical] = 1

        x["c"] = x["c"].astype("category")

        box_init = sdutil._make_box(x)
        boxlim1 = box_init.copy()
        boxlim1.a = [0.5, 1]
        boxlim1.c = [set("b")] * 2

        boxlim2 = box_init.copy()
        boxlim2.a = [0.1, 0.5]
        boxlim2.c = [set("a")] * 2

        with self.assertRaises(AttributeError):

            class WrongTestFormatter(sdutil.OutputFormatterMixin):
                @property
                def boxes(self):
                    pass

                @property
                def stats(self):
                    pass

            formatter = WrongTestFormatter()
            formatter.boxes = [boxlim1, boxlim2]
            formatter.stats = [
                {"coverage": 0.5, "density": 1},
                {"coverage": 0.5, "density": 1},
            ]

        class TestFormatter(sdutil.OutputFormatterMixin):

            @property
            def boxes(self):
                return self._boxes

            @property
            def stats(self):
                return self._stats

        formatter = TestFormatter()
        formatter._boxes = [boxlim1, boxlim2]
        formatter._stats = [
            {"coverage": 0.5, "density": 1},
            {"coverage": 0.5, "density": 1},
        ]
        formatter.x = x

        formatter.show_boxes()
        plt.draw()
        plt.close("all")

        boxes = formatter.boxes_to_dataframe()

        expected_boxes = pd.DataFrame(
            [[{"b"}, {"b"}, {"a"}, {"a"}], [0.5, 1, 0.1, 0.5]],
            index=["c", "a"],
            columns=pd.MultiIndex(
                levels=[["box 1", "box 2"], ["max", "min"]],
                codes=[[0, 0, 1, 1], [1, 0, 1, 0]],
            ),
        )

        a = expected_boxes.equals(boxes)
        self.assertTrue(a)

        # check stats
        stats = formatter.stats_to_dataframe()
        expected_stats = pd.DataFrame(
            [[0.5, 1], [0.5, 1]],
            index=["box 1", "box 2"],
            columns=["coverage", "density"],
        )

        self.assertTrue(expected_stats.equals(stats))


#
# if __name__ == "__main__":
#     unittest.main()

#     suite = unittest.TestSuite()
#     suite.addTest(PrimTestCase("test_write_boxes_to_stdout"))
#     unittest.TextTestRunner().run(suite)

    def test_drop_uninformative_columns(self):
        x = pd.DataFrame(
            {
                "scenario": np.arange(4),
                "a": [0.1, 0.5, 0.2, 0.9],
                "constant_float": 1.0,
                "constant_int": 3,
                "model": pd.Categorical(["m"] * 4),
                "policy": pd.Categorical([np.nan] * 4),
                "with_missing": pd.Categorical(["p", np.nan, "p", np.nan]),
                "lever": pd.Categorical(["p", "q", "p", "q"]),
            }
        )
        dropped = sdutil._drop_uninformative_columns(x)
        self.assertEqual(dropped.columns.tolist(), ["a", "with_missing", "lever"])
        self.assertIn("scenario", x.columns)  # input left untouched

        kept = sdutil._drop_uninformative_columns(x, exclude=())
        self.assertIn("scenario", kept.columns)

        with self.assertRaises(EMAError):
            sdutil._drop_uninformative_columns(x[["scenario", "constant_float"]])
