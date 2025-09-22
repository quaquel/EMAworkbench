"""Tests for Prim."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ema_workbench.analysis import prim
from ema_workbench.analysis.prim import PrimBox
from test import utilities

# Created on Mar 13, 2012
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


def flu_classify(data):
    """Helper function for test data."""
    # get the output for deceased population
    result = data["deceased_population_region_1"]

    # make an empty array of length equal to number of cases
    classes = np.zeros(result.shape[0])

    # if deceased population is higher then 1.000.000 people, classify as 1
    classes[result[:, -1] > 1000000] = 1

    return classes


class TestPrimBox:
    """Test for prim box."""

    def test_init(self):
        x = pd.DataFrame([(0, 1, 2), (2, 5, 6), (3, 2, 1)], columns=["a", "b", "c"])
        y = np.array([0, 1, 0])

        prim_obj = prim.Prim(x, y)
        box = PrimBox(prim_obj, prim_obj.box_init, prim_obj.yi)

        assert box.peeling_trajectory.shape == (1, 8)

        with pytest.raises(ValueError):
            x = pd.DataFrame([(0, 1, 2), (2, 5, 6), (3, 2, 1)], columns=["a", "b", "c"])
            y = np.array([[0, 0], [0, 0], [0, 0]])
            prim.Prim(x, y)
        with pytest.raises(ValueError):
            x = pd.DataFrame([(0, 1, 2), (2, 5, 6), (3, 2, 1)], columns=["a", "b", "c"])
            y = np.array([0, 0])
            prim.Prim(x, y)

    def test_select(self):
        x = pd.DataFrame([(0, 1, 2), (2, 5, 6), (3, 2, 1)], columns=["a", "b", "c"])
        y = np.array([1, 1, 0])

        prim_obj = prim.Prim(x, y)
        box = PrimBox(prim_obj, prim_obj.box_init, prim_obj.yi)

        new_box_lim = pd.DataFrame([(0, 1, 1), (2, 5, 6)], columns=["a", "b", "c"])
        indices = np.array([0, 1], dtype=int)
        box.update(new_box_lim, indices)

        box.select(0)
        assert np.all(box.yi == prim_obj.yi)

    def test_inspect(self):
        x = pd.DataFrame([(0, 1, 2), (2, 5, 6), (3, 2, 1)], columns=["a", "b", "c"])
        y = np.array([1, 1, 0])

        prim_obj = prim.Prim(x, y)
        box = PrimBox(prim_obj, prim_obj.box_init, prim_obj.yi)

        new_box_lim = pd.DataFrame([(0, 1, 1), (2, 5, 6)], columns=["a", "b", "c"])
        indices = np.array([0, 1], dtype=int)
        box.update(new_box_lim, indices)

        box.inspect(1)
        box.inspect()
        box.inspect(style="graph")
        box.inspect(style="data")

        box.inspect([0, 1])

        fig, axes = plt.subplots(2)
        box.inspect([0, 1], ax=axes, style="graph")

        fig, ax = plt.subplots()
        box.inspect(0, ax=ax, style="graph")

        with pytest.raises(ValueError):
            fig, axes = plt.subplots(3)
            box.inspect([0, 1], ax=axes, style="graph")
        with pytest.raises(ValueError):
            box.inspect(style="some unknown style")
        with pytest.raises(TypeError):
            box.inspect([0, "a"])

    def test_show_ppt(self):
        x = pd.DataFrame([(0, 1, 2), (2, 5, 6), (3, 2, 1)], columns=["a", "b", "c"])
        y = np.array([1, 1, 0])

        prim_obj = prim.Prim(x, y)
        box = PrimBox(prim_obj, prim_obj.box_init, prim_obj.yi)

        cols = ["mean", "mass", "coverage", "density", "res_dim"]
        data = np.zeros((100, 5))
        data[:, 0:4] = np.random.rand(100, 4)
        data[:, 4] = np.random.randint(0, 5, size=(100,))
        box.peeling_trajectory = pd.DataFrame(data, columns=cols)

        box.show_ppt()

    def test_show_tradeoff(self):
        x = pd.DataFrame([(0, 1, 2), (2, 5, 6), (3, 2, 1)], columns=["a", "b", "c"])
        y = np.array([1, 1, 0])

        prim_obj = prim.Prim(x, y)
        box = PrimBox(prim_obj, prim_obj.box_init, prim_obj.yi)

        cols = ["mean", "mass", "coverage", "density", "res_dim"]
        data = np.zeros((100, 5))
        data[:, 0:4] = np.random.rand(100, 4)
        data[:, 4] = np.random.randint(0, 5, size=(100,))
        box.peeling_trajectory = pd.DataFrame(data, columns=cols)

        box.show_tradeoff()

    def test_update(self):
        x = pd.DataFrame([(0, 1, 2), (2, 5, 6), (3, 2, 1)], columns=["a", "b", "c"])
        y = np.array([1, 1, 0])

        prim_obj = prim.Prim(x, y)
        box = PrimBox(prim_obj, prim_obj.box_init, prim_obj.yi)

        new_box_lim = pd.DataFrame([(0, 1, 1), (2, 5, 6)], columns=["a", "b", "c"])
        indices = np.array([0, 1], dtype=int)
        box.update(new_box_lim, indices)

        assert box.peeling_trajectory["mean"][1] == 1
        assert box.peeling_trajectory["coverage"][1] == 1
        assert box.peeling_trajectory["density"][1] == 1
        assert box.peeling_trajectory["res_dim"][1] == 1
        assert box.peeling_trajectory["mass"][1] == 2 / 3

    def test_drop_restriction(self):
        x = pd.DataFrame([(0, 1, 2), (2, 5, 6), (3, 2, 1)], columns=["a", "b", "c"])
        y = np.array([1, 1, 0])

        prim_obj = prim.Prim(x, y)
        box = PrimBox(prim_obj, prim_obj.box_init, prim_obj.yi)

        new_box_lim = pd.DataFrame([(0, 1, 1), (2, 2, 6)], columns=["a", "b", "c"])
        indices = np.array([0, 1], dtype=int)
        box.update(new_box_lim, indices)

        box.drop_restriction("b")

        correct_box_lims = pd.DataFrame([(0, 1, 1), (2, 5, 6)], columns=["a", "b", "c"])
        box_lims = box.box_lims[-1]
        names = box_lims.columns
        for entry in names:
            lim_correct = correct_box_lims[entry]
            lim_box = box_lims[entry]
            for i in range(len(lim_correct)):
                assert lim_correct[i] == lim_box[i]

        assert box.peeling_trajectory["mean"][2] == 1
        assert box.peeling_trajectory["coverage"][2] == 1
        assert box.peeling_trajectory["density"][2] == 1
        assert box.peeling_trajectory["res_dim"][2] == 1
        assert box.peeling_trajectory["mass"][2] == 2 / 3

    def test_calculate_quasi_p(self):
        pass

    def test_resample(self):
        experiments, outcomes = utilities.load_flu_data()
        y = flu_classify(outcomes)

        alg = prim.Prim(experiments, y)
        box = alg.find_box()

        box.resample()

    def test_show_pairs_scatter(self):
        experiments, outcomes = utilities.load_flu_data()
        y = flu_classify(outcomes)

        experiments = experiments.iloc[::2]
        y = y[::2]

        alg = prim.Prim(experiments, y)
        box = alg.find_box()

        box.show_pairs_scatter(i=10)
        plt.draw()
        box.show_pairs_scatter(diag="cdf", upper="hist")
        plt.draw()
        box.show_pairs_scatter(diag="cdf", upper="contour")
        plt.draw()
        grid = box.show_pairs_scatter(dims=["normal_contact_rate_region_1", "infection_ratio_region_1"])
        plt.draw()
        assert grid.axes.shape == (2,2)

        with pytest.raises(ValueError):
            box.show_pairs_scatter(diag="wrong value", upper="contour")

    def test_inspect_tradeoff(self):
        experiments, outcomes = utilities.load_flu_data()
        y = flu_classify(outcomes)

        alg = prim.Prim(experiments, y)
        box = alg.find_box()
        box.inspect_tradeoff()


class TestPrim:
    """Test for prim algorithm."""

    def test_boxes(self):
        x = pd.DataFrame([(0, 1, 2), (2, 5, 6), (3, 2, 1)], columns=["a", "b", "c"])
        y = np.array([0, 1, 1])

        prim_obj = prim.Prim(x, y)
        boxes = prim_obj.boxes

        assert len(boxes) == 1, "box length not correct"

        # real data test case
        x, outcomes = utilities.load_flu_data()
        y = flu_classify(outcomes)

        prim_obj = prim.Prim(x, y)
        prim_obj.find_box()
        boxes = prim_obj.boxes
        assert len(boxes) == 1, "box length not correct"

    def test_quantile(self):
        data = pd.Series(np.arange(10))
        assert prim.get_quantile(data, 0.9) == 8.5
        assert prim.get_quantile(data, 0.95) == 8.5
        assert prim.get_quantile(data, 0.1) == 0.5
        assert prim.get_quantile(data, 0.05) == 0.5

        data = pd.Series(1)
        assert prim.get_quantile(data, 0.9) == 1
        assert prim.get_quantile(data, 0.95) == 1
        assert prim.get_quantile(data, 0.1) == 1
        assert prim.get_quantile(data, 0.05) == 1

        data = pd.Series([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9])
        assert prim.get_quantile(data, 0.9) == 8.5
        assert prim.get_quantile(data, 0.95) == 8.5
        assert prim.get_quantile(data, 0.1) == 1.5
        assert prim.get_quantile(data, 0.05) == 1.5

    # def test_box_init(self):
    #     fixme this is all regression style tests
    #     # test init box without NANS
    #     x = pd.DataFrame([(0, 1, 2), (2, 5, 6), (3, 2, 7)], columns=["a", "b", "c"])
    #     y = np.array([0, 1, 2])
    #
    #     prim_obj = prim.Prim(x, y, threshold=0.5, mode=RuleInductionType.REGRESSION)
    #     box_init = prim_obj.box_init
    #
    #     # some test on the box
    #     assert box_init.loc[0, "a"] == 0
    #     assert box_init.loc[1, "a"] == 3
    #     assert box_init.loc[0, "b"] == 1
    #     assert box_init.loc[1, "b"] == 5
    #     assert box_init.loc[0, "c"] == 2
    #     assert box_init.loc[1, "c"] == 7
    #
    #     # heterogeneous without NAN
    #     x = pd.DataFrame(
    #         [
    #             [0.1, 0, "a"],
    #             [0.2, 1, "b"],
    #             [0.3, 2, "a"],
    #             [0.4, 3, "b"],
    #             [0.5, 4, "a"],
    #             [0.6, 5, "a"],
    #             [0.7, 6, "b"],
    #             [0.8, 7, "a"],
    #             [0.9, 8, "b"],
    #             [1.0, 9, "a"],
    #         ],
    #         columns=["a", "b", "c"],
    #     )
    #     y = np.arange(0, x.shape[0])
    #
    #     prim_obj = prim.Prim(x, y, threshold=0.5, mode=RuleInductionType.REGRESSION)
    #     box_init = prim_obj.box_init
    #
    #     # some test on the box
    #     assert box_init["a"][0] == 0.1
    #     assert box_init["a"][1] == 1.0
    #     assert box_init["b"][0] == 0
    #     assert box_init["b"][1] == 9
    #     assert box_init["c"][0] == {"a", "b"}
    #     assert box_init["c"][1] == {"a", "b"}

    def test_find_box(self):
        x, outcomes = utilities.load_flu_data()
        y = flu_classify(outcomes)

        prim_obj = prim.Prim(x, y)
        box_1 = prim_obj.find_box()
        prim_obj._update_yi_remaining(prim_obj)

        after_find = box_1.yi.shape[0] + prim_obj.yi_remaining.shape[0]
        assert after_find, prim_obj.y.shape[0]

        box_2 = prim_obj.find_box()
        prim_obj._update_yi_remaining(prim_obj)

        after_find = (
            box_1.yi.shape[0] + box_2.yi.shape[0] + prim_obj.yi_remaining.shape[0]
        )
        assert after_find, prim_obj.y.shape[0]

    def test_discrete_peel(self):
        x = pd.DataFrame(
            np.random.randint(0, 10, size=(100,), dtype=int), columns=["a"]
        )
        y = np.zeros(100)
        y[x.a > 5] = 1

        primalg = prim.Prim(x, y)
        boxlims = primalg.box_init
        box = prim.PrimBox(primalg, boxlims, primalg.yi)

        peels = primalg._discrete_peel(box, "a", 0, primalg.x_int)

        assert len(peels) == 2
        for peel in peels:
            assert len(peel) == 2

            indices, tempbox = peel

            assert isinstance(indices, np.ndarray)
            assert isinstance(tempbox, pd.DataFrame)

        # have modified boxlims as starting point
        primalg = prim.Prim(x, y)
        boxlims = primalg.box_init
        boxlims.a = [1, 8]
        box = prim.PrimBox(primalg, boxlims, primalg.yi)

        peels = primalg._discrete_peel(box, "a", 0, primalg.x_int)

        assert len(peels) == 2
        for peel in peels:
            assert len(peel) == 2

            indices, tempbox = peel

            assert isinstance(indices, np.ndarray)
            assert isinstance(tempbox, pd.DataFrame)

        # have modified boxlims as starting point
        x.a[x.a > 5] = 5
        primalg = prim.Prim(x, y)
        boxlims = primalg.box_init
        boxlims.a = [5, 8]
        box = prim.PrimBox(primalg, boxlims, primalg.yi)

        peels = primalg._discrete_peel(box, "a", 0, primalg.x_int)
        assert len(peels) == 2

        x.a[x.a < 5] = 5
        primalg = prim.Prim(x, y)
        boxlims = primalg.box_init
        boxlims.a = [5, 8]
        box = prim.PrimBox(primalg, boxlims, primalg.yi)

        peels = primalg._discrete_peel(box, "a", 0, primalg.x_int)
        assert len(peels) == 2

    def test_categorical_peel(self):
        x = pd.DataFrame(
            list(
                zip(
                    np.random.rand(10),
                    ["a", "b", "a", "b", "a", "a", "b", "a", "b", "a"],
                )
            ),
            columns=["a", "b"],
        )

        y = np.random.randint(0, 1, (10,))
        y = y.astype(int)

        prim_obj = prim.Prim(x, y)
        box_lims = pd.DataFrame([(0, {"a", "b"}), (1, {"a", "b"})], columns=["a", "b"])
        box = prim.PrimBox(prim_obj, box_lims, prim_obj.yi)

        u = "b"
        x = x.select_dtypes(exclude=np.number).values
        j = 0
        peels = prim_obj._categorical_peel(box, u, j, x)

        assert len(peels) == 2

        for peel in peels:
            pl = peel[1][u]
            assert len(pl[0]) == 1
            assert len(pl[1]) == 1

        a = ("a",)
        b = ("b",)
        x = pd.DataFrame(
            list(zip(np.random.rand(10), [a, b, a, b, a, a, b, a, b, a])),
            columns=["a", "b"],
        )

        y = np.random.randint(0, 2, (10,))
        y = y.astype(int)

        prim_obj = prim.Prim(x, y)
        box_lims = prim_obj.box_init
        box = prim.PrimBox(prim_obj, box_lims, prim_obj.yi)

        u = "b"
        x = x.select_dtypes(exclude=np.number).values
        j = 0
        peels = prim_obj._categorical_peel(box, u, j, x)

        assert len(peels) == 2

        for peel in peels:
            pl = peel[1][u]
            assert len(pl[0]) == 1
            assert len(pl[1]) == 1

    def test_categorical_paste(self):
        a = np.random.rand(10)
        b = ["a", "b", "a", "b", "a", "a", "b", "a", "b", "a"]
        x = pd.DataFrame(list(zip(a, b)), columns=["a", "b"])
        x["b"] = x["b"].astype("category")

        y = np.random.randint(0, 2, (10,))
        y = y.astype(int)

        prim_obj = prim.Prim(x, y)
        box_lims = pd.DataFrame([(0, {"a"}), (1, {"a"})], columns=x.columns)

        yi = np.where(x.loc[:, "b"] == "a")

        box = prim.PrimBox(prim_obj, box_lims, yi)

        u = "b"
        pastes = prim_obj._categorical_paste(box, u, x, ["b"])

        assert len(pastes) == 1

        for paste in pastes:
            indices, box_lims = paste

            assert indices.shape[0] == 10
            assert box_lims[u][0] == {"a", "b"}

    def test_constrained_prim(self):
        experiments, outcomes = utilities.load_flu_data()
        y = flu_classify(outcomes)

        prim.run_constrained_prim(experiments, y, issignificant=True)


class TestRegressionPrim:
    """Test prim in regression mode."""

    def test_all(self):
        experiments, outcomes = utilities.load_flu_data()
        y = outcomes["deceased_population_region_1"][:, -1]

        alg = prim.RegressionPrim(experiments, y)
        box = alg.find_box()

        box.inspect(10)
        box.inspect(10, style="graph")
        plt.draw()
        ret = box.inspect(10, style="data")[0]
        assert len(ret) == 2

        box.show_pairs_scatter()
        plt.draw()

        alg = prim.RegressionPrim(experiments, y, maximization=False)
        assert alg._maximization is False
        box = alg.find_box()

        column = box.peeling_trajectory["mean"]
        assert column.iloc[0] > column.iloc[-1]


def test_pca():
    experiments, outcomes = utilities.load_flu_data()
    y = outcomes["deceased_population_region_1"][:, -1]

    # fail on non numeric columns
    with pytest.raises(ValueError):
        prim.pca_preprocess(experiments, y)
    experiments = experiments.drop(['scenario', 'policy', 'model'], axis=1)

    # fail on y not being binary
    with pytest.raises(ValueError):
        prim.pca_preprocess(experiments, y)

    # behave correctly
    y = y > 1e6
    rotated_experiments, rotation = prim.pca_preprocess(experiments, y)

    assert rotated_experiments.shape == experiments.shape

    subsets = {
        "region 1": [
            "additional_seasonal_immune_population_fraction_R1",
            "fatality_ratio_region_1",
            "infection_ratio_region_1",
            "initial_immune_fraction_of_the_population_of_region_1",
            "normal_contact_rate_region_1",
            "permanent_immune_population_fraction_R1",
            "recovery_time_region_1",
            "root_contact_rate_region_1",
            "susceptible_to_immune_population_delay_time_region_1",
        ],
        "region 2": [
            "additional_seasonal_immune_population_fraction_R2",
            "fatality_rate_region_2",
            "infection_rate_region_2",
            "initial_immune_fraction_of_the_population_of_region_2",
            "normal_contact_rate_region_2",
            "permanent_immune_population_fraction_R2",
            "recovery_time_region_2",
            "root_contact_ratio_region_2",
            "susceptible_to_immune_population_delay_time_region_2"],
    }
    experiments = experiments.drop("normal_interregional_contact_rate", axis=1)
    rotated_experiments, rotation = prim.pca_preprocess(experiments, y, subsets)
    assert rotated_experiments.shape == experiments.shape