""" """

import unittest
import unittest.mock as mock

import scipy as sp

from ema_workbench.em_framework import parameters


class RealParameterTestCase(unittest.TestCase):
    def test_instantiation(self):
        name = "test"
        resolution = [0, 1, 2]
        lower_bound = 0
        upper_bound = 2.1
        par = parameters.RealParameter(
            name, lower_bound, upper_bound, resolution=resolution
        )

        self.assertEqual(par.name, name)
        self.assertEqual(par.resolution, resolution)
        self.assertEqual(par.lower_bound, lower_bound)
        self.assertEqual(par.upper_bound, upper_bound)

        name = "test"
        resolution = [0, 1, 2]
        lower_bound = 2.1
        upper_bound = 0

        with self.assertRaises(ValueError):
            par = parameters.RealParameter(
                name, lower_bound, upper_bound, resolution=resolution
            )

        with self.assertRaises(ValueError):
            resolution = [-1, 0]
            par = parameters.RealParameter(
                name, lower_bound, upper_bound, resolution=resolution
            )

            resolution = [0, 1, 3]
            par = parameters.RealParameter(
                name, lower_bound, upper_bound, resolution=resolution
            )

    def test_comparison(self):
        name = "test"
        resolution = [0, 1, 2]
        lower_bound = 0
        upper_bound = 2.0
        par1 = parameters.RealParameter(
            name, lower_bound, upper_bound, resolution=resolution
        )
        par2 = parameters.RealParameter(
            name, lower_bound, upper_bound, resolution=resolution
        )

        self.assertTrue(par1 == par2)

        name = "test"
        par1 = parameters.RealParameter(
            name, lower_bound, upper_bound, resolution=resolution
        )

        name = "what"
        par2 = parameters.RealParameter(
            name, lower_bound, upper_bound, resolution=resolution
        )
        self.assertFalse(par1 == par2)

    def test_dist(self):
        name = "test"
        resolution = [0, 1, 2]
        lower_bound = 0
        upper_bound = 2.1
        par = parameters.RealParameter(
            name, lower_bound, upper_bound, resolution=resolution
        )

        self.assertEqual(par.dist.dist.name, "uniform")
        self.assertEqual(par.lower_bound, lower_bound)
        self.assertEqual(par.upper_bound, upper_bound)

    def test_from_dist(self):
        par = parameters.RealParameter.from_dist(
            "test",
            sp.stats.uniform(0, 1),
            resolution=[0, 1],  # @UndefinedVariable
        )

        self.assertEqual(par.name, "test")
        self.assertEqual(par.dist.dist.name, "uniform")
        self.assertEqual(par.lower_bound, 0)
        self.assertEqual(par.upper_bound, 1)
        self.assertEqual(par.resolution, [0, 1])

        with self.assertRaises(ValueError):
            parameters.RealParameter.from_dist(
                "test", sp.stats.randint(0, 1)
            )  # @UndefinedVariable
            parameters.RealParameter.from_dist(
                "test",
                sp.stats.uniform(0, 1),
                blaat=[0, 1],  # @UndefinedVariable
            )


class IntegerParameterTestCase(unittest.TestCase):
    def test_instantiation(self):
        name = "test"
        resolution = [0, 1, 2]
        lower_bound = 0
        upper_bound = 2
        par = parameters.IntegerParameter(
            name, lower_bound, upper_bound, resolution=resolution
        )

        self.assertEqual(par.name, name)
        self.assertEqual(par.resolution, resolution)
        self.assertEqual(par.lower_bound, lower_bound)
        self.assertEqual(par.upper_bound, upper_bound)

        name = "test"
        resolution = [0, 1, 2]
        lower_bound = 2
        upper_bound = 0

        with self.assertRaises(ValueError):
            par = parameters.IntegerParameter(
                name, lower_bound, upper_bound, resolution=resolution
            )

        with self.assertRaises(ValueError):
            resolution = [-1, 0]
            par = parameters.IntegerParameter(
                name, lower_bound, upper_bound, resolution=resolution
            )

            resolution = [0, 1, 3]
            par = parameters.IntegerParameter(
                name, lower_bound, upper_bound, resolution=resolution
            )

        with self.assertRaises(ValueError):
            par = parameters.IntegerParameter(
                name, lower_bound, 2.1, resolution=resolution
            )

            par = parameters.IntegerParameter(name, 0.0, 2, resolution=resolution)

        with self.assertRaises(ValueError):
            par = parameters.IntegerParameter(
                name, lower_bound, upper_bound, resolution=[0, 1.5, 2]
            )

    def test_dist(self):
        name = "test"
        resolution = [0, 1, 2]
        lower_bound = 0
        upper_bound = 2
        par = parameters.IntegerParameter(
            name, lower_bound, upper_bound, resolution=resolution
        )

        self.assertEqual(par.dist.dist.name, "randint")
        self.assertEqual(par.lower_bound, lower_bound)
        self.assertEqual(par.upper_bound, upper_bound)

    def test_from_dist(self):
        par = parameters.IntegerParameter.from_dist(
            "test",
            sp.stats.randint(0, 10),
            resolution=[0, 9],  # @UndefinedVariable
        )

        self.assertEqual(par.name, "test")
        self.assertEqual(par.dist.dist.name, "randint")
        self.assertEqual(par.lower_bound, 0)
        self.assertEqual(par.upper_bound, 9)
        self.assertEqual(par.resolution, [0, 9])

        with self.assertRaises(ValueError):
            parameters.IntegerParameter.from_dist(
                "test", sp.stats.uniform(0, 1)
            )  # @UndefinedVariable
            parameters.IntegerParameter.from_dist(
                "test",
                sp.stats.randint(0, 1),
                blaat=[0, 1],  # @UndefinedVariable
            )
            parameters.IntegerParameter.from_dist(
                "test",
                sp.stats.randint(0, 10),
                resolution=[0, 9],  # @UndefinedVariable
            )


class CategoricalParameterTestCase(unittest.TestCase):
    def test_instantiation(self):
        name = "test"
        values = ("a", "b")
        par = parameters.CategoricalParameter(name, values)

        self.assertEqual(par.name, name)
        self.assertEqual(par.resolution, [0, 1])
        self.assertEqual(par.lower_bound, 0)
        self.assertEqual(par.upper_bound, 1)

    def test_index_for_cat(self):
        name = "test"
        values = ("a", "b")
        par1 = parameters.CategoricalParameter(name, values)

        self.assertEqual(par1.index_for_cat("a"), 0)
        self.assertEqual(par1.index_for_cat("b"), 1)

        with self.assertRaises(ValueError):
            par1.index_for_cat("c")

    def test_cat_for_index(self):
        name = "test"
        values = ("a", "b")
        par1 = parameters.CategoricalParameter(name, values)

        self.assertEqual(par1.cat_for_index(0).name, "a")
        self.assertEqual(par1.cat_for_index(1).name, "b")

        with self.assertRaises(IndexError):
            par1.cat_for_index(3)

    def test_from_dist(self):
        pass


class ParametersTestCase(unittest.TestCase):
    @mock.patch("ema_workbench.em_framework.parameters.pd")
    def test_to_csv(self, mock_pandas):
        params = [
            parameters.RealParameter("a", 0.1, 1.5),
            parameters.IntegerParameter("b", 0, 10),
            parameters.CategoricalParameter("c", ["a", "b"]),
        ]

        parameters.parameters_to_csv(params, "a.csv")

        # TODO:: add assertions
        mock_pandas.DataFrame.from_dict.assert_called()


if __name__ == "__main__":
    unittest.main()


def test_parameters_csv_round_trip(tmp_path):
    """Regression test: parameters written to csv can be read back.

    Previously, the type of the parameter was not stored, the indices
    rather than the values of the categories were stored, and reading a
    csv with a categorical parameter failed.
    """
    params = [
        parameters.RealParameter("a", 0, 1.5),
        parameters.IntegerParameter("b", 1, 9),
        parameters.CategoricalParameter("c", ["x", "y", "z"]),
        parameters.BooleanParameter("d"),
        parameters.RealParameter("e", 1, 2),
    ]
    file_name = str(tmp_path / "parameters.csv")
    parameters.parameters_to_csv(params, file_name)
    loaded = parameters.parameters_from_csv(file_name)

    assert [type(p) for p in loaded] == [type(p) for p in params]
    assert [p.name for p in loaded] == [p.name for p in params]
    for original, new in zip(params, loaded):
        assert new.lower_bound == original.lower_bound
        assert new.upper_bound == original.upper_bound
    assert [c.value for c in loaded[2].categories] == ["x", "y", "z"]
    assert [c.value for c in loaded[3].categories] == [False, True]
    assert isinstance(loaded[1].lower_bound, int)


def test_parameters_from_csv_docstring_example(tmp_path):
    """Regression test: the example from the docstring of parameters_from_csv."""
    import pytest

    file_name = tmp_path / "parameters.csv"
    file_name.write_text(
        "NAME,TYPE,,,\na_real,real,0,1.1,\nan_int,int,1,9,\na_categorical,cat,a,b,c\n"
    )
    loaded = parameters.parameters_from_csv(str(file_name))

    assert isinstance(loaded[0], parameters.RealParameter)
    assert (loaded[0].lower_bound, loaded[0].upper_bound) == (0, 1.1)
    assert isinstance(loaded[1], parameters.IntegerParameter)
    assert (loaded[1].lower_bound, loaded[1].upper_bound) == (1, 9)
    assert isinstance(loaded[2], parameters.CategoricalParameter)
    assert [c.value for c in loaded[2].categories] == ["a", "b", "c"]

    # without a type column, the type is inferred
    file_name.write_text("name,,,\na_real,0,1.1,\na_categorical,a,b,c\ntwo_cats,a,b,\n")
    loaded = parameters.parameters_from_csv(str(file_name))
    assert [type(p) for p in loaded] == [
        parameters.RealParameter,
        parameters.CategoricalParameter,
        parameters.CategoricalParameter,
    ]

    with pytest.raises(ValueError):
        file_name.write_text("name,type,,,\na_real,real,0,1.1,2\n")
        parameters.parameters_from_csv(str(file_name))


def test_parameter_map_slicing():
    """Slicing a ParameterMap returns a ParameterMap."""
    parameter_map = parameters.ParameterMap()
    parameter_map.extend(
        [
            parameters.RealParameter("a", 0, 1),
            parameters.IntegerParameter("b", 0, 10, shape=(2,)),
            parameters.CategoricalParameter("c", ["x", "y"]),
        ]
    )

    subset = parameter_map[1:]
    assert isinstance(subset, parameters.ParameterMap)
    assert list(subset.keys()) == ["b", "c"]
    assert [p.name for p in subset.latent_parameters] == ["b_0", "b_1", "c"]

    # copy and extend still return a ParameterMap
    assert isinstance(subset.copy(), parameters.ParameterMap)
    assert isinstance(
        subset + parameters.RealParameter("d", 0, 1), parameters.ParameterMap
    )

    # original is unchanged
    assert list(parameter_map.keys()) == ["a", "b", "c"]
