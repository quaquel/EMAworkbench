"""


"""

import unittest
import copy
from ema_workbench.em_framework import util


class TestNamedObject(unittest.TestCase):
    pass


class TestNamedObjectMap(unittest.TestCase):
    pass


class TestNamedDict(unittest.TestCase):
    def test_namedict(self):
        name = "test"
        kwargs = {"a": 1, "b": 2}

        nd = util.NamedDict(name, **kwargs)

        self.assertEqual(nd.name, name, "name not equal")

        for key, value in nd.items():
            self.assertEqual(kwargs[key], value, "kwargs not set on inner dict correctly")

        kwargs = {"a": 1, "b": 2}

        nd = util.NamedDict(**kwargs)

        self.assertEqual(nd.name, repr(kwargs), "name not equal")
        for key, value in nd.items():
            self.assertEqual(kwargs[key], value, "kwargs not set on inner dict correctly")

        # test len
        self.assertEqual(2, len(nd), "length not correct")

        # test in
        for entry in kwargs:
            self.assertIn(entry, nd, f"{entry} not in NamedDict")

        # test addition
        nd["c"] = 3
        self.assertIn("c", nd, "additional item not added")
        self.assertEqual(3, nd["c"])

        # test removal
        del nd["c"]
        self.assertNotIn("c", nd, "item not removed successfully")


class TestCombine(unittest.TestCase):
    def test_combine_two_dicts_no_overlap(self):
        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3, "d": 4}
        expected = {"a": 1, "b": 2, "c": 3, "d": 4}
        result = util.combine(dict1, dict2)
        self.assertEqual(result, expected)

    def test_combine_multiple_dicts_no_overlap(self):
        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3, "d": 4}
        dict3 = {"e": 5, "f": 6}
        expected = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}
        result = util.combine(dict1, dict2, dict3)
        self.assertEqual(result, expected)

    def test_combine_two_dicts_with_overlap(self):
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 2, "c": 3}
        with self.assertRaises(util.EMAError):
            util.combine(dict1, dict2)

    def test_combine_multiple_dicts_with_overlap(self):
        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3, "d": 4}
        dict3 = {"a": 1, "f": 6}
        with self.assertRaises(util.EMAError):
            util.combine(dict1, dict2, dict3)

    def test_combine_empty_dicts(self):
        dict1 = {}
        dict2 = {}
        expected = {}
        result = util.combine(dict1, dict2)
        self.assertEqual(result, expected)

    def test_combine_single_dict(self):
        dict1 = {"a": 1, "b": 2}
        expected = copy.deepcopy(dict1)
        result = util.combine(dict1)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
