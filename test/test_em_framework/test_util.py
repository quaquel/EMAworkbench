"""tests for em_framework.util."""

import copy

import pytest

from ema_workbench.em_framework import util
from ema_workbench.em_framework.util import NamedObjectMap, Variable


def test_namedict():
    """Test NamedDict."""
    name = "test"
    kwargs = {"a": 1, "b": 2}

    nd = util.NamedDict(name, **kwargs)

    assert nd.name == name, "name not equal"

    for key, value in nd.items():
        assert kwargs[key] == value, "kwargs not set on inner dict correctly"

    kwargs = {"a": 1, "b": 2}

    nd = util.NamedDict(**kwargs)

    assert nd.name == repr(kwargs), "name not equal"
    for key, value in nd.items():
        assert kwargs[key] == value, "kwargs not set on inner dict correctly"

    # test len
    assert len(nd) == 2, "length not correct"

    # test in
    for entry in kwargs:
        assert entry in nd, f"{entry} not in NamedDict"

    # test addition
    nd["c"] = 3
    assert "c" in nd, "additional item not added"
    assert nd["c"] == 3

    # test removal
    del nd["c"]
    assert "c" not in nd, "item not removed successfully"


def test_combine_two_dicts_no_overlap():
    """Tests for combine."""
    dict1 = {"a": 1, "b": 2}
    dict2 = {"c": 3, "d": 4}
    expected = {"a": 1, "b": 2, "c": 3, "d": 4}
    result = util.combine(dict1, dict2)
    assert result == expected


def test_combine_multiple_dicts_no_overlap():
    """Tests for combine."""
    dict1 = {"a": 1, "b": 2}
    dict2 = {"c": 3, "d": 4}
    dict3 = {"e": 5, "f": 6}
    expected = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}
    result = util.combine(dict1, dict2, dict3)
    assert result == expected


def test_combine_two_dicts_with_overlap():
    """Tests for combine."""
    dict1 = {"a": 1, "b": 2}
    dict2 = {"b": 2, "c": 3}
    with pytest.raises(util.EMAError):
        util.combine(dict1, dict2)


def test_combine_multiple_dicts_with_overlap():
    """Tests for combine."""
    dict1 = {"a": 1, "b": 2}
    dict2 = {"c": 3, "d": 4}
    dict3 = {"a": 1, "f": 6}
    with pytest.raises(util.EMAError):
        util.combine(dict1, dict2, dict3)


def test_combine_empty_dicts():
    """Tests for combine."""
    dict1 = {}
    dict2 = {}
    expected = {}
    result = util.combine(dict1, dict2)
    assert result == expected


def test_combine_single_dict():
    """Tests for combine."""
    dict1 = {"a": 1, "b": 2}
    expected = copy.deepcopy(dict1)
    result = util.combine(dict1)
    assert result == expected



def test_named_object_map():
    """Tests for NamedObjectMap."""
    with pytest.raises(TypeError):
        util.NamedObjectMap(float)

    # test __setitem__
    map = NamedObjectMap(Variable)
    map[0] = Variable("a")
    assert len(map) == 1

    map[0] = Variable("some_other_a")
    assert len(map) == 1

    with pytest.raises(IndexError):
        map[3] = Variable("b")
    with pytest.raises(KeyError):
        map["c"] = Variable("b")
    with pytest.raises(TypeError):
        map["c"] = "b"

    map["b"] = Variable("b")
    assert len(map) == 2

    # test __getitem__
    map = NamedObjectMap(Variable)
    for name in "abcd":
        map[name] = Variable(name)
    assert len(map) == 4

    assert map["a"] == map[0]
    assert map["b"].name == "b"
    assert map[1].name == "b"

    # test __delitem__
    v = map["a"]
    del map["a"]
    assert len(map) == 3
    assert v not in map

    v = map[0]
    del map[0]
    assert len(map) == 2
    assert v not in map

    # map.clear()
    map.clear()
    assert len(map) == 0

    # map.extend()
    map.extend(Variable("a"))
    assert len(map) == 1

    map.extend([Variable(entry) for entry in "bcd"])
    assert len(map) == 4

    with pytest.raises(TypeError):
        map.extend(1235465)

    # + operator
    map = NamedObjectMap(Variable)
    map = map + [Variable(entry) for entry in "abcd"]
    assert len(map) == 4

    map = NamedObjectMap(Variable)
    map += [Variable(entry) for entry in "abcd"]
    assert len(map) == 4

    # iter and keys
    keys = list(map.keys())
    assert keys == ["a", "b", "c", "d"]

    for entry in map:
        assert entry == map[entry.name]

