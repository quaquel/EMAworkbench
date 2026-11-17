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


def test_named_object_map_extend_checks_type():
    """Regression test: extend only accepts objects of the kind of the map.

    Previously, any named object was accepted, so for example an outcome
    could be added to the uncertainties of a model.
    """

    class OtherVariable(util.NamedObject):
        pass

    map = NamedObjectMap(Variable)
    map.extend(Variable("a"))

    with pytest.raises(TypeError):
        map.extend(OtherVariable("b"))
    with pytest.raises(TypeError):
        map.extend([Variable("c"), OtherVariable("d")])
    with pytest.raises(TypeError):
        map.extend(["e"])
    with pytest.raises(TypeError):
        map += OtherVariable("f")

    # nothing is added if one of the items is invalid
    assert list(map.keys()) == ["a"]

    # subclasses are fine
    class SubVariable(Variable):
        pass

    map.extend([SubVariable("g")])
    assert list(map.keys()) == ["a", "g"]


def test_namedict_copy_is_independent():
    """Regression test: a copy of a NamedDict must not share its data."""
    original = util.NamedDict("a", x=1)
    copied = original.copy()
    copied["y"] = 2

    assert "y" not in original
    assert copied.name == original.name

    combined = original | {"z": 3}
    assert "z" not in original
    assert combined["z"] == 3


def test_named_object_map_mapping_methods():
    """Regression test: values, items, comparison, and membership.

    Previously, values, items, and comparing two maps raised a KeyError,
    and checking if an object is in the map was always False.
    """
    a, b = Variable("a"), Variable("b")
    map = NamedObjectMap(Variable)
    map.extend([a, b])

    assert list(map.values()) == [a, b]
    assert list(map.items()) == [("a", a), ("b", b)]
    assert dict(map.items()) == {"a": a, "b": b}

    # membership by name and by object
    assert "a" in map
    assert "c" not in map
    assert a in map
    assert Variable("a") not in map  # same name, different object

    # comparison
    other = map.copy()
    assert map == other
    assert map == other
    other.extend(Variable("c"))
    assert map != other
    assert map != {"a": a, "b": b}
    assert map != NamedObjectMap(util.NamedObject)


def test_named_object_map_slicing():
    """Tests for slicing a NamedObjectMap."""
    map = NamedObjectMap(Variable)
    map.extend([Variable(entry) for entry in "abcde"])

    # getitem with a slice returns a new map of the same class
    subset = map[1:3]
    assert isinstance(subset, NamedObjectMap)
    assert subset is not map
    assert subset.kind is map.kind
    assert list(subset.keys()) == ["b", "c"]
    assert subset["b"] is map["b"]  # objects are not copied

    # negative indices, steps, and empty slices
    assert list(map[-2:].keys()) == ["d", "e"]
    assert list(map[::2].keys()) == ["a", "c", "e"]
    assert list(map[::-1].keys()) == ["e", "d", "c", "b", "a"]
    assert len(map[10:]) == 0

    # the subset is independent of the original
    subset["f"] = Variable("f")
    del subset["b"]
    assert list(map.keys()) == ["a", "b", "c", "d", "e"]

    # delitem with a slice
    del map[1:3]
    assert list(map.keys()) == ["a", "d", "e"]

    # slice assignment is not supported
    with pytest.raises(TypeError):
        map[0:2] = [Variable("x"), Variable("y")]

    # negative int indexing
    assert map[-1].name == "e"


def test_named_object_map_setitem_by_position():
    """Regression test: assigning by integer position.

    Previously, appending stored the object under itself instead of its name,
    and replacing an entry in a map with more than one entry collapsed the
    map into a single entry.
    """
    map = NamedObjectMap(Variable)
    map.extend([Variable(entry) for entry in "abc"])

    # append at the end
    map[3] = Variable("d")
    assert list(map.keys()) == ["a", "b", "c", "d"]
    assert map["d"].name == "d"

    # replace in the middle, keeping the order
    map[1] = Variable("x")
    assert list(map.keys()) == ["a", "x", "c", "d"]
    assert all(isinstance(entry, Variable) for entry in map)

    # replace with the same name
    new_c = Variable("c")
    map[2] = new_c
    assert map["c"] is new_c
    assert list(map.keys()) == ["a", "x", "c", "d"]

    # negative positions
    map[-1] = Variable("y")
    assert list(map.keys()) == ["a", "x", "c", "y"]

    # out of range
    with pytest.raises(IndexError):
        map[5] = Variable("z")
    with pytest.raises(IndexError):
        map[-6] = Variable("z")

    # a name already used at another position is not allowed
    with pytest.raises(KeyError):
        map[0] = Variable("x")
    with pytest.raises(KeyError):
        map[4] = Variable("a")
    assert list(map.keys()) == ["a", "x", "c", "y"]
