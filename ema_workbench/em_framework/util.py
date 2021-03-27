"""utilities used throughout em_framework"""
import copy
from collections import OrderedDict

from collections import UserDict

import itertools

from ..util import EMAError

# from .parameters import Parameter

# Created on Jul 16, 2016
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['NamedObject', 'NamedDict', 'Counter', 'representation']


class NamedObject:

    def __init__(self, name):
        self.name = name


class Counter:
    """helper function for generating counter based names for NamedDicts"""

    def __init__(self, startfrom=0):
        self._counter = itertools.count(startfrom)

    def __call__(self, *args):
        return next(self._counter)


def representation(named_dict):
    """helper function for generating repr based names for NamedDicts"""
    return repr(named_dict)


class Variable(NamedObject):
    """Root class for input parameters and outcomes """

    @property
    def variable_name(self):
        if self._variable_name is not None:
            return self._variable_name
        else:
            return [self.name]

    @variable_name.setter
    def variable_name(self, name):
        if isinstance(name, str):
            name = [name]
        self._variable_name = name


class NamedObjectMap:

    def __init__(self, type):  # @ReservedAssignment
        super(NamedObjectMap, self).__init__()
        self.type = type
        self._data = OrderedDict()

        if not issubclass(type, NamedObject):
            raise TypeError("type must be a NamedObject")

    def clear(self):
        self._data = OrderedDict()

    def copy(self):
        copy = NamedObjectMap(self.type)
        copy._data = self._data.copy()

        return copy

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        if isinstance(key, int):
            for i, (_, v) in enumerate(self._data.items()):
                if i == key:
                    return v
            raise KeyError(key)
        else:
            return self._data[key]

    def __setitem__(self, key, value):
        if not isinstance(value, self.type):
            raise TypeError("can only add " + self.type.__name__ + " objects")

        if isinstance(key, int):
            self._data = OrderedDict([(value.name, value) if i == key else (
                k, v) for i, (k, v) in enumerate(self._data.items())])
        else:
            if value.name != key:
                raise ValueError(
                    "key does not match name of " + self.type.__name__)

            self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data.values())

    def __contains__(self, item):
        return item in self._data

    def extend(self, value):
        if isinstance(value, NamedObject):
            self._data[value.name] = value
        elif hasattr(value, "__iter__"):
            for item in value:
                self._data[item.name] = item
        else:
            raise TypeError("can only add " + str(type) + " objects")

    def __add__(self, value):
        self.extend(value)
        return self

    def __iadd__(self, value):
        self.extend(value)
        return self

    def keys(self):
        return self._data.keys()


class NamedObjectMapDescriptor:
    def __init__(self, kind):
        self.kind = kind

    def __get__(self, instance, owner):
        if instance is None:
            return self
        try:
            return getattr(instance, self.internal_name)
        except AttributeError:
            map = NamedObjectMap(self.kind)  # @ReservedAssignment
            setattr(instance, self.internal_name, map)
            return map

    def __set__(self, instance, values):
        try:
            map = getattr(instance, self.internal_name)  # @ReservedAssignment
        except AttributeError:
            map = NamedObjectMap(self.kind)  # @ReservedAssignment
            setattr(instance, self.internal_name, map)

        map.extend(values)

    def __set_name__(self, owner, name):
        self.name = name
        self.internal_name = '_' + name


class NamedDict(UserDict, NamedObject):

    def __init__(self, name=representation, **kwargs):
        super(NamedDict, self).__init__(**kwargs)
        if name is None:
            raise ValueError()
        elif callable(name):
            name = name(self)
        self.name = name


def combine(*args):
    """combine scenario and policy into a single experiment dict

    Parameters
    ----------
    args : two or more dicts that need to be combined


    Returns
    -------
    a single unified dict containing the entries from all dicts

    Raises
    ------
    EMAError
        if a keyword argument exists in more than one dict
    """
    experiment = copy.deepcopy(args[0])
    for entry in args[1::]:
        overlap = set(experiment.keys()).intersection(set(entry.keys()))
        if overlap:
            raise EMAError(
                'parameters exist in {} and {}, overlap is {}'.format(
                    experiment, entry, overlap))
        experiment.update(entry)

    return experiment


def determine_objects(models, attribute, union=True):
    """determine the parameters over which to sample

    Parameters
    ----------
    models : a collection of AbstractModel instances
    attribute : {'uncertainties', 'levers', 'outcomes'}
    union : bool, optional
            in case of multiple models, sample over the union of
            levers, or over the intersection of the levers

    Returns
    -------
    collection of Parameter instances

    """
    try:
        models = iter(models)
    except TypeError:
        # we assume that there is only a single model passed
        models = iter([models])

    named_objects = getattr(next(models), attribute).copy()
    intersection = set(named_objects.keys())

    # gather parameters across all models
    for model in models:
        model_params = getattr(model, attribute)

        # relies on name based identity, do we want that?
        named_objects.extend(model_params)

        intersection = intersection.intersection(model_params.keys())

    # in case not union, remove all parameters not in intersection
    if not union:
        params_to_remove = set(named_objects.keys()) - intersection
        for key in params_to_remove:
            del named_objects[key]
    return named_objects
