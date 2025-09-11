"""utilities used throughout em_framework."""

# Created on Jul 16, 2016
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


import itertools
from collections import OrderedDict, UserDict

import tqdm

from ..util import EMAError

__all__ = [
    "Counter",
    "NamedDict",
    "NamedObject",
    "NamedObjectMap",
    "NamedObjectMapDescriptor",
    "ProgressTrackingMixIn",
    "combine",
    "determine_objects",
    "representation",
]


class NamedObject:
    """Base object with a name attribute."""

    def __init__(self, name:str):
        self.name:str = name


class Counter:
    """helper function for generating counter based names for NamedDicts."""

    def __init__(self, startfrom=0):
        self._counter = itertools.count(startfrom)

    def __call__(self, *args):# noqa: D102
        return next(self._counter)


def representation(named_dict):
    """Helper function for generating repr based names for NamedDicts."""
    return repr(named_dict)


class Variable(NamedObject):
    """Root class for input parameters and outcomes."""

    @property
    def variable_name(self):
        if self._variable_name is None:
            return [self.name]
        else:
            return self._variable_name

    @variable_name.setter
    def variable_name(self, name):
        if isinstance(name, str):
            name = [name]
        self._variable_name = name

    def __init__(self, name:str, variable_name:str|list[str]|None=None):
        if not name.isidentifier():
            raise DeprecationWarning(
                f"'{name}' is not a valid Python identifier. Starting from version 3.0 of the EMAworkbench, names must be valid python identifiers"
            )
        super().__init__(name)
        self.variable_name = variable_name


class NamedObjectMap:
    """A named object mapping class."""

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self._data = OrderedDict()

        if not issubclass(kind, NamedObject):
            raise TypeError(
                f"Type must be a (subclass of a) NamedObject, not {type(kind)}"
            )

    def clear(self): # noqa: D102
        self._data = OrderedDict()

    def copy(self): # noqa: D102
        copy = NamedObjectMap(self.kind)
        copy._data = self._data.copy()

        return copy

    def __len__(self): # noqa: D105
        return len(self._data)

    def __getitem__(self, key): # noqa: D105
        if isinstance(key, int):
            for i, (_, v) in enumerate(self._data.items()):
                if i == key:
                    return v
            raise KeyError(key)
        else:
            return self._data[key]

    def __setitem__(self, key, value): # noqa: D105
        if not isinstance(value, self.kind):
            raise TypeError(
                f"Can only add {self.kind.__name__} objects, not {type(value)}"
            )

        if isinstance(key, int):
            self._data = OrderedDict(
                [
                    (value.name, value) if i == key else (k, v)
                    for i, (k, v) in enumerate(self._data.items())
                ]
            )
        else:
            if value.name != key:
                raise ValueError(
                    f"Key ({key}) does not match name of {self.kind.__name__}"
                )

            self._data[key] = value

    def __delitem__(self, key): # noqa: D105
        del self._data[key]

    def __iter__(self): # noqa: D105
        return iter(self._data.values())

    def __contains__(self, item): # noqa: D105
        return item in self._data

    def extend(self, value): # noqa: D102
        if isinstance(value, NamedObject):
            self._data[value.name] = value
        elif hasattr(value, "__iter__"):
            for item in value:
                self._data[item.name] = item
        else:
            raise TypeError(f"Can only add {type!s} objects")

    def __add__(self, value): # noqa: D105
        data = self.copy()
        data.extend(value)
        return data

    def __iadd__(self, value): # noqa: D105
        self.extend(value)
        return self

    def keys(self): # noqa: D102
        return self._data.keys()


class NamedObjectMapDescriptor:
    """Descriptor class for named objects."""

    def __init__(self, kind):
        self.kind = kind

    def __get__(self, instance, owner): # noqa: D105
        if instance is None:
            return self
        try:
            return getattr(instance, self.internal_name)
        except AttributeError:
            mapping = NamedObjectMap(self.kind)  # @ReservedAssignment
            setattr(instance, self.internal_name, mapping)
            return mapping

    def __set__(self, instance, values): # noqa: D105
        try:
            mapping = getattr(instance, self.internal_name)  # @ReservedAssignment
        except AttributeError:
            mapping = NamedObjectMap(self.kind)  # @ReservedAssignment
            setattr(instance, self.internal_name, mapping)

        mapping.extend(values)

    def __set_name__(self, owner, name): # noqa: D105
        self.name = name
        self.internal_name = "_" + name


class NamedDict(UserDict, NamedObject):
    """Named dictionary class."""

    def __init__(self, name=representation, **kwargs):
        super().__init__(**kwargs)
        if name is None:
            raise ValueError()
        elif callable(name):
            name = name(self)
        self.name = name


def combine(*args):
    """Combine scenario and policy into a single experiment dict.

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
    experiment = {}

    for entry in args:
        for key, value in entry.items():
            if key in experiment:
                raise EMAError(
                    f"Parameters exist in both {experiment} and {entry}, overlap is {key}"
                )
            experiment[key] = value

    return experiment


def determine_objects(models, attribute, union=True):
    """Determine the parameters over which to sample.

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


class ProgressTrackingMixIn:
    """Mixin for monitoring progress.

    Parameters
    ----------
    N : int
        total number of experiments
    reporting_interval : int
                         nfe between logging progress
    logger : logger instance
    log_progress : bool, optional
    log_func : callable, optional
               function called with self as only argument, should invoke
               self._logger with custom log message

    Attributes
    ----------
    i : int
    reporting_interval : int
    log_progress : bool
    log_func : callable
    pbar : {None, tqdm.tqdm instance}
           if log_progress is true, None, if false tqdm.tqdm instance


    """

    def __init__(
        self,
        N, # noqa: N803
        reporting_interval,
        logger,
        log_progress=False,
        log_func=lambda self: self._logger.info(f"{self.i} experiments completed"),
    ):
        """Init."""
        # TODO:: how to enable variable log messages which might include
        # different attributes?

        self.i = 0
        self.reporting_interval = reporting_interval
        self._logger = logger
        self.log_progress = log_progress
        self.log_func = log_func

        if not log_progress:
            self.pbar = tqdm.tqdm(total=N, ncols=79)

    def __call__(self, n): # noqa: D102
        self.i += n
        self._logger.debug(f"{self.i} experiments performed")

        if not self.log_progress:
            self.pbar.update(n=n)

            if self.i >= self.pbar.total:
                self.close()
        else:
            if self.i % self.reporting_interval == 0:
                self.log_func(self)

    def close(self): # noqa: D102
        try:
            self.pbar.__exit__(None, None, None)
        except AttributeError:
            pass
