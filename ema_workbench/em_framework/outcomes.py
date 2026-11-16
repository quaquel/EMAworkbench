"""Module for outcome classes."""

import abc
import collections
import numbers
import tarfile
import warnings
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ema_workbench.util.ema_exceptions import EMAError

from ..util import get_module_logger
from .util import Variable

# Created on 24 mei 2011
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = [
    "ArrayOutcome",
    "Constraint",
    "Outcome",
    "ScalarOutcome",
    "TimeSeriesOutcome",
    "register",
]
_logger = get_module_logger(__name__)


class Register:
    """helper class for storing outcomes to disk.

    this class stores outcomes by name, and is used to save_results
    to look up how to save each outcome.

    Raises
    ------
    ValueError if a given outcome name already exists but with a different
    outcome class.

    """

    def __init__(self):
        self.outcomes = {}

    def __call__(self, outcome):
        if outcome.name not in self.outcomes:
            self.outcomes[outcome.name] = outcome.__class__
        elif not isinstance(outcome, self.outcomes[outcome.name]):
            raise ValueError(
                "outcome with this name but of different class already exists"
            )
        else:
            pass  # multiple instances of the same class and name is fine

    def serialize(
        self, name: str, values: np.ndarray | pd.DataFrame
    ) -> tuple[BytesIO, str]:
        """Serialize the given outcome.

        Parameters
        ----------
        name : str
        values : numpy array or dataframe

        Returns
        -------
        BytesIO, str

        """
        # an outcome that has not been declared raises a KeyError, rather than
        # guessing how it should be stored
        stream, extension = self.outcomes[name].to_disk(values)

        return stream, f"{name}.{extension}"

    def deserialize(
        self, name: str, filename: str, archive: tarfile.TarFile
    ) -> np.ndarray:
        """Serialize the given outcome."""
        return self.outcomes[name].from_disk(filename, archive)


register = Register()


def _dataarray_to_disk(values: xr.DataArray) -> tuple[BytesIO, str]:
    """Write a DataArray as netCDF, which keeps the dimensions and coordinates."""
    fh = BytesIO()
    values.to_netcdf(fh, engine="h5netcdf")
    return fh, "nc"


def _dataarray_from_disk(filename: str, archive: tarfile.TarFile) -> xr.DataArray:
    """Read a DataArray that has been stored as netCDF."""
    f = archive.extractfile(filename)
    return xr.load_dataarray(BytesIO(f.read()), engine="h5netcdf")


@contextmanager
def _ignore_missing_time_index():
    """Create a TimeSeriesOutcome without the warning for a missing time index.

    For code of the workbench itself that creates a TimeSeriesOutcome without a
    time index, e.g., when loading results, so the user cannot act on the warning.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*has no time index", category=DeprecationWarning
        )
        yield


def _values_equal(a: Any, b: Any) -> bool:
    """Compare two attribute values, which can be or contain numpy arrays."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_values_equal(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    return a == b


class Outcome(Variable, metaclass=abc.ABCMeta):
    """Base Outcome class.

    Parameters
    ----------
    name : str
           Name of the outcome.
    kind : {INFO, MINIMIZE, MAXIMIZE}, optional
    variable_name : str, optional
                    if the name of the outcome in the underlying model
                    is different from the name of the outcome, you can
                    supply the variable name as an optional argument,
                    if not provided, defaults to name
    function : callable, optional
               a callable to perform postprocessing on data retrieved
               from model
    shape : {tuple, None} optional
            must be used in conjunction with dtype. Enables pre-allocation
            of data structure for storing results.
    dtype : datatype, optional
            must be used in conjunction with shape. Enables pre-allocation
            of data structure for storing results.

    Attributes
    ----------
    name : str
    kind : int
    variable_name : str
    function : callable
    shape : tuple
    dtype : dataype

    """

    __metaclass__ = abc.ABCMeta

    MINIMIZE = -1
    MAXIMIZE = 1
    INFO = 0

    def __init__(
        self,
        name: str,
        kind: int = INFO,
        variable_name: str | Sequence[str] | None = None,
        function: Callable[..., Any] | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype | type | None = None,
    ) -> None:
        """Init."""
        super().__init__(name)

        if function is not None and not callable(function):
            raise ValueError("function must be a callable")
        if variable_name:  # noqa: SIM102
            if (not isinstance(variable_name, str)) and (
                not all(isinstance(elem, str) for elem in variable_name)
            ):
                raise ValueError("variable name must be a string or list of strings")
        if (shape is not None and dtype is None) or (
            dtype is not None and shape is None
        ):
            raise ValueError("if using shape or dtype, both need to be provided")

        register(self)

        self.kind = kind

        if variable_name:
            if isinstance(variable_name, str):
                variable_name = [variable_name]

            self.variable_name = tuple(variable_name)
        else:
            self.variable_name = variable_name

        self.function = function
        self.shape = shape
        self.dtype = dtype

    def process(self, values: list[Any]) -> Any:
        """Process the values."""
        if self.function:
            var_names = self.variable_name

            n_variables = len(var_names)
            try:
                n_values = len(values)
            except TypeError:
                n_values = None

            if (n_values is None) and (n_variables == 1):
                return self.function(values)
            elif n_variables != n_values:
                raise ValueError(
                    f"Number of variables is {n_variables}, number of outputs is {n_values}. They should be equal."
                )
            else:
                return self.function(*values)
        else:
            if len(values) > 1:
                raise EMAError(
                    "More than one value returned without processing function"
                )

            return values[0]

    def __eq__(self, other):  # noqa: D105
        comparison = [
            all(
                hasattr(self, key) == hasattr(other, key)
                and _values_equal(getattr(self, key), getattr(other, key, None))
                for key in self.__dict__
            ),
            self.__class__ == other.__class__,
        ]
        return all(comparison)

    def __repr__(self, *args, **kwargs):  # noqa: D105
        klass = self.__class__.__name__
        name = self.name

        rep = f"{klass}('{name}'"

        if self.variable_name != [self.name]:
            rep += f", variable_name={self.variable_name}"
        if self.function:
            rep += f", function={self.function}"

        rep += ")"

        return rep

    def __hash__(self):  # noqa: D105
        items = [self.name, self._variable_name, self.shape]
        items = tuple(entry for entry in items if entry is not None)

        return hash(items)

    @classmethod
    @abc.abstractmethod
    def to_disk(cls, values: np.ndarray | pd.DataFrame) -> tuple[BytesIO, str]:
        """Helper function for writing outcome to disk.

        Parameters
        ----------
        values : obj
            data to store

        Returns
        -------
        BytesIO

        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_disk(cls, filename: str, archive: tarfile.TarFile) -> np.ndarray:
        """Helper function for loading from disk.

        Parameters
        ----------
        filename : str
        archive : Tarfile

        """
        raise NotImplementedError

    @classmethod
    def get_subclasses(cls):  # noqa: D102
        # fixme can be handled more cleanly using python built in stuff
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass


class ScalarOutcome(Outcome):
    """Scalar Outcome class.

    Parameters
    ----------
    name : str
           Name of the outcome.
    kind : {INFO, MINIMIZE, MAXIMIZE}, optional
    variable_name : str, optional
                    if the name of the outcome in the underlying model
                    is different from the name of the outcome, you can
                    supply the variable name as an optional argument,
                    if not provided, defaults to name
    function : callable, optional
               a callable to perform post processing on data retrieved
               from model
    dtype : datatype, optional
            Enables pre-allocation of data structure for storing results.

    Attributes
    ----------
    name : str
    kind : int
    variable_name : str
    function : callable
    shape : tuple
    dtype : datatype

    """

    def __init__(
        self,
        name: str,
        kind: int = Outcome.INFO,
        variable_name: str | Sequence[str] | None = None,
        function: Callable[..., Any] | None = None,
        dtype: np.dtype | type | None = None,
    ) -> None:
        """Init."""
        shape = None
        if dtype is not None:
            shape = (1,)
        super().__init__(
            name,
            kind,
            variable_name=variable_name,
            function=function,
            dtype=dtype,
            shape=shape,
        )

    def process(self, values):  # noqa: D102
        values = super().process(values)
        if not isinstance(values, numbers.Number):
            raise EMAError(
                f"Outcome {self.name} should be a scalar, but is a {type(values)}: {values}"
            )
        return values

    @classmethod
    def to_disk(cls, values):
        """Helper function for writing outcome to disk.

        Parameters
        ----------
        values : 1D array


        Returns
        -------
        BytesIO
        filename


        """
        fh = BytesIO()
        data = pd.DataFrame(values)
        fh.write(data.to_csv(header=False, index=False, encoding="UTF-8").encode())
        return fh, "cls"

    @classmethod
    def from_disk(cls, filename, archive):  # noqa: D102
        f = archive.extractfile(filename)
        values = pd.read_csv(f, index_col=False, header=None).values
        values = np.reshape(values, (values.shape[0],))

        return values


class ArrayOutcome(Outcome):
    """Array Outcome class for n-dimensional arrays.

    Parameters
    ----------
    name : str
           Name of the outcome.
    variable_name : str, optional
                    if the name of the outcome in the underlying model
                    is different from the name of the outcome, you can
                    supply the variable name as an optional argument,
                    if not provided, defaults to name
    function : callable, optional
               a callable to perform postprocessing on data retrieved
               from model
    shape : {tuple, None} optional
            must be used in conjunction with dtype. Enables pre-allocation
            of data structure for storing results.
    dtype : datatype, optional
            must be used in conjunction with shape. Enables pre-allocation
            of data structure for storing results.
    dims : str or tuple of str, optional
           the names of the dimensions of the outcome of a single experiment,
           e.g., ("x", "y"). If provided, the results for this outcome are
           returned as an xarray DataArray with the dimensions
           ("experiment", *dims), instead of as a numpy array. For a model
           with replications, where the replications are not aggregated by
           function, the dimensions are ("experiment", "replication", *dims).
    coords : dict, optional
             the coordinates for one or more of the dimensions, as 1D
             array-likes, e.g., {"x": [0, 10, 20]}. Can only be used
             together with dims. Dimensions without coordinates are positional.

    Attributes
    ----------
    name : str
    kind : int
    variable_name : str
    function : callable
    shape : tuple
    dtype : datatype
    dims : tuple of str, or None
    coords : dict of str to 1D array

    """

    # names of dimensions that are added by the workbench
    reserved_dims = ("experiment", "replication")

    def __init__(
        self,
        name,
        variable_name=None,
        function=None,
        shape=None,
        dtype=None,
        dims=None,
        coords=None,
    ):
        """Init."""
        super().__init__(
            name,
            variable_name=variable_name,
            function=function,
            shape=shape,
            dtype=dtype,
        )

        if dims is None and coords is not None:
            raise ValueError("coords can only be used together with dims")

        if dims is not None:
            dims = (dims,) if isinstance(dims, str) else tuple(dims)
            if not dims or not all(isinstance(dim, str) for dim in dims):
                raise ValueError("dims must be a string or a tuple of strings")
            if len(set(dims)) != len(dims):
                raise ValueError(f"the names in dims must be unique, got {dims}")
            reserved = [dim for dim in dims if dim in self.reserved_dims]
            if reserved:
                raise ValueError(
                    f"{reserved} cannot be used in dims, these dimensions are "
                    f"added by the workbench"
                )
        self.dims = dims

        self.coords = {}
        for dim, values in (coords or {}).items():
            if dim not in dims:
                raise ValueError(
                    f"coords are given for '{dim}', which is not in dims {dims}"
                )
            values = np.asarray(values) # noqa: PLW2901
            if values.ndim != 1:
                raise ValueError(f"the coords for '{dim}' must be one dimensional")
            self.coords[dim] = values

    def process(self, values):  # noqa: D102
        values = super().process(values)
        if not isinstance(values, collections.abc.Iterable):
            raise TypeError(
                f"Outcome '{self.name}' expects an iterable type (like list, tuple, or array), "
                f"but received a {type(values).__name__}."
            )
        return values

    def _determine_dims(self, ndim: int, replicated: bool) -> tuple[str, ...]:
        """Return the dimensions of the outcome of a single experiment.

        Parameters
        ----------
        ndim : int
               the number of dimensions of the outcome of a single experiment
        replicated : bool
                     whether the outcome is produced by a model with replications

        Raises
        ------
        EMAError
            if the number of dimensions does not match the declared dims

        """
        if ndim == len(self.dims):
            return self.dims
        if replicated and ndim == len(self.dims) + 1:
            return ("replication", *self.dims)

        expected = f"{len(self.dims)}"
        if replicated:
            expected += f" or, with replications, {len(self.dims) + 1}"
        raise EMAError(
            f"outcome '{self.name}' is declared with dims {self.dims}, so the "
            f"number of dimensions of the outcome of an experiment should be "
            f"{expected}, but is {ndim}"
        )

    def check_shape(
        self,
        shape: tuple[int, ...],
        replicated: bool = False,
        coords: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Check the shape of the outcome of a single experiment.

        Parameters
        ----------
        shape : tuple of int
                the shape of the outcome of a single experiment
        replicated : bool, optional
                     whether the outcome is produced by a model with replications
        coords : dict, optional
                 coordinates that are not declared on the outcome, e.g., a
                 time index returned by the model

        Raises
        ------
        EMAError
            if the shape does not match the declared dims and coords

        """
        dims = self._determine_dims(len(shape), replicated)
        all_coords = {**self.coords, **(coords or {})}
        for dim, size in zip(dims, shape):
            if dim in all_coords and all_coords[dim].size != size:
                raise EMAError(
                    f"outcome '{self.name}' has {size} values along '{dim}', but "
                    f"the coordinates for '{dim}' have {all_coords[dim].size} values"
                )

    def label_results(
        self,
        values: np.ndarray,
        replicated: bool = False,
        coords: dict[str, np.ndarray] | None = None,
    ) -> xr.DataArray:
        """Turn the results for this outcome into a DataArray.

        Parameters
        ----------
        values : ndarray
                 the results of all experiments, with the experiments along
                 the first axis
        replicated : bool, optional
                     whether the outcome is produced by a model with replications
        coords : dict, optional
                 coordinates that are not declared on the outcome, e.g., a
                 time index returned by the model

        Returns
        -------
        DataArray
            with the dimensions ("experiment", *dims), or
            ("experiment", "replication", *dims). The experiments and the
            replications are positional, so they have no coordinates.

        """
        self.check_shape(values.shape[1:], replicated=replicated, coords=coords)
        dims = ("experiment", *self._determine_dims(values.ndim - 1, replicated))
        all_coords = {**self.coords, **(coords or {})}
        return xr.DataArray(values, dims=dims, coords=all_coords, name=self.name)

    @classmethod
    def to_disk(cls, values):
        """Helper function for writing outcome to disk.

        A numpy array is stored as csv, or as npy if it has more than two
        dimensions. A DataArray is stored as netCDF, which keeps its
        dimensions and coordinates.

        Parameters
        ----------
        values : ND array or DataArray

        Returns
        -------
        BytesIO
        filename

        """
        if isinstance(values, xr.DataArray):
            return _dataarray_to_disk(values)

        if values.ndim < 3:
            fh = BytesIO()
            data = pd.DataFrame(values)
            fh.write(data.to_csv(header=False, index=False, encoding="UTF-8").encode())
            extension = "csv"
        else:
            fh = BytesIO()
            np.save(fh, values)
            extension = "npy"

        return fh, extension

    @classmethod
    def from_disk(cls, filename, archive):  # noqa: D102
        if filename.endswith(".nc"):
            return _dataarray_from_disk(filename, archive)

        f = archive.extractfile(filename)

        if filename.endswith("csv"):
            return pd.read_csv(f, index_col=False, header=None).values
        elif filename.endswith("npy"):
            array_file = BytesIO()
            array_file.write(f.read())
            array_file.seek(0)
            return np.load(array_file)
        else:
            raise EMAError(
                f"Unknown file extension {filename}. Only .csv, .npy, and .nc are supported."
            )


class TimeSeriesOutcome(ArrayOutcome):
    """TimeSeries Outcome class.

    Parameters
    ----------
    name : str
           Name of the outcome.
    variable_name : str, optional
                    if the name of the outcome in the underlying model
                    is different from the name of the outcome, you can
                    supply the variable name as an optional argument,
                    if not provided, defaults to name
    function : callable, optional
               a callable to perform postprocessing on data retrieved
               from model
    shape : {tuple, None} optional
            must be used in conjunction with dtype. Enables pre-allocation
            of data structure for storing results.
    dtype : datatype, optional
            must be used in conjunction with shape. Enables pre-allocation
            of data structure for storing results.
    time : 1D array-like, optional
           the time index of the time series, e.g.,
           np.arange(0, 48.25, 0.25). Use this if the index is known up front.
    time_variable : str, optional
                    the name of the output of the model that contains the
                    time index. Use this if the index comes from the model.
                    The index has to be the same for all experiments.
                    Cannot be used together with time.

    Attributes
    ----------
    name : str
    kind : int
    variable_name : str
    function : callable
    shape : tuple
    dtype : datatype
    dims : tuple of str, or None
    coords : dict of str to 1D array
    time_variable : str, or None

    Notes
    -----
    If time or time_variable is provided, the results for this outcome are
    returned as an xarray DataArray with the dimensions ("experiment", "time"),
    and the time index as coordinates of the time dimension. For a model with
    replications, where the replications are not aggregated by function, the
    dimensions are ("experiment", "replication", "time").

    Without time or time_variable, the results are returned as a numpy array.
    This is deprecated: from version 4.0 of the workbench, the results of all
    time series outcomes are returned as a DataArray.

    All time series of an outcome need to have the same length and the same
    time index.

    """

    def __init__(
        self,
        name,
        variable_name=None,
        function=None,
        shape=None,
        dtype=None,
        time=None,
        time_variable=None,
    ):
        """Init."""
        if time is not None and time_variable is not None:
            raise ValueError("use either time or time_variable, not both")
        if time_variable is not None and not isinstance(time_variable, str):
            raise ValueError("time_variable must be a string")

        dims = None
        coords = None
        if time is not None:
            dims = ("time",)
            coords = {"time": time}
        elif time_variable is not None:
            dims = ("time",)
        else:
            warnings.warn(
                f"TimeSeriesOutcome '{name}' has no time index, so its results are "
                f"returned as a numpy array. This is deprecated: from version 4.0, "
                f"the results of all time series outcomes are returned as an xarray "
                f"DataArray. Specify the time index with time or time_variable to "
                f"switch now.",
                DeprecationWarning,
                stacklevel=2,
            )

        super().__init__(
            name,
            variable_name=variable_name,
            function=function,
            shape=shape,
            dtype=dtype,
            dims=dims,
            coords=coords,
        )
        self.time_variable = time_variable

    def attach_time(self, value: Any, time: Any) -> xr.DataArray:
        """Attach the time index returned by the model to the outcome of an experiment.

        Parameters
        ----------
        value : array-like
                the time series of a single experiment, or a time series for
                each replication of the experiment
        time : array-like
               the time index as returned by the model, or the time index for
               each replication of the experiment

        Returns
        -------
        DataArray
            with the dimensions ("time",) or ("replication", "time")

        Raises
        ------
        EMAError
            if the time index differs between replications, or does not match
            the time series

        """
        value = np.asarray(value)
        time = np.asarray(time)

        if time.ndim == 2:
            # a time index for each replication
            if not (time == time[0]).all():
                raise EMAError(
                    f"the time index '{self.time_variable}' of outcome "
                    f"'{self.name}' differs between replications"
                )
            time = time[0]

        if time.ndim != 1:
            raise EMAError(
                f"the time index '{self.time_variable}' of outcome '{self.name}' "
                f"should be one dimensional, but has {time.ndim} dimensions"
            )
        if value.ndim not in (1, 2):
            raise EMAError(
                f"outcome '{self.name}' should be a time series, or a time series "
                f"for each replication, but has {value.ndim} dimensions"
            )
        if value.shape[-1] != time.size:
            raise EMAError(
                f"outcome '{self.name}' has {value.shape[-1]} time steps, but the "
                f"time index '{self.time_variable}' has {time.size} values"
            )

        dims = ("time",) if value.ndim == 1 else ("replication", "time")
        return xr.DataArray(value, dims=dims, coords={"time": time})

    @classmethod
    def to_disk(cls, values):
        """Helper function for writing outcome to disk.

        A numpy array is stored as csv, or as npy if it has more than two
        dimensions. A DataArray, which is what the results are if the outcome
        has a time index, is stored as netCDF, which keeps its dimensions and
        the time index.

        Parameters
        ----------
        values : 2D array or DataArray


        Returns
        -------
        BytesIO
        filename

        """
        if isinstance(values, xr.DataArray):
            return _dataarray_to_disk(values)

        if values.ndim > 2:
            # time series for each replication, which do not fit in a csv
            return super().to_disk(values)

        warnings.warn(
            "Timeseries outcome seralization still to be tested!!", stacklevel=2
        )
        fh = BytesIO()
        data = pd.DataFrame(values)
        fh.write(data.to_csv(header=True, index=False, encoding="UTF-8").encode())
        return fh, "csv"

    @classmethod
    def from_disk(cls, filename, archive):  # noqa: D102
        if filename.endswith((".nc", ".npy")):
            return super().from_disk(filename, archive)

        f = archive.extractfile(filename)

        if filename.endswith("csv"):
            return pd.read_csv(f, index_col=False, header=0).values
        else:
            raise EMAError(
                f"Unknown file extension {filename}. Only .csv, .npy, and .nc are supported."
            )


class Constraint(ScalarOutcome):
    """Constraints class that can be used when defining constrained optimization problems.

    Parameters
    ----------
    name : str
    parameter_names : str or collection of str
    outcome_names : str or collection of str
    function : callable

    Attributes
    ----------
    name : str
    function : callable
               The function should return the distance from the feasibility
               threshold, given the model outputs with a variable name. The
               distance should be 0 if the constraint is met.
    parameter_names : str, list of str
                      name(s) of the uncertain parameter(s) and/or
                      lever parameter(s) to which the constraint applies
    outcome_names : str, list of str
                    name(s) of the outcome(s) to which the constraint applies

    """

    def __init__(
        self,
        name,
        function: Callable,
        parameter_names: list[str] | str | None = None,
        outcome_names: list[str] | str | None = None,
    ):
        """Init."""
        if not callable(function):
            raise ValueError("function must be a callable")

        if not parameter_names:
            parameter_names = []
        elif isinstance(parameter_names, str):
            parameter_names = [parameter_names]

        if not outcome_names:
            outcome_names = []
        elif isinstance(outcome_names, str):
            outcome_names = [outcome_names]

        variable_names = parameter_names + outcome_names
        if len(variable_names) == 0:
            raise ValueError(
                "No decision variable names or objective names to which constraint applies provided"
            )

        super().__init__(
            name,
            kind=Outcome.INFO,
            variable_name=variable_names,
            function=function,
        )

        self.parameter_names = parameter_names
        self.outcome_names = outcome_names

    def process(self, values):  # noqa: D102
        value = super().process(values)
        assert value >= 0
        return value


def create_outcomes(outcomes, **kwargs):
    """Helper function for creating multiple outcomes.

    Parameters
    ----------
    outcomes : DataFrame, or something convertible to a DataFrame
               in case of string, the string will be passed

    Returns
    -------
    list

    """
    if isinstance(outcomes, str):
        outcomes = pd.read_csv(outcomes, **kwargs)
    elif not isinstance(outcomes, pd.DataFrame):
        outcomes = pd.DataFrame.from_dict(outcomes)

    for entry in ["name", "type"]:
        if entry not in outcomes.columns:
            raise ValueError(f"No {entry} column in outcome DataFrame")

    temp_outcomes = []
    for _, row in outcomes.iterrows():
        name = row["name"]
        kind = row["type"]

        if kind == "scalar":
            outcome = ScalarOutcome(name)
        elif kind == "timeseries":
            # the time index cannot be specified here
            with _ignore_missing_time_index():
                outcome = TimeSeriesOutcome(name)
        else:
            raise ValueError(
                f"Unknown type for outcome {name}: {kind}. Should be 'scalar' or 'timeseries'"
            )
        temp_outcomes.append(outcome)
    return temp_outcomes
