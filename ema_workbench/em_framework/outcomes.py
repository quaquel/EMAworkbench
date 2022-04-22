"""
Module for outcome classes

"""
import abc
import collections
import numbers
import warnings
from io import BytesIO

import numpy as np
import pandas as pd

from ema_workbench.util.ema_exceptions import EMAError
from .util import Variable
from ..util import get_module_logger

# Created on 24 mei 2011
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = [
    "ScalarOutcome",
    "ArrayOutcome",
    "TimeSeriesOutcome",
    "Constraint",
    "AbstractOutcome",
    "register",
]
_logger = get_module_logger(__name__)


class Register:
    """helper class for storing outcomes to disk

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
                "outcome with this name but of different class " "already exists"
            )
        else:
            pass  # multiple instances of the same class and name is fine

    def serialize(self, name, values):
        """

        Parameters
        ----------
        name : str
        values : numpy array or dataframe

        Returns
        -------
        BytesIO, str

        """

        try:
            stream, extension = self.outcomes[name].to_disk(values)
        except KeyError:
            _logger.warning(
                "outcome not defined, falling back on " "ArrayOutcome.to_disk"
            )
            stream, extension = ArrayOutcome.to_disk(values)

        return stream, f"{name}.{extension}"

    def deserialize(self, name, filename, archive):
        return self.outcomes[name].from_disk(filename, archive)


register = Register()


class AbstractOutcome(Variable):
    """
    Base Outcome class

    Parameters
    ----------
    name : str
           Name of the outcome.
    kind : {INFO, MINIMZE, MAXIMIZE}, optional
    variable_name : str, optional
                    if the name of the outcome in the underlying model
                    is different from the name of the outcome, you can
                    supply the variable name as an optional argument,
                    if not provided, defaults to name
    function : callable, optional
               a callable to perform postprocessing on data retrieved
               from model
    expected_range : 2 tuple, optional
                     expected min and max value for outcome,
                     used by HyperVolume convergence metric
    shape : {tuple, None} optional

    Attributes
    ----------
    name : str
    kind : int
    variable_name : str
    function : callable
    shape : tuple

    """

    __metaclass__ = abc.ABCMeta

    MINIMIZE = -1
    MAXIMIZE = 1
    INFO = 0

    def __init__(
        self,
        name,
        kind=INFO,
        variable_name=None,
        function=None,
        expected_range=None,
        shape=None,
    ):
        super().__init__(name)

        if function is not None and not callable(function):
            raise ValueError("function must be a callable")
        if variable_name:
            if (not isinstance(variable_name, str)) and (
                not all(isinstance(elem, str) for elem in variable_name)
            ):
                raise ValueError("variable name must be a string or list of strings")
        if expected_range is not None and len(expected_range) != 2:
            raise ValueError("expected_range must be a min-max tuple")

        register(self)

        self.kind = kind

        if variable_name:
            if isinstance(variable_name, str):
                variable_name = [
                    variable_name,
                ]

            self.variable_name = tuple(variable_name)
        else:
            self.variable_name = variable_name

        self.function = function
        self._expected_range = expected_range
        self.shape = shape

    def process(self, values):
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
                    ("number of variables is {}, " "number of outputs is {}").format(
                        n_variables, n_values
                    )
                )
            else:
                return self.function(*values)
        else:
            if len(values) > 1:
                raise EMAError(
                    "more than one value returned without " "processing function"
                )

            return values[0]

    def __eq__(self, other):
        comparison = [
            all(
                hasattr(self, key) == hasattr(other, key)
                and getattr(self, key) == getattr(other, key)
                for key in self.__dict__.keys()
            ),
            self.__class__ == other.__class__,
        ]
        return all(comparison)

    def __repr__(self, *args, **kwargs):
        klass = self.__class__.__name__
        name = self.name

        rep = f"{klass}('{name}'"

        if self.variable_name != [self.name]:
            rep += f", variable_name={self.variable_name}"
        if self.function:
            rep += f", function={self.function}"

        rep += ")"

        return rep

    def __hash__(self):
        items = [self.name, self._variable_name, self._expected_range, self.shape]
        items = tuple(entry for entry in items if entry is not None)

        return hash(items)

    @classmethod
    @abc.abstractmethod
    def to_disk(cls, values):
        """helper function for writing outcome to disk

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
    def from_disk(cls, filename, archive):
        """helper function for loading from disk

        Parameters
        ----------
        filename : str
        archive : Tarfile

        Returns
        -------

        """
        raise NotImplementedError

    @classmethod
    def get_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass


class ScalarOutcome(AbstractOutcome):
    """
    Scalar Outcome class

    Parameters
    ----------
    name : str
           Name of the outcome.
    kind : {INFO, MINIMZE, MAXIMIZE}, optional
    variable_name : str, optional
                    if the name of the outcome in the underlying model
                    is different from the name of the outcome, you can
                    supply the variable name as an optional argument,
                    if not provided, defaults to name
    function : callable, optional
               a callable to perform post processing on data retrieved
               from model
    expected_range : collection, optional
                     expected min and max value for outcome,
                     used by HyperVolume convergence metric

    Attributes
    ----------
    name : str
    kind : int
    variable_name : str
    function : callable
    shape : tuple
    expected_range : tuple

    """

    @property
    def expected_range(self):
        if self._expected_range is None:
            raise ValueError(f"no expected_range is set for {self.variable_name}")
        return self._expected_range

    @expected_range.setter
    def expected_range(self, expected_range):
        self._expected_range = expected_range

    def __init__(
        self,
        name,
        kind=AbstractOutcome.INFO,
        variable_name=None,
        function=None,
        expected_range=None,
    ):
        super().__init__(name, kind, variable_name=variable_name, function=function)
        self.expected_range = expected_range

    def process(self, values):
        values = super().process(values)
        if not isinstance(values, numbers.Number):
            raise EMAError(
                f"outcome {self.name} should be a scalar, but is"
                f" {type(values)}: {values}"
            )
        return values

    @classmethod
    def to_disk(cls, values):
        """helper function for writing outcome to disk


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
    def from_disk(cls, filename, archive):
        f = archive.extractfile(filename)
        values = pd.read_csv(f, index_col=False, header=None).values
        values = np.reshape(values, (values.shape[0],))

        return values


class ArrayOutcome(AbstractOutcome):
    """Array Outcome class for n-dimensional collections

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
    expected_range : 2 tuple, optional
                     expected min and max value for outcome,
                     used by HyperVolume convergence metric
    shape : {tuple, None}, optional

    Attributes
    ----------
    name : str
    kind : int
    variable_name : str
    function : callable
    shape : tuple
    expected_range : tuple


    """

    def __init__(
        self, name, variable_name=None, function=None, expected_range=None, shape=None
    ):
        super().__init__(
            name,
            variable_name=variable_name,
            function=function,
            expected_range=expected_range,
            shape=shape,
        )

    def process(self, values):
        values = super().process(values)
        if not isinstance(values, collections.abc.Iterable):
            raise EMAError(f"outcome {self.name} should be a collection")
        return values

    @classmethod
    def to_disk(cls, values):
        """helper function for writing outcome to disk

        Parameters
        ----------
        values : ND array

        Returns
        -------
        BytesIO
        filename

        """

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
    def from_disk(cls, filename, archive):
        f = archive.extractfile(filename)

        if filename.endswith("csv"):
            return pd.read_csv(f, index_col=False, header=None).values
        elif filename.endswith("npy"):
            array_file = BytesIO()
            array_file.write(f.read())
            array_file.seek(0)
            return np.load(array_file)
        else:
            raise EMAError("unknown file extension")


class TimeSeriesOutcome(ArrayOutcome):
    """
    TimeSeries Outcome class

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
    expected_range : 2 tuple, optional
                     expected min and max value for outcome,
                     used by HyperVolume convergence metric
    shape : {tuple, None}, optional

    Attributes
    ----------
    name : str
    kind : int
    variable_name : str
    function : callable
    shape : tuple
    expected_range : tuple

    """

    def __init__(
        self, name, variable_name=None, function=None, expected_range=None, shape=None
    ):
        super().__init__(
            name,
            variable_name=variable_name,
            function=function,
            expected_range=expected_range,
            shape=shape,
        )

    @classmethod
    def to_disk(cls, values):
        """helper function for writing outcome to disk

        Parameters
        ----------
        values : DataFrame


        Returns
        -------
        StringIO
        filename

        """
        warnings.warn("still to be tested!!")
        fh = BytesIO()
        data = pd.DataFrame(values)
        fh.write(data.to_csv(header=True, index=False, encoding="UTF-8").encode())
        return fh, "csv"

    @classmethod
    def from_disk(cls, filename, archive):
        f = archive.extractfile(filename)

        if filename.endswith("csv"):
            return pd.read_csv(f, index_col=False, header=0).values
        else:
            raise EMAError("unknown file extension")


class Constraint(ScalarOutcome):
    """Constraints class that can be used when defining constrained
    optimization problems.

    Parameters
    ----------
    name : str
    parameter_names : str or collection of str
    outcome_names : str or collection of str
    function : callable

    Attributes
    ----------
    name : str
    parameter_names : str, list of str
                      name(s) of the uncertain parameter(s) and/or
                      lever parameter(s) to which the constraint applies
    outcome_names : str, list of str
                    name(s) of the outcome(s) to which the constraint applies
    function : callable
               The function should return the distance from the feasibility
               threshold, given the model outputs with a variable name. The
               distance should be 0 if the constraint is met.

    """

    def __init__(self, name, parameter_names=None, outcome_names=None, function=None):
        assert callable(function)
        if not parameter_names:
            parameter_names = []
        elif isinstance(parameter_names, str):
            parameter_names = [parameter_names]

        if not outcome_names:
            outcome_names = []
        elif isinstance(outcome_names, str):
            outcome_names = [outcome_names]

        variable_names = parameter_names + outcome_names

        super().__init__(
            name,
            kind=AbstractOutcome.INFO,
            variable_name=variable_names,
            function=function,
        )

        self.parameter_names = parameter_names
        self.outcome_names = outcome_names

    def process(self, values):
        value = super().process(values)
        assert value >= 0
        return value


def create_outcomes(outcomes, **kwargs):
    """Helper function for creating multiple outcomes

    Parameters
    ----------
    outcomes : DataFrame, or something convertable to a DataFrame
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
            raise ValueError(f"no {entry} column in dataframe")

    temp_outcomes = []
    for _, row in outcomes.iterrows():
        name = row["name"]
        kind = row["type"]

        if kind == "scalar":
            outcome = ScalarOutcome(name)
        elif kind == "timeseries":
            outcome = TimeSeriesOutcome(name)
        else:
            raise ValueError("unknown type for " + name)
        temp_outcomes.append(outcome)
    return temp_outcomes
