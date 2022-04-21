"""parameters and related helper classes and functions"""
import abc
import numbers

import pandas as pd
import scipy as sp

from .util import NamedObject, Variable, NamedObjectMap
from ..util import get_module_logger

# Created on Jul 14, 2016
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = [
    "Constant",
    "RealParameter",
    "IntegerParameter",
    "CategoricalParameter",
    "BooleanParameter",
    "Category",
    "parameters_from_csv",
    "parameters_to_csv",
    "Parameter",
]
_logger = get_module_logger(__name__)


class Bound(metaclass=abc.ABCMeta):
    def __get__(self, instance, cls):
        try:
            bound = instance.__dict__[self.internal_name]
        except KeyError:
            bound = self.get_bound(instance)
            self.__set__(instance, bound)
        return bound

    def __set__(self, instance, value):
        instance.__dict__[self.internal_name] = value

    def __set_name__(self, cls, name):
        self.name = name
        self.internal_name = "_" + name


class UpperBound(Bound):
    def get_bound(self, instance):
        bound = instance.dist.ppf(1.0)
        return bound


class LowerBound(Bound):
    def get_bound(self, owner):
        ppf_zero = 0

        if isinstance(owner.dist.dist, sp.stats.rv_discrete):  # @UndefinedVariable
            # ppf at actual zero for rv_discrete gives lower bound - 1
            # due to a quirk in the scipy.stats implementation
            # so we use the smallest positive float instead
            ppf_zero = 5e-324

        bound = owner.dist.ppf(ppf_zero)
        return bound


class Constant(NamedObject):
    """Constant class,

    can be used for any parameter that has to be set to a fixed value

    """

    def __init__(self, name, value):
        super().__init__(name)
        self.value = value

    def __repr__(self, *args, **kwargs):
        return f"{self.__class__.__name__}('{self.name}', {self.value})"


class Category(Constant):
    def __init__(self, name, value):
        super().__init__(name, value)


def create_category(cat):
    if isinstance(cat, Category):
        return cat
    else:
        return Category(str(cat), cat)


class Parameter(Variable, metaclass=abc.ABCMeta):
    """Base class for any model input parameter

    Parameters
    ----------
    name : str
    lower_bound : int or float
    upper_bound : int or float
    resolution : collection
    pff : bool
          if true, sample over this parameter using resolution in case of
          partial factorial sampling

    Raises
    ------
    ValueError
        if lower bound is larger than upper bound
    ValueError
        if entries in resolution are outside range of lower_bound and
        upper_bound

    """

    lower_bound = LowerBound()
    upper_bound = UpperBound()
    default = None

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, value):
        if value:
            if (min(value) < self.lower_bound) or (max(value) > self.upper_bound):
                raise ValueError(
                    "resolution not consistent with lower and " "upper bound"
                )
        self._resolution = value

    def __init__(
        self,
        name,
        lower_bound,
        upper_bound,
        resolution=None,
        default=None,
        variable_name=None,
        pff=False,
    ):
        super().__init__(name)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.resolution = resolution
        self.default = default
        self.variable_name = variable_name
        self.pff = pff

    @classmethod
    def from_dist(cls, name, dist, **kwargs):
        """alternative constructor for creating a parameter from a frozen
        scipy.stats distribution directly

        Parameters
        ----------
        dist : scipy stats frozen dist
        **kwargs : valid keyword arguments for Parameter instance

        """
        assert isinstance(
            dist, sp.stats._distn_infrastructure.rv_frozen
        )  # @UndefinedVariable
        self = cls.__new__(cls)
        self.dist = dist
        self.name = name
        self.resolution = None
        self.variable_name = None
        self.ppf = None

        for k, v in kwargs.items():
            if k in {"default", "resolution", "variable_name", "pff"}:
                setattr(self, k, v)
            else:
                raise ValueError(f"unknown property {k} for Parameter")

        return self

    def __eq__(self, other):
        if not isinstance(self, other.__class__):
            return False

        self_keys = set(self.__dict__.keys())
        other_keys = set(other.__dict__.keys())
        if self_keys - other_keys:
            return False
        else:
            for key in self_keys:
                if key != "dist":
                    if getattr(self, key) != getattr(other, key):
                        return False
                else:
                    # name, parameters
                    self_dist = getattr(self, key)
                    other_dist = getattr(other, key)
                    if self_dist.dist.name != other_dist.dist.name:
                        return False
                    if self_dist.args != other_dist.args:
                        return False

            else:
                return True

    def __str__(self):
        return self.name


class RealParameter(Parameter):
    """real valued model input parameter

    Parameters
    ----------
    name : str
    lower_bound : int or float
    upper_bound : int or float
    resolution : iterable
    variable_name : str, or list of str

    Raises
    ------
    ValueError
        if lower bound is larger than upper bound
    ValueError
        if entries in resolution are outside range of lower_bound and
        upper_bound

    """

    def __init__(
        self,
        name,
        lower_bound,
        upper_bound,
        resolution=None,
        default=None,
        variable_name=None,
        pff=False,
    ):
        super().__init__(
            name,
            lower_bound,
            upper_bound,
            resolution=resolution,
            default=default,
            variable_name=variable_name,
            pff=pff,
        )

        self.dist = sp.stats.uniform(
            lower_bound, upper_bound - lower_bound
        )  # @UndefinedVariable

    @classmethod
    def from_dist(cls, name, dist, **kwargs):
        if not isinstance(dist.dist, sp.stats.rv_continuous):  # @UndefinedVariable
            raise ValueError("dist should be instance of rv_continouos")
        return super().from_dist(name, dist, **kwargs)


class IntegerParameter(Parameter):
    """integer valued model input parameter

    Parameters
    ----------
    name : str
    lower_bound : int
    upper_bound : int
    resolution : iterable
    variable_name : str, or list of str

    Raises
    ------
    ValueError
        if lower bound is larger than upper bound
    ValueError
        if entries in resolution are outside range of lower_bound and
        upper_bound, or not an integer instance
    ValueError
        if lower_bound or upper_bound is not an integer instance

    """

    def __init__(
        self,
        name,
        lower_bound,
        upper_bound,
        resolution=None,
        default=None,
        variable_name=None,
        pff=False,
    ):
        super().__init__(
            name,
            lower_bound,
            upper_bound,
            resolution=resolution,
            default=default,
            variable_name=variable_name,
            pff=pff,
        )

        lb_int = float(lower_bound).is_integer()
        up_int = float(upper_bound).is_integer()

        if not (lb_int and up_int):
            raise ValueError("lower bound and upper bound must be integers")

        self.lower_bound = int(lower_bound)
        self.upper_bound = int(upper_bound)

        self.dist = sp.stats.randint(
            self.lower_bound, self.upper_bound + 1
        )  # @UndefinedVariable

        try:
            for idx, entry in enumerate(self.resolution):
                if not float(entry).is_integer():
                    raise ValueError("all entries in resolution should be " "integers")
                else:
                    self.resolution[idx] = int(entry)
        except TypeError:
            # if self.resolution is None
            pass

    @classmethod
    def from_dist(cls, name, dist, **kwargs):
        if not isinstance(dist.dist, sp.stats.rv_discrete):  # @UndefinedVariable
            raise ValueError("dist should be instance of rv_discrete")
        return super().from_dist(name, dist, **kwargs)


class CategoricalParameter(IntegerParameter):
    """categorical model input parameter

    Parameters
    ----------
    name : str
    categories : collection of obj
    variable_name : str, or list of str
    multivalue : boolean
                 if categories have a set of values, for each variable_name
                 a different one.
    # TODO: should multivalue not be a seperate class?
    # TODO: multivalue as label is also horrible

    """

    @property
    def categories(self):
        return self._categories

    @categories.setter
    def categories(self, values):
        self._categories.extend(values)

    def __init__(
        self,
        name,
        categories,
        default=None,
        variable_name=None,
        pff=False,
        multivalue=False,
    ):
        lower_bound = 0
        upper_bound = len(categories) - 1

        if upper_bound == 0:
            raise ValueError("there should be more than 1 category")

        super().__init__(
            name,
            lower_bound,
            upper_bound,
            resolution=None,
            default=default,
            variable_name=variable_name,
            pff=pff,
        )
        cats = [create_category(cat) for cat in categories]

        self._categories = NamedObjectMap(Category)

        self.categories = cats
        self.resolution = [i for i in range(len(self.categories))]
        self.multivalue = multivalue

    def index_for_cat(self, category):
        """return index of category

        Parameters
        ----------
        category : object

        Returns
        -------
        int


        """
        for i, cat in enumerate(self.categories):
            if cat.name == category:
                return i
        raise ValueError("category not found")

    def cat_for_index(self, index):
        """return category given index

        Parameters
        ----------
        index  : int

        Returns
        -------
        object

        """

        return self.categories[index]

    def __repr__(self, *args, **kwargs):
        template1 = "CategoricalParameter('{}', {}, default={})"
        template2 = "CategoricalParameter('{}', {})"

        if self.default:
            representation = template1.format(self.name, self.resolution, self.default)
        else:
            representation = template2.format(self.name, self.resolution)

        return representation

    def from_dist(self, name, dist):
        # TODO:: how to handle this
        # probably need to pass categories as list and zip
        # categories to integers implied by dist
        raise NotImplementedError(
            "custom distributions over categories " "not supported yet"
        )


class BooleanParameter(CategoricalParameter):
    """boolean model input parameter

    A BooleanParameter is similar to a CategoricalParameter, except
    the category values can only be True or False.

    Parameters
    ----------
    name : str
    variable_name : str, or list of str

    """

    def __init__(self, name, default=None, variable_name=None, pff=False):
        super().__init__(
            name,
            categories=[True, False],
            default=default,
            variable_name=variable_name,
            pff=pff,
        )


def parameters_to_csv(parameters, file_name):
    """Helper function for writing a collection of parameters to a csv file

    Parameters
    ----------
    parameters : collection of Parameter instances
    file_name :  str


    The function iterates over the collection and turns these into a data
    frame prior to storing them. The resulting csv can be loaded using the
    parameters_from_csv function. Note that currently we don't store resolution
    and default attributes.

    """

    params = {}

    for i, param in enumerate(parameters):

        if isinstance(param, CategoricalParameter):
            values = param.resolution
        else:
            values = param.lower_bound, param.upper_bound

        dict_repr = {j: value for j, value in enumerate(values)}
        dict_repr["name"] = param.name

        params[i] = dict_repr

    params = pd.DataFrame.from_dict(params, orient="index")

    # for readability it is nice if name is the first column, so let's
    # ensure this
    cols = params.columns.tolist()
    cols.insert(0, cols.pop(cols.index("name")))
    params = params.reindex(columns=cols)

    # we can now safely write the dataframe to a csv
    pd.DataFrame.to_csv(params, file_name, index=False)


def parameters_from_csv(uncertainties, **kwargs):
    """Helper function for creating many Parameters based on a DataFrame
    or csv file

    Parameters
    ----------
    uncertainties : str, DataFrame
    **kwargs : dict, arguments to pass to pandas.read_csv

    Returns
    -------
    list of Parameter instances


    This helper function creates uncertainties. It assumes that the
    DataFrame or csv file has a column titled 'name', optionally a type column
    {int, real, cat}, can be included as well. the remainder of the columns
    are handled as values for the parameters. If type is not specified,
    the function will try to infer type from the values.

    Note that this function does not support the resolution and default kwargs
    on parameters.

    An example of a csv:

    NAME,TYPE,,,
    a_real,real,0,1.1,
    an_int,int,1,9,
    a_categorical,cat,a,b,c

    this CSV file would result in

    [RealParameter('a_real', 0, 1.1, resolution=[], default=None),
     IntegerParameter('an_int', 1, 9, resolution=[], default=None),
     CategoricalParameter('a_categorical', ['a', 'b', 'c'], default=None)]

    """

    if isinstance(uncertainties, str):
        uncertainties = pd.read_csv(uncertainties, **kwargs)
    elif not isinstance(uncertainties, pd.DataFrame):
        uncertainties = pd.DataFrame.from_dict(uncertainties)
    else:
        uncertainties = uncertainties.copy()

    parameter_map = {
        "int": IntegerParameter,
        "real": RealParameter,
        "cat": CategoricalParameter,
        "bool": BooleanParameter,
    }

    # check if names column is there
    if ("NAME" not in uncertainties) and ("name" not in uncertainties):
        raise IndexError("name column missing")
    elif "NAME" in uncertainties.columns:
        names = uncertainties["NAME"]
        uncertainties.drop(["NAME"], axis=1, inplace=True)
    else:
        names = uncertainties["name"]
        uncertainties.drop(["name"], axis=1, inplace=True)

    # check if type column is there
    infer_type = False
    if ("TYPE" not in uncertainties) and ("type" not in uncertainties):
        infer_type = True
    elif "TYPE" in uncertainties:
        types = uncertainties["TYPE"]
        uncertainties.drop(["TYPE"], axis=1, inplace=True)
    else:
        types = uncertainties["type"]
        uncertainties.drop(["type"], axis=1, inplace=True)

    uncs = []
    for i, row in uncertainties.iterrows():
        name = names[i]
        values = row.values[row.notnull().values]
        type = None  # @ReservedAssignment

        if infer_type:
            if len(values) != 2:
                type = "cat"  # @ReservedAssignment
            else:
                l, u = values

                if isinstance(l, numbers.Integral) and isinstance(u, numbers.Integral):
                    type = "int"  # @ReservedAssignment
                else:
                    type = "real"  # @ReservedAssignment

        else:
            type = types[i]  # @ReservedAssignment

            if (type != "cat") and (len(values) != 2):
                raise ValueError(
                    "too many values specified for {}, is {}, should be 2".format(
                        name, values.shape[0]
                    )
                )

        if type == "cat":
            uncs.append(parameter_map[type](name, values))
        else:
            uncs.append(parameter_map[type](name, *values))
    return uncs
