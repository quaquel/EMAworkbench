'''parameters and collections of parameters'''
import abc
import itertools
import numbers
import pandas
import six
import warnings

from .util import (NamedObject, Variable, NamedObjectMap, Counter,
                   NamedDict, combine)
from ..util import get_module_logger

# Created on Jul 14, 2016
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = [
    'Parameter',
    'RealParameter',
    'IntegerParameter',
    'BooleanParameter',
    'CategoricalParameter',
    'create_parameters',
    'experiment_generator',
    'Policy',
    'Scenario',
    'Experiment']
_logger = get_module_logger(__name__)


class Constant(NamedObject):
    '''Constant class,

    can be used for any parameter that has to be set to a fixed value

    '''

    def __init__(self, name, value):
        super(Constant, self).__init__(name)
        self.value = value

    def __repr__(self, *args, **kwargs):
        return '{}(\'{}\', {})'.format(self.__class__.__name__,
                                       self.name, self.value)


class Category(Constant):
    def __init__(self, name, value):
        super(Category, self).__init__(name, value)


def create_category(cat):
    if isinstance(cat, Category):
        return cat
    else:
        return Category(str(cat), cat)


class Parameter(Variable):
    ''' Base class for any model input parameter

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

    '''

    __metaclass__ = abc.ABCMeta

    INTEGER = 'integer'
    UNIFORM = 'uniform'

    def __init__(self, name, lower_bound, upper_bound, resolution=None,
                 default=None, variable_name=None, pff=False):
        super(Parameter, self).__init__(name)

        if resolution is None:
            resolution = []

        for entry in resolution:
            if not ((entry >= lower_bound) and (entry <= upper_bound)):
                raise ValueError(('resolution not consistent with lower and '
                                  'upper bound'))

        if lower_bound >= upper_bound:
            raise ValueError('upper bound should be larger than lower bound')

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.resolution = resolution
        self.default = default
        self.variable_name = variable_name
        self.pff = pff

    def __eq__(self, other):
        comparison = [all(hasattr(self, key) == hasattr(other, key) and
                          getattr(self, key) == getattr(other, key) for key
                          in self.__dict__.keys())]
        comparison.append(self.__class__ == other.__class__)
        return all(comparison)

    def __str__(self):
        return self.name

    def __repr__(self, *args, **kwargs):
        start = '{}(\'{}\', {}, {}'.format(self.__class__.__name__,
                                           self.name,
                                           self.lower_bound, self.upper_bound)

        if self.resolution:
            start += ', resolution={}'.format(self.resolution)
        if self.default:
            start += ', default={}'.format(self.default)
        if self.variable_name != [self.name]:
            start += ', variable_name={}'.format(self.variable_name)
        if self.pff:
            start += ', pff={}'.format(self.pff)

        start += ')'

        return start


class RealParameter(Parameter):
    ''' real valued model input parameter

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

    '''

    def __init__(self, name, lower_bound, upper_bound, resolution=None,
                 default=None, variable_name=None, pff=False):
        super(
            RealParameter,
            self).__init__(
            name,
            lower_bound,
            upper_bound,
            resolution=resolution,
            default=default,
            variable_name=variable_name,
            pff=pff)

        self.dist = Parameter.UNIFORM

    @property
    def params(self):
        return (self.lower_bound, self.upper_bound - self.lower_bound)


class IntegerParameter(Parameter):
    ''' integer valued model input parameter

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
        upper_bound, or not an numbers.Integral instance
    ValueError
        if lower_bound or upper_bound is not an numbers.Integral instance

    '''

    def __init__(self, name, lower_bound, upper_bound, resolution=None,
                 default=None, variable_name=None, pff=False):
        super(
            IntegerParameter,
            self).__init__(
            name,
            lower_bound,
            upper_bound,
            resolution=resolution,
            default=default,
            variable_name=variable_name,
            pff=pff)

        lb_int = isinstance(lower_bound, numbers.Integral)
        up_int = isinstance(upper_bound, numbers.Integral)

        if not (lb_int or up_int):
            raise ValueError('lower bound and upper bound must be integers')

        for entry in self.resolution:
            if not isinstance(entry, numbers.Integral):
                raise ValueError(('all entries in resolution should be '
                                  'integers'))

        self.dist = Parameter.INTEGER

    @property
    def params(self):
        # scipy.stats.randit uses closed upper bound, hence the +1
        return (self.lower_bound, self.upper_bound + 1)


class CategoricalParameter(IntegerParameter):
    ''' categorical model input parameter

    Parameters
    ----------
    name : str
    categories : collection of obj
    variable_name : str, or list of str
    multivalue : boolean
                 if categories have a set of values, for each variable_name
                 a different one.

    '''

    @property
    def categories(self):
        return self._categories

    @categories.setter
    def categories(self, values):
        self._categories.extend(values)

    def __init__(self, name, categories, default=None, variable_name=None,
                 pff=False, multivalue=False):
        lower_bound = 0
        upper_bound = len(categories) - 1

        if upper_bound == 0:
            raise ValueError('there should be more than 1 category')

        super(
            CategoricalParameter,
            self).__init__(
            name,
            lower_bound,
            upper_bound,
            resolution=None,
            default=default,
            variable_name=variable_name,
            pff=pff)
        cats = [create_category(cat) for cat in categories]

        self._categories = NamedObjectMap(Category)

        self.categories = cats
        self.resolution = [i for i in range(len(self.categories))]
        self.multivalue = multivalue

    def index_for_cat(self, category):
        '''return index of category

        Parameters
        ----------
        category : object

        Returns
        -------
        int


        '''
        for i, cat in enumerate(self.categories):
            if cat.name == category:
                return i
        raise ValueError("category not found")

    def cat_for_index(self, index):
        '''return category given index

        Parameters
        ----------
        index  : int

        Returns
        -------
        object

        '''

        return self.categories[index]

    def invert(self, name):
        ''' invert a category to an integer

        Parameters
        ----------
        name : obj
               category

        Raises
        ------
        ValueError
            if category is not found

        '''
        warnings.warn('deprecated, use index_for_cat instead')
        return self.index_for_cat(name)

    def __repr__(self, *args, **kwargs):
        template1 = 'CategoricalParameter(\'{}\', {}, default={})'
        template2 = 'CategoricalParameter(\'{}\', {})'

        if self.default:
            representation = template1.format(self.name, self.resolution,
                                              self.default)
        else:
            representation = template2.format(self.name, self.resolution)

        return representation


class BinaryParameter(CategoricalParameter):
    ''' a categorical model input parameter that is only True or False

    Parameters
    ----------
    name : str
    '''

    def __init__(self, name, default=None, ):
        super(
            BinaryParameter,
            self).__init__(
            name,
            categories=[
                False,
                True],
            default=default)


class BooleanParameter(IntegerParameter):
    ''' boolean model input parameter

    A BooleanParameter is similar to a CategoricalParameter, except
    the category values can only be True or False.

    Parameters
    ----------
    name : str
    variable_name : str, or list of str

    '''

    def __init__(self, name, default=None, variable_name=None,
                 pff=False):
        super(BooleanParameter, self).__init__(
            name, 0, 1, resolution=None, default=default,
            variable_name=variable_name, pff=pff)

        self.categories = [False, True]
        self.resolution = [0, 1]

    def __repr__(self, *args, **kwargs):
        template1 = 'BooleanParameter(\'{}\', default={})'
        template2 = 'BooleanParameter(\'{}\', )'

        if self.default:
            representation = template1.format(self.name,
                                              self.default)
        else:
            representation = template2.format(self.name, )

        return representation


class Policy(NamedDict):
    '''Helper class representing a policy
    
    Attributes
    ----------
    name : str, int, or float
    id : int
    
    all keyword arguments are wrapped into a dict.
    
    '''
    # TODO:: separate id and name
    # if name is not provided fall back on id
    # id will always be a number and can be generated by
    # a counter
    # the new experiment class can than take the names from
    # policy and scenario to create a unique name while also
    # multiplying the ID's (assuming we count from 1 onward) to get
    # a unique experiment ID
    id_counter = Counter(1)

    def __init__(self, name=Counter(), **kwargs):

        # TODO: perhaps move this to seperate function that internally uses
        # counter
        if isinstance(name, int):
            name = f"policy {name}"

        super(Policy, self).__init__(name, **kwargs)
        self.id = Policy.id_counter()

    def to_list(self, parameters):
        '''get list like representation of policy where the
        parameters are in the order of levers'''

        return [self[param.name] for param in parameters]

    def __repr__(self):
        return "Policy({})".format(super(Policy, self).__repr__())


class Scenario(NamedDict):
    '''Helper class representing a scenario
    
    Attributes
    ----------
    name : str, int, or float
    id : int
    
    all keyword arguments are wrapped into a dict.
    
    '''
    
    # we need to start from 1 so scenario id is known
    id_counter = Counter(1)

    def __init__(self, name=Counter(), **kwargs):
        super(Scenario, self).__init__(name, **kwargs)
        self.id = Scenario.id_counter()

    def __repr__(self):
        return "Scenario({})".format(super(Scenario, self).__repr__())


class Case(NamedObject):
    '''A convenience object that contains a specification
    of the model, policy, and scenario to run

    TODO:: we need a better name for this. probably this should be
    named Experiment, while Experiment should be
    ExperimentReplication

    '''

    def __init__(self, name, model_name, policy, scenario, experiment_id):
        super(Case, self).__init__(name)
        self.experiment_id = experiment_id
        self.policy = policy
        self.model_name = model_name
        self.scenario = scenario


class Experiment(NamedDict):
    '''helper class that combines scenario, policy, any constants, and
    replication information (seed etc) into a single dictionary.

    '''

    def __init__(self, scenario, policy, constants, replication=None):
        scenario_id = scenario.id
        policy_id = policy.id

        if replication is None:
            replication_id = 1
        else:
            replication_id = replication.id
            constants = combine(constants, replication)

        # this is a unique identifier for an experiment
        # we might also create a better looking name
        self.id = scenario_id * policy_id * replication_id
        name = '{}_{}_{}'.format(scenario.name, policy.name, replication_id)

        super(Experiment, self).__init__(
            name, **combine(scenario, policy, constants))


def experiment_generator(scenarios, model_structures, policies):
    '''

    generator function which yields experiments

    Parameters
    ----------
    designs : iterable of dicts
    model_structures : list
    policies : list

    Notes
    -----
    this generator is essentially three nested loops: for each model structure,
    for each policy, for each scenario, return the experiment. This means
    that designs should not be a generator because this will be exhausted after
    the running the first policy on the first model.

    '''
    jobs = itertools.product(model_structures, policies, scenarios)

    for i, job in enumerate(jobs):
        msi, policy, scenario = job
        name = '{} {} {}'.format(msi.name, policy.name, i)
        case = Case(name, msi.name, policy, scenario, i)
        yield case


def parameters_to_csv(parameters, file_name):
    '''Helper function for writing a collection of parameters to a csv file

    Parameters
    ----------
    parameters : collection of Parameter instances
    file_name :  str


    The function iterates over the collection and turns these into a data
    frame prior to storing them. The resulting csv can be loaded using the
    create_parameters function. Note that currently we don't store resolution
    and default attributes.

    '''

    params = {}

    for i, param in enumerate(parameters):

        if isinstance(param, CategoricalParameter):
            values = param.resolution
        else:
            values = param.lower_bound, param.upper_bound

        dict_repr = {j: value for j, value in enumerate(values)}
        dict_repr['name'] = param.name

        params[i] = dict_repr

    params = pandas.DataFrame.from_dict(params, orient='index')

    # for readability it is nice if name is the first column, so let's
    # ensure this
    cols = params.columns.tolist()
    cols.insert(0, cols.pop(cols.index('name')))
    params = params.reindex(columns=cols)

    # we can now safely write the dataframe to a csv
    pandas.DataFrame.to_csv(params, file_name, index=False)


def create_parameters(uncertainties, **kwargs):
    '''Helper function for creating many Parameters based on a DataFrame
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

    '''

    if isinstance(uncertainties, six.string_types):
        uncertainties = pandas.read_csv(uncertainties, **kwargs)
    elif not isinstance(uncertainties, pandas.DataFrame):
        uncertainties = pandas.DataFrame.from_dict(uncertainties)
    else:
        uncertainties = uncertainties.copy()

    parameter_map = {'int': IntegerParameter,
                     'real': RealParameter,
                     'cat': CategoricalParameter,
                     'bool': BooleanParameter,
                     }

    # check if names column is there
    if ('NAME' not in uncertainties) and ('name' not in uncertainties):
        raise IndexError('name column missing')
    elif ('NAME' in uncertainties.columns):
        names = uncertainties['NAME']
        uncertainties.drop(['NAME'], axis=1, inplace=True)
    else:
        names = uncertainties['name']
        uncertainties.drop(['name'], axis=1, inplace=True)

    # check if type column is there
    infer_type = False
    if ('TYPE' not in uncertainties) and ('type' not in uncertainties):
        infer_type = True
    elif ('TYPE' in uncertainties):
        types = uncertainties['TYPE']
        uncertainties.drop(['TYPE'], axis=1, inplace=True)
    else:
        types = uncertainties['type']
        uncertainties.drop(['type'], axis=1, inplace=True)

    uncs = []
    for i, row in uncertainties.iterrows():
        name = names[i]
        values = row.values[row.notnull().values]
        type = None  # @ReservedAssignment

        if infer_type:
            if len(values) != 2:
                type = 'cat'  # @ReservedAssignment
            else:
                l, u = values

                if isinstance(
                        l, numbers.Integral) and isinstance(
                        u, numbers.Integral):
                    type = 'int'  # @ReservedAssignment
                else:
                    type = 'real'  # @ReservedAssignment

        else:
            type = types[i]  # @ReservedAssignment

            if (type != 'cat') and (len(values) != 2):
                raise ValueError(
                    'too many values specified for {}, is {}, should be 2'.format(
                        name, values.shape[0]))

        if type == 'cat':
            uncs.append(parameter_map[type](name, values))
        else:
            uncs.append(parameter_map[type](name, *values))
    return uncs
