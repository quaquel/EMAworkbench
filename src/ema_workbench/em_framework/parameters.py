'''


'''
from __future__ import (unicode_literals, print_function, absolute_import,
                        division)

import abc
import itertools
import numbers
import pandas
import six
import warnings

from .util import NamedObject
from ema_workbench.em_framework.util import NamedDict
# from ema_workbench.em_framework.model import AbstractModel

# Created on Jul 14, 2016
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['Parameter','RealParameter', 'IntegerParameter',
           'CategoricalParameter', 'create_uncertainties']


class Constant(NamedObject):
    '''Constant class, 
    
    can be used for any parameters that have to be set to a fixed value
    
    '''
    
    def __init__(self, name, value):
        super(Constant, self).__init__(name)
        self.value = value


class Parameter(NamedObject):
    ''' Base class for any model input parameter
    
    Parameters
    ----------
    name : str
    lower_bound : int or float
    upper_bound : int or float
    resolution : collection
    
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


    @property
    def variable_name(self):
        if self._variable_name != None:
            return self._variable_name
        else:
            return [self.name]
        
    @variable_name.setter
    def variable_name(self, name):
        if isinstance(name, basestring):
            name = [name]
        self._variable_name = name

    def __init__(self, name, lower_bound, upper_bound, resolution=None,
                 default=None, variable_name=None):
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
        
    def __eq__ (self, other):
        comparison = [all(hasattr(self, key) == hasattr(other, key) and
                          getattr(self, key) == getattr(other, key) for key 
                          in self.__dict__.keys())]
        comparison.append(self.__class__ == other.__class__)
        return all(comparison)

    def __str__(self):
        return self.name
            
            
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
                 default=None, variable_name=None):
        super(RealParameter, self).__init__(name, lower_bound, upper_bound,
                                    resolution=resolution, default=default,
                                    variable_name=variable_name)
        
        self.dist = Parameter.UNIFORM
        
    @property
    def params(self):
        return (self.lower_bound, self.upper_bound-self.lower_bound)
    
    def __repr__(self, *args, **kwargs):
        start = 'RealParameter(\'{}\', {}, {}'.format(self.name, 
                                          self.lower_bound, self.upper_bound)
        
        if self.resolution:
            start += ', resolution={}'.format(self.resolution)
        if self.default:
            start += ', default={}'.format(self.default)
            
        start += ')'
        
        return start
        
        
class IntegerParameter(Parameter):
    ''' integer valued model input parameter
    
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
        upper_bound, or not an numbers.Integral instance
    ValueError 
        if lower_bound or upper_bound is not an numbers.Integral instance
    
    '''
    
    def __init__(self, name, lower_bound, upper_bound, resolution=None, 
                 default=None, variable_name=None):
        super(IntegerParameter, self).__init__(name, lower_bound, upper_bound, 
                                    resolution=resolution, default=default,
                                    variable_name=variable_name)
        
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
        return (self.lower_bound, self.upper_bound)
    
    def __repr__(self, *args, **kwargs):
        start = 'IntegerParameter(\'{}\', {}, {}'.format(self.name, 
                                          self.lower_bound, self.upper_bound)
        
        if self.resolution:
            start += ', resolution={}'.format(self.resolution)
        if self.default:
            start += ', default={}'.format(self.default)
            
        start += ')'
        
        return start


class CategoricalParameter(IntegerParameter):
    ''' categorical model input parameter
    
    Parameters
    ----------
    name : str
    categories : collection of obj
    variable_name : str, or list of str
    
    '''
    
    def __init__(self, name, categories, default=None, variable_name=None):
        lower_bound = 0
        upper_bound = len(categories)

        super(CategoricalParameter, self).__init__(name, lower_bound, 
                            upper_bound, resolution=None, default=default,
                            variable_name=variable_name)
        self.resolution = list(categories)
        
    def index_for_cat(self, category):
        '''return index of category
        
        Parameters
        ----------
        category : object
        
        Returns
        -------
        int
        
        
        '''
        
        return self.resolution.index(category)
    
    def cat_for_index(self, index):
        '''return category associated with index
        
        Parameters
        ----------
        index  : int
        
        Returns
        -------
        object
        
        '''
        
        return self.resolution[index]
        
        
    def transform(self, value):
        '''transform an integer to a category 
        
        Parameters
        ----------
        name : int
               value for which you want the category
               
        Raises
        ------
        IndexError
            if value is out of bounds
        '''
        warnings.warn('deprecated, use cat_for_index instead')

        return self.cat_for_index(value)
    
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


class Policy(NamedDict):
    # idea: make use of NamedObjectMap --> underneath this is an
    # ordered dict, so can always cast values to list / array
    # makes decisions ala the lake model easier to implement
    # combine this with some kind of function that generates many of the
    # same lever objects but with an _id appended to name
    # and lake problem becomes easy
    #
    # problem: won't work as long as you pass any kwargs using **kwarg
    # notation. This removes the ordering
    # practical issue: switching NamedDict from UserDict to 
    # OrderedDict creates a world of pain
    
    def to_list(self, parameters):
        '''get list like representation of policy where the
        parameters are in the order of levers'''
        
        return [self[param.name] for param in parameters]
        

class Scenario(NamedDict):
    job_counter = itertools.count()
    
    def __init__(self, name=None, **kwargs):
        if name == None:
            name = six.next(Scenario.job_counter)
        super(Scenario, self).__init__(name, **kwargs)

class Experiment(NamedObject):
    def __init__(self, name, model, policy, scenario, experiment_id):
        super(Experiment, self).__init__(name)
        self.experiment_id = experiment_id
        self.policy = policy
        self.model = model
        self.scenario = scenario


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
        
        dict_repr = {j:value for j, value in enumerate(values)}
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
    '''Helper function for creating many Parameters based on a dataframe
    or csv file
    
    Parameters
    ----------
    uncertainties : str, dataframe
    **kwargs : dict, arguments to pass to pandas.read_csv
    
    Returns
    -------
    list of Parameter instances
    
    
    This helper function creates uncertainties. It assumes that the 
    dataframe or csv file has a column titled 'name', optionally a type column
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
                     'cat': CategoricalParameter}
    
#     def cast(obj):
#         obj = str(obj)
#         
#         try:
#             obj = int(obj)
#         except ValueError:
#             try:
#                 obj = float(obj)
#             except ValueError:
#                 pass
#         return obj

    # check if names column is there
    if ('NAME' not in uncertainties) and ('name' not in uncertainties):
        raise IndexError('name column missing')
    elif ('NAME' in uncertainties.columns):
        names = uncertainties.ix[:, 'NAME']
        uncertainties.drop(['NAME'], axis=1, inplace=True)
    else:
        names = uncertainties.ix[:, 'name']
        uncertainties.drop(['name'], axis=1, inplace=True)
    
    # check if type column is there
    infer_type = False
    if ('TYPE' not in uncertainties) and ('type' not in uncertainties):
        infer_type = True
    elif ('TYPE' in uncertainties):
        types = uncertainties.ix[:, 'TYPE']
        uncertainties.drop(['TYPE'], axis=1, inplace=True)
    else:
        types = uncertainties.ix[:, 'type']
        uncertainties.drop(['type'], axis=1, inplace=True)
    
    uncs = []
    for i, row in uncertainties.iterrows():
        name = names[i]
        values = row.values[row.notnull().values]
#         values = [cast(entry) for entry in values]
        type = None
        
        if infer_type:
            if len(values) != 2:
                type = 'cat'
            else:
                l, u = values
                
                if isinstance(l, numbers.Integral) and isinstance(u, numbers.Integral):
                    type = 'int'
                else:
                    type = 'real'
            
        else:
            type = types[i]
            
            if (type != 'cat') and (len(values) != 2):
                raise ValueError('to0 many values specified for {}, is {}, should be 2'.format(name, values.shape[0]))
            
        if type=='cat':
            uncs.append(parameter_map[type](name, values))
        else:
            uncs.append(parameter_map[type](name, *values))
    return uncs