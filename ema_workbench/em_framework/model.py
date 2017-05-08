'''
This module specifies the abstract base class for interfacing with models. 
Any model that is to be controlled from the workbench is controlled via
an instance of an extension of this abstract base class. 

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import abc
import copy
import operator
import os
import six
import warnings
from ema_workbench.util import ema_logging
from ema_workbench.em_framework.parameters import CategoricalParameter


try:
    from collections import MutableMapping
except ImportError:
    from collections.abc import MutableMapping  # @UnusedImport

from collections import defaultdict

from .util import (NamedObject, combine, NamedObjectMapDescriptor)
from .parameters import Parameter, Constant
from .outcomes import AbstractOutcome

from ..util import debug, EMAError
from ..util.ema_logging import method_logger

# Created on 23 dec. 2010
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
# 

__all__ = ['AbstractModel', 'Model', 'FileModel', 'Replicator', 
           'SingleReplication']

#==============================================================================
# abstract Model class 
#==============================================================================
class ModelMeta(abc.ABCMeta):
    
    def __new__(mcls, name, bases, namespace):  # @NoSelf
        
        for key, value in namespace.items():
            if isinstance(value, NamedObjectMapDescriptor):
                value.name = key
                value.internal_name = '_'+key
       
        return abc.ABCMeta.__new__(mcls, name, bases, namespace)


class AbstractModel(six.with_metaclass(ModelMeta, NamedObject)):
    '''
    :class:`ModelStructureInterface` is one of the the two main classes used 
    for performing EMA. This is an abstract base class and cannot be used 
    directly. When extending this class :meth:`model_init` and 
    :meth:`run_model` have to be implemented. 
    
    
    Attributes
    ----------
    uncertainties : listlike
                    list of parameter 
    levers : listlike
             list of parameter instances
    outcomes : listlike
               list of outcome instances
    name : str
           alphanumerical name of model structure interface
    output : dict
             this should be a dict with the names of the outcomes as key
    
    '''
    
    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, outputs):
        for outcome in self.outcomes:
            data = [outputs[var] for var in outcome.variable_name]
            self._output[outcome.name] = outcome.process(data)
    
    @property
    def outcome_variables(self):
        if self._outcome_variables is None:
            self._outcome_variables = [var for o in self.outcomes for var in 
                                       o.variable_name]
        return self._outcome_variables
         
    uncertainties = NamedObjectMapDescriptor(Parameter)
    levers = NamedObjectMapDescriptor(Parameter)
    outcomes = NamedObjectMapDescriptor(AbstractOutcome)
    constants = NamedObjectMapDescriptor(Constant)
        
    def __init__(self, name):
        """
        interface to the model
        
        Parameters
        ----------
        name : str
               name of the modelInterface. The name should contain only
               alpha-numerical characters.        

               
        Raises
        ------
        EMAError if name contains non alpha-numerical characters
        
        """
        super(AbstractModel, self).__init__(name)

        if not self.name.isalnum():
            raise EMAError("name of model should only contain alpha numerical\
                            characters")

        self._outcome_variables = None
        self._output = {}
        
    @method_logger
    def model_init(self, policy):
        '''
        Method called to initialize the model.
        
        Parameters
        ----------
        policy : dict
                 policy to be run.
 
        
        Note
        ----
        This method should always be implemented. Although in simple cases, a 
        simple pass can suffice.
         
        '''
        self.policy = policy

        remove = []
        for key, value in policy.items():
            if hasattr(self, key):
                setattr(self, key, value)
                remove.append(key)
        
        for k in remove:
            del policy[k]
                
        

    def _transform(self, sampled_parameters, parameters):
        
        if not parameters:
            # no parameters defined, so nothing to transform, mainly
            # useful for manual specification of scenario /  policy
            # without having to define uncertainties / levers
            return
        
        temp = {}
        for par in parameters:
            # only keep uncertainties that exist in this model
            try:
                value = sampled_parameters[par.name]
            except KeyError:
                if par.default is not None:
                    value = par.default
                else:
                    ema_logging.debug('{} not found'.format(par.name))
                    continue

            multivalue = False
            if isinstance(par, CategoricalParameter):
                if par.multivalue == True:
                    multivalue = True
                    values = value
            
            # translate uncertainty name to variable name
            for i, varname in enumerate(par.variable_name):
                # a bit hacky implementation, investigate some kind of 
                # zipping of variable_names and values
                if multivalue:
                    value = values[i]
                
                temp[varname] = value
        
        sampled_parameters.data = temp

    @method_logger    
    def run_model(self, scenario, policy):
        """
        Method for running an instantiated model structure. 
        
        Parameters
        ----------
        scenario : Scenario instance
        policy : Policy instance
        
        
        """
        if not self.initialized(policy):
            self.model_init(policy)
        
        #TODO:: here we need to add policies and constants in some manner
        self._transform(scenario, self.uncertainties)
        self._transform(policy, self.levers)

    @method_logger
    def initialized(self, policy):
        '''check if model has been initialized 

        Parameters
        ----------
        policy : a Policy instance
        
        '''
        
        try:
            return self.policy.name == policy.name
        except AttributeError:
            return False

    @method_logger
    def retrieve_output(self):
        """
        Method for retrieving output after a model run.
        
        Returns
        -------
        dict with the results of a model run. 
        """
        warnings.warn('deprecated, use model.output instead')
        return self.output
    
    @method_logger
    def reset_model(self):
        """
        Method for reseting the model to its initial state. The default
        implementation only sets the outputs to an empty dict. 

        """
        self._output = {}
    
    @method_logger
    def cleanup(self):
        '''
        This model is called after finishing all the experiments, but 
        just prior to returning the results. This method gives a hook for
        doing any cleanup, such as closing applications. 
        
        In case of running in parallel, this method is called during 
        the cleanup of the pool, just prior to removing the temporary 
        directories. 
        
        '''
        pass
    
    def as_dict(self):
        '''returns a dict representation of the model'''
        
        def join_attr(field):
            joined = ', '.join([repr(entry) for entry in 
                                sorted(field, key=operator.attrgetter('name'))])
            return '[{}]'.format(joined)
        model_spec = {}
        
        klass = self.__class__.__name__
        name = self.name
                
        uncs = ''
        for uncertainty in self.uncertainties:
            uncs += '\n' + repr(uncertainty)
            
        model_spec['class'] = klass
        model_spec['name'] = name
        model_spec['uncertainties'] = join_attr(self.uncertainties)
        model_spec['outcomes'] = join_attr(self.outcomes)
        model_spec['constants'] = join_attr(self.constants)
            
        return model_spec

class Replicator(AbstractModel):
    @property
    def replications(self):
        return self._replications
    
    @replications.setter
    def replications(self, replications):
        
        # int
        if isinstance(replications, int):
            # TODO:: use a repeating generator instead
        
            self._replications = [{} for _ in range(replications)]
            self.nreplications = replications
        elif isinstance(replications, list):
            # should we check if all are dict?
            self._replications = replications
            self.nreplications = len(replications) 
        else:
            raise TypeError("replications should be int or list not {}".format(type(replications)))

    @method_logger
    def run_model(self, scenario, policy):
        """
        Method for running an instantiated model structure. 
        
        Parameters
        ----------
        scenario : Scenario instance
        policy : Policy instance
        
        
        """
        super(Replicator, self).run_model(scenario, policy)
        
        constants = {c.name:c.value for c in self.constants}
        
        outputs = defaultdict(list)
        
        policy = copy.deepcopy(self.policy)
        partial_experiment = combine(scenario, policy, constants)
        
        for _, rep in enumerate(self.replications):
            experiment = copy.deepcopy(partial_experiment)
            experiment = combine(experiment, rep)
            output = self.run_experiment(experiment)
            for key, value in output.items():
                outputs[key].append(value)
            
        self.output = outputs


class SingleReplication(AbstractModel):
    @method_logger
    def run_model(self, scenario, policy):
        """
        Method for running an instantiated model structure. 
        
        Parameters
        ----------
        scenario : Scenario instance
        policy : Policy instance
        
        
        """
        super(SingleReplication, self).run_model(scenario, policy) 
        # TODO:: should this not be moved up?
        constants = {c.name:c.value for c in self.constants}
        experiment = combine(scenario, self.policy, constants)
        self.output = self.run_experiment(experiment)


class BaseModel(AbstractModel):
    '''
    generic class for working with models implemented as a Python callable 
    
    Parameters
    ----------
    name : str
    function : callable
               a function with each of the uncertain parameters as a keyword
               argument
    
    
    Attributes
    ----------
    uncertainties : listlike
                    list of parameter 
    levers : listlike
             list of parameter instances
    outcomes : listlike
               list of outcome instances
    name : str
           alphanumerical name of model structure interface
    output : dict
             this should be a dict with the names of the outcomes as key
    working_directory : str
                        absolute path, all file operations in the model
                        structure interface should be resolved from this
                        directory. 
    
    '''

    
    def __init__(self, name, function=None):
        super(BaseModel, self).__init__(name)
        
        if not callable(function):
            raise ValueError('function should be callable')
        
        self.function = function

    @method_logger
    def run_experiment(self, experiment):
        """
        Method for running an instantiated model structure. 
        
        Parameters
        ----------
        experiment : dict like
        
        
        """
        model_output = self.function(**experiment)
        
        # TODO: might it be possible to somehow abstract this
        # perhaps expose a get_data on modelInterface?
        # different connectors can than implement only this
        # get method
        results  = {}
        for i, variable in enumerate(self.outcome_variables):
            try:
                value = model_output[variable]
            except KeyError:
                ema_logging.warning(variable +' not found in model output')
                value  = None
            except TypeError:
                value = model_output[i]
            results[variable] = value
        return results
        
    def as_dict(self):
        model_specs = super(BaseModel, self).as_dict()
        model_specs['function'] = self.function
        return model_specs


# TODO:: is it required that FileModel has a model_file?
# might instead be some data files used by a Python Process
# Solution, have an intermediate class, WorkingDirectoryModel
# FileModel than extents that class with the model_file
class FileModel(AbstractModel):
    @property
    def working_directory(self):
        return self._working_directory
    
    @working_directory.setter
    def working_directory(self, path):
        wd = os.path.abspath(path)
        debug('setting working directory to '+ wd)
        self._working_directory = wd

    def __init__(self, name, wd=None, model_file=None):
        """interface to the model
        
        interface to the model
        
        Parameters
        ----------
        name : str
               name of the modelInterface. The name should contain only
               alpha-numerical characters.        
        working_directory : str
                            working_directory for the model. 
        model_file  : str
                     The model file relative to working directory
               
        Raises
        ------
        EMAError 
            if name contains non alpha-numerical characters
        ValueError
            if model_file cannot be found
        
        """
        super(FileModel, self).__init__(name)
        self.working_directory = wd

        path_to_file = os.path.join(self.working_directory, model_file)
        if not os.path.isfile(path_to_file):
            raise ValueError('cannot find model file')
        
        self.model_file = model_file

    def as_dict(self):
        model_specs = super(FileModel, self).as_dict()
        model_specs['model_file'] = self.model_file
        model_specs['working_directory'] = self.working_directory
        return model_specs

class Model(SingleReplication, BaseModel):
    pass