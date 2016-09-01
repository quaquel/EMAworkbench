'''
This module specifies the abstract base class for interfacing with models. 
Any model that is to be controlled from the workbench is controlled via
an instance of an extension of this abstract base class. 

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import abc
from functools import wraps
import os
import warnings
from ema_workbench.util import ema_logging
from ema_workbench.em_framework.parameters import CategoricalParameter


try:
    from collections import MutableMapping
except ImportError:
    from collections.abc import MutableMapping


from .util import (NamedObject, NamedObjectMap, combine, 
                   NamedObjectMapDescriptor)
from .parameters import Parameter, Constant
from .outcomes import AbstractOutcome

from ..util import debug, EMAError
from ..util.ema_logging import method_logger

# Created on 23 dec. 2010
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
# TODO:: move working directory as an argument to FileModel, no
# need to have that in AbstractModel, or Model

__all__ = ['AbstractModel', 'Model']

#==============================================================================
# abstract Model class 
#==============================================================================
class ModelMeta(abc.ABCMeta):
    
    def __new__(mcls, name, bases, namespace):
        
        for key, value in namespace.items():
            if isinstance(value, NamedObjectMapDescriptor):
                value.name = key
                value.internal_name = '_'+key
       
        return abc.ABCMeta.__new__(mcls, name, bases, namespace)


class AbstractModel(NamedObject):
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
    
    __metaclass__ = ModelMeta

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, outputs):
        for key, value in outputs.items():
            self._output[key] = self.outcomes[key].process(value)
            
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

        # update any attribute on object that is found in policy
        for key, value in policy.items():
            if hasattr(self, key):
                value = policy.pop(key)
                setattr(self, key, value)
                

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
            
        # TODO:: useless name, implementation wise a mess
        # needs to be called always when calling run_model,
        # but then, either put it in super (requires inplace updating
        # of scenario, which is possible), but also reduces what can
        # be done when extending run_model because it makes calling
        # super obligatory at the start
        #
        # updating scenario requires updating the inner data dict
        # so scenario.data = temp_scenario
        #
        # we can do it in the super, but have it available as a separate
        # function as well. Extending / implementing model interfaces
        # is not something most users will have to do
        # 
        # TODO:: this unraveling of scenario should also be supported
        # for policies if we want to ensure that lever based 
        # policies function properly
        temp_scenario = {}
        for unc in self.uncertainties:
            # only keep uncertainties that exist in this model
            try:
                value = scenario[unc.name]
            except KeyError:
                if unc.default is not None:
                    value = unc.default
                            
            # TODO:: translate categories
            multivalue = False
            if isinstance(unc, CategoricalParameter):
                category = unc.categories[value]
                
                value = category.value
                
                if category.multivalue == True:
                    multivalue = True
                    values = value
            
            # translate uncertainty name to variable name
            for i, varname in enumerate(unc.variable_name):
                # a bit hacky implementation, investigate some kind of 
                # zipping of variable_names and values
                if multivalue:
                    value = values[i]
                
                temp_scenario[varname] = value
        
        scenario.data = temp_scenario
        

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
        self.output = {}
    
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
    

class Model(AbstractModel):
    '''
    :class:`ModelStructureInterface` is one of the the two main classes used 
    for performing EMA. This is an abstract base class and cannot be used 
    directly. When extending this class :meth:`model_init` and 
    :meth:`run_model` have to be implemented. 
    
    Parameters
    ----------
    name : str
    wd : str
         string specifying the path of the working directory used by function
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
        super(Model, self).__init__(name)
        
        if not callable(function):
            raise ValueError('function should be callable')
        
        self.function = function

    @method_logger
    def run_model(self, scenario, policy):
        """
        Method for running an instantiated model structure. 
        
        Parameters
        ----------
        scenario : Scenario instance
        policy : Policy instance
        
        
        """
        super(Model, self).run_model(scenario, policy)
        
        constants = {c.name:c.value for c in self.constants}
        
        policy = self.policy.to_list(self.levers)
        
        experiment = combine(scenario, self.policy, constants)
        
        result = self.function(**experiment)
        
        results  = {}
        for outcome in self.outcomes:
            varname = outcome.variable_name
            if isinstance(varname, basestring):
                result[outcome.name] = result[varname]
            else:
                result[outcome.name] = [result[var] for var in varname]
        self.output = results

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
        
        if not os.path.isfile(self.working_directory+model_file):
            raise ValueError('cannot find model file')
        
        self.model_file = model_file

