'''
This module specifies the abstract base class for interfacing with models. 
Any model that is to be controlled from the workbench is controlled via
an instance of an extension of this abstract base class. 

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import abc
import os

from .util import NamedObject, NamedObjectMap, combine
from .parameters import Parameter, Constant
from .outcomes import AbstractOutcome
from ..util import debug, EMAError


# Created on 23 dec. 2010
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
# TODO:: move working directory as an argument to FileModel, no
# need to have that in AbstractModel, or Model

__all__ = ['AbstractModel', 'Model']

#==============================================================================
# abstract Model class 
#==============================================================================

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
    
    __metaclass__ = abc.ABCMeta
    
    name = None 
    output = {}
    _working_directory = None

    @property
    def uncertainties(self):
        return self._uncertainties
 
    @uncertainties.setter
    def uncertainties(self, uncs):
        self._uncertainties.extend(uncs)

    @property
    def outcomes(self):
        return self._outcomes
 
    @outcomes.setter
    def outcomes(self, outcomes):
        self._outcomes.extend(outcomes)

    @property
    def levers(self):
        return self._levers
 
    @levers.setter
    def levers(self, levers):
        self._levers.extend(levers)
        
    @property
    def constants(self):
        return self._constants
 
    @constants.setter
    def constants(self, constants):
        self._constants.extend(constants)

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
            
        self._uncertainties = NamedObjectMap(Parameter)
        self._levers = NamedObjectMap(Parameter)
        self._outcomes = NamedObjectMap(AbstractOutcome)
        self._constants = NamedObjectMap(Constant)
        
    def model_init(self, policy, kwargs):
        '''
        Method called to initialize the model.
        
        Parameters
        ----------
        policy : dict
                 policy to be run.
        kwargs : dict
                 keyword arguments to be used by model_intit. This
                 gives users to the ability to pass any additional 
                 arguments. 
        
        Note
        ----
        This method should always be implemented. Although in simple cases, a 
        simple pass can suffice.
        
        Note
        ----
        Anything that is relative to `self.working_directory` should be 
        specified in :meth:`model_init` and not in :meth:`src`. Otherwise, 
        the code will not work when running it in parallel. The reason for this 
        is that the working directory is being updated to reflect the working
        directory of the worker
         
        '''
        self.policy = policy

    
    @abc.abstractmethod
    def run_model(self, case):
        """
        Method for running an instantiated model structure. 
        
        Parameters
        ----------
        case : dict
               keyword arguments for running the model. The case is a dict with 
               the names of the uncertainties as key, values are the values
               to which to set these uncertainties. 
        
        Note
        ----
        This method should always be implemented.
        
        """

    def retrieve_output(self):
        """
        Method for retrieving output after a model run.
        
        Returns
        -------
        dict with the results of a model run. 
        """
        return self.output
    
    def reset_model(self):
        """
        Method for reseting the model to its initial state. The default
        implementation only sets the outputs to an empty dict. 

        """
        self.output = {}
    
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
        self.function = function
    
    def run_model(self, case):
        """
        Method for running an instantiated model structure. 
        
        Parameters
        ----------
        case : dict
               keyword arguments for running the model. The case is a dict with 
               the names of the uncertainties as key, values are the values
               to which to set these uncertainties. 
        
        Note
        ----
        This method should always be implemented.
        
        """
        
        if not callable(self.function):
            raise EMAError('no callable function specified')
            
        constants = {c.name:c.value for c in self.constants}
        experiment = combine(case, self.policy, constants)
        result = self.function(**experiment)
        
        self.output = {outcome.name:result[outcome.name] for outcome in 
                       self.outcomes}

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
        
        if not os.path.isfile(self.working_directory+model_file):
            raise ValueError('cannot find model file')
        
        self.model_file = model_file
        self.working_directory = wd

    def model_init(self, policy, kwargs):
        AbstractModel.model_init(self, policy, kwargs)   

