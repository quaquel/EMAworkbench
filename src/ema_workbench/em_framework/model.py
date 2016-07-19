'''
This module specifies the abstract base class for interfacing with models. 
Any model that is to be controlled from the workbench is controlled via
an instance of an extension of this abstract base class. 

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import abc
import os

from ..util import debug, EMAError
from ema_workbench.em_framework.util import NamedObject

# Created on 23 dec. 2010
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['ModelStructureInterface']

#==============================================================================
# abstract Model class 
#==============================================================================
class ModelStructureInterface(NamedObject):
    '''
    :class:`ModelStructureInterface` is one of the the two main classes used 
    for performing EMA. This is an abstract base class and cannot be used 
    directly. When extending this class :meth:`model_init` and 
    :meth:`run_model` have to be implemented. 
    
    
    Attributes
    ----------
    uncertainties : list
                    list of uncertainty instances
    outcomes : list
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
    
    __metaclass__ = abc.ABCMeta
    
    uncertainties = []
    outcomes = []
    levers = []
    name = None 
    output = {}
    _working_directory = None

#    TODO:: this will break existing model interface classes
#     @property
#     def uncertainties(self):
#         return self._uncertainties
#     
#     @uncertainties.setter
#     def uncertainties(self, uncs):
#         self._uncertainties.extend((uncs))

    def __init__(self, name, working_directory):
        """
        interface to the model
        
        Parameters
        ----------
        name : str
               name of the modelInterface. The name should contain only
               alpha-numerical characters.        
        working_directory : str
                            working_directory for the model. 
               
        Raises
        ------
        EMAError if name contains non alpha-numerical characters
        
        """
        super(ModelStructureInterface, self).__init__(name)

        if working_directory:
            self.set_working_directory(working_directory)
    
        if not self.name.isalnum():
            raise EMAError("name of model should only contain alpha numerical\
                            characters")
        
    @property
    def working_directory(self):
        return self._working_directory
    
    @working_directory.setter
    def working_directory(self, path):
        self.set_working_directory(path)
    
    @abc.abstractmethod
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

    def get_model_uncertainties(self):
        """
        Method for retrieving model structure uncertainties.
        
        Returns
        -------
        list of the uncertainties of the model interface.
        """
        return self.uncertainties    
    
    def set_working_directory(self, wd):
        '''
        Method for setting the working directory of the model interface. This
        method is used in case of running models in parallel. In this case,
        each worker process will have its own working directory, to avoid 
        having to share files across processes. This requires the need to
        update the working directory to the new working directory. 
        
        Parameters
        ----------
        wd : str
             The new working directory.
        
        '''
        
        wd = os.path.abspath(wd)
        debug('setting working directory to '+ wd)
        
        self._working_directory = wd