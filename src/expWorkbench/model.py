'''
Created on 23 dec. 2010

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import division

import abc
import os

from ema_logging import debug
from ema_exceptions import EMAError
SVN_ID = '$Id: model.py 1027 2012-11-21 19:56:50Z jhkwakkel $'
__all__ = ['ModelStructureInterface']

#==============================================================================
# abstract Model class 
#==============================================================================
class ModelStructureInterface(object):
    '''
    :class:`ModelStructureInterface` is one of the the two main classes used 
    for performing EMA. This class should be extended to provide an interface 
    to the actual model. 
    
    '''
    
    __metaclass__ = abc.ABCMeta
    
    #: list of uncertainty instances
    uncertainties = []
    
    #: list of outcome instances
    outcomes = []
    
    #: name of the model interface
    name = None 
    
    #: results, this should be a dict with the names of the outcomes as key
    output = {}

    workingDirectory = None
    
    def __init__(self, workingDirectory, name):
        """
        interface to the model
        
        :param workingDirectory: workingDirectory for the model. 
        :param name: name of the modelInterface. The name should contain only
                     alphanumerical characters. 
        """
        self.name=None
        self.workingDirectory=None
        
        super(ModelStructureInterface, self).__init__()
        if workingDirectory:
            self.set_working_directory(workingDirectory)
    
        if not name.isalnum():
            raise EMAError("name of model should only contain alpha numerical\
                            characters")
        
        self.name = name
    
    @abc.abstractmethod
    def model_init(self, policy, kwargs):
        '''
        Method called to initialize the model.
        
        :param policy: policy to be run.
        :param kwargs: keyword arguments to be used by model_intit. This
                       gives users to the ability to pass any additional 
                       arguments. 
        
        .. note:: This method should always be implemented. Although in simple
                  cases, a simple pass can suffice. 
        '''
    
    @abc.abstractmethod
    def run_model(self, case):
        """
        Method for running an instantiated model structure. 
        
        This method should always be implemented.
        
        :param case: keyword arguments for running the model. The case is a 
                     dict with the names of the uncertainties as key, and
                     the values to which to set these uncertainties. 
        
        .. note:: This method should always be implemented.
        
        """

    def retrieve_output(self):
        """
        Method for retrieving output after a model run.
        
        :return: the results of a model run. 
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
        
        :return: list of the uncertainties of the model interface.
        """
        return self.uncertainties    
    
    def set_working_directory(self, wd):
        '''
        Method for setting the working directory of the model interface. This
        method is used in case of running models in parallel. In this case,
        each worker process will have its own working directory, to avoid 
        having to share files across processes. This requires the need to
        update the working directory to the new working directory. 
        
        :param wd: The new working directory.
        
        '''
        
        wd = os.path.abspath(wd)
        debug('setting working directory to '+ wd)
        
        self.workingDirectory = wd