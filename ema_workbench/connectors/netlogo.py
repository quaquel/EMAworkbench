'''
This module specifies a generic ModelStructureInterface for controlling
NetLogo models. 
'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
from ema_workbench.em_framework.model import Replicator, SingleReplication

try:
    import jpype
except ImportError:
    jpype = None
import os

import numpy as np
import six

from ..em_framework.model import FileModel
from ..util.ema_logging import method_logger
from ..util import warning, debug 

                         
from . import pyNetLogo

# Created on 15 mrt. 2013
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['NetLogoModel']

class BaseNetLogoModel(FileModel):
    '''Base class for interfacing with netlogo models. This class
    extends :class:`em_framework.ModelStructureInterface`.
    
    Attributes
    ----------
    model_file : str
                 a relative path from the working directory to the model
    run_length : int
                 number of ticks
    command_format : str
                     default format for set operations in logo
    working_directory : str
    name : str
    
    '''
    command_format = "set {0} {1}"

    def __init__(self, name, wd=None, model_file=None):
        """
        init of class
        
        Parameters
        ----------
        working_directory : str
                            working_directory for the model. 
        name : str
               name of the modelInterface. The name should contain only
               alpha-numerical characters.
               
        Raises
        ------
        EMAError if name contains non alpha-numerical characters
        
        Note
        ----
        Anything that is relative to `self.working_directory`should be 
        specified in `model_init` and not in `src`. Otherwise, the code 
        will not work when running it in parallel. The reason for this is that 
        the working directory is being updated by parallelEMA to the worker's 
        separate working directory prior to calling `model_init`.
        
        """
        super(BaseNetLogoModel, self).__init__(name, wd=wd, model_file=model_file)
    
        self.run_length = None
       
    @method_logger
    def model_init(self, policy):
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
        
        '''
        super(BaseNetLogoModel, self).model_init(policy)
        debug("trying to start NetLogo")
        self.netlogo = pyNetLogo.NetLogoLink()
        debug("netlogo started")
        path = os.path.join(self.working_directory, self.model_file)
        self.netlogo.load_model(path)
        debug("model opened")
        
    @method_logger
    def run_experiment(self, experiment):
        """
        Method for running an instantiated model structure. 
        
        Parameters
        ----------
        experiment : dict like

        
        Raises
        ------
        jpype.JavaException if there is any exception thrown by the netlogo 
        model
        
        
        """
#         super(BaseNetLogoModel, self).run_model(scenario, policy)
        
        for key, value in experiment.items():
            try:
                self.netlogo.command(self.command_format.format(key, value))
            except jpype.JavaException as e:
                warning('variable {} throws exception: {}'.format(key,
                                                                  str(e)))
            
        debug("model parameters set successfully")
          
        # finish setup and invoke run
        self.netlogo.command("setup")
        
        # TODO:: it is possible to take advantage of of fact
        # that not all outcomes are time series
        # In that case, we need not embed the get command in the go
        # routine, but can do them at the end
        commands = []
        fns = {}
        for variable in self.outcome_variables:
            fn = r'{0}{3}{1}{2}'.format(self.working_directory,
                           variable,
                           ".txt",
                           os.sep)
            fns[variable] = fn
            fn = '"{}"'.format(fn)
            fn = fn.replace(os.sep, '/')
            
            if self.netlogo.report('is-agentset? {}'.format(variable)):
                # if name is name of an agentset, we
                # assume that we should count the total number of agents
                nc = r'{2} {0} {3} {4} {1}'.format(fn,
                                                   variable,
                                                   "file-open",
                                                   'file-write',
                                                   'count')
            else:
                # it is not an agentset, so assume that it is 
                # a reporter / global variable
                
                nc = r'{2} {0} {3} {1}'.format(fn,
                                               variable,
                                               "file-open",
                                               'file-write')
            commands.append(nc)
                

        c_start = "repeat {} [".format(self.run_length)
        c_close = "go ]"
        c_middle = " ".join(commands)
#         c_end = " ".join(end_commands)
        command = " ".join((c_start, c_middle, c_close))
        debug(command)
        self.netlogo.command(command)
        
        # after the last go, we have not done a write for the outcomes
        # so we do that now
        self.netlogo.command(c_middle)
        
        # we also need to save the non time series outcomes
#         self.netlogo.command(c_end)
        
        self.netlogo.command("file-close-all")
        return self._handle_outcomes(fns)

    def retrieve_output(self):
        """
        Method for retrieving output after a model run.
        
        Returns
        -------
        dict with the results of a model run. 
        
        """
        return self.output
    
    def cleanup(self):
        '''
        This model is called after finishing all the experiments, but 
        just prior to returning the results. This method gives a hook for
        doing any cleanup, such as closing applications. 
        
        In case of running in parallel, this method is called during 
        the cleanup of the pool, just prior to removing the temporary 
        directories. 
        
        '''
        self.netlogo.kill_workspace()
        jpype.shutdownJVM()

    def _handle_outcomes(self, fns):
        '''helper function for parsing outcomes'''
        
        results = {}
        for key, value in fns.items():
            with open(value) as fh:
                result = fh.readline()
                result = result.strip()
                result = result.split()
                result = [float(entry) for entry in result]
                results[key] = np.asarray(result)
            os.remove(value)        
        
        temp_output = {}
        for outcome in self.outcomes:
            varname = outcome.variable_name
            if len(varname)==1:
                varname = varname[0]
                temp_output[outcome.name] = results[varname]
            else:
                temp_output[outcome.name] = [results[var] for var in varname]
        return temp_output
         
class NetLogoModel(Replicator, BaseNetLogoModel):
    pass

class SingleReplicationNetLogoModel(SingleReplication, BaseNetLogoModel):
    pass