'''
Created on 15 mrt. 2013

@author: localadmin
'''
import jpype
import os

import numpy as np

from expWorkbench import ModelStructureInterface, warning, debug,\
                         EMAError
import pyNetLogo

class NetLogoModelStructureInterface(ModelStructureInterface):
    model_file = None
    run_length = None
    command_format = "set {0} {1}"

    def __init__(self, working_directory, name):
        """
        interface to the model
        
        :param working_directory: working_directory for the model. 
        :param name: name of the modelInterface. The name should contain only
                     alphanumerical characters. 
        """
        self.name=None
        self.working_directory=None
        
        super(ModelStructureInterface, self).__init__()
        if working_directory:
            self.set_working_directory(working_directory)
    
        if not name.isalnum():
            raise EMAError("name of model should only contain alpha numerical\
                            characters")
        
        self.name = name
    
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
        self.policy = policy
        
        self.netlogo = pyNetLogo.NetLogoLink()
        debug("netlogo started")
        path = self.working_directory+self.model_file
        self.netlogo.load_model(path)
        debug("model opened")
        
    
    def run_model(self, case):
        """
        Method for running an instantiated model structure. 
        
        This method should always be implemented.
        
        :param case: keyword arguments for running the model. The case is a 
                     dict with the names of the uncertainties as key, and
                     the values to which to set these uncertainties. 
        
        .. note:: This method should always be implemented.
        
        """
        for key, value in case.iteritems():
            try:
                self.netlogo.command(self.command_format.format(key, value))
            except jpype.JavaException as e:
                warning('variable {0} throws exception: {}'.format((key,
                                                                    str(e))))
            
        debug("model parameters set successfully")
          
        # finish setup and invoke run
        self.netlogo.command("setup")
        
        time_commands = []
        end_commands = []
        fns = {}
        for outcome in self.outcomes:
            name = outcome.name
            fn = r'{0}{3}{1}{2}'.format(self.working_directory,
                           name,
                           ".txt",
                           os.sep)
            fns[name] = fn
            fn = '"{}"'.format(fn)
            fn = fn.replace(os.sep, '/')
            
            if self.netlogo.report('is-agentset? {}'.format(name)):
                # if name is name of an agentset, we
                # assume that we should count the total number of agents
                nc = r'{2} {0} {3} {4} {1}'.format(fn,
                                                   name,
                                                   "file-open",
                                                   'file-write',
                                                   'count')
            else:
                # it is not an agentset, so assume that it is 
                # a reporter / global variable
                
                nc = r'{2} {0} {3} {1}'.format(fn,
                                               name,
                                               "file-open",
                                               'file-write')
            if outcome.time:
                time_commands.append(nc)
            else:
                end_commands.append(nc)
                

        c_start = "repeat {} [".format(self.run_length)
        c_close = "go ]"
        c_middle = " ".join(time_commands)
        c_end = " ".join(end_commands)
        command = " ".join((c_start, c_middle, c_close))
        debug(command)
        self.netlogo.command(command)
        
        # after the last go, we have not done a write for the outcomes
        # so we do that now
        self.netlogo.command(c_middle)
        
        # we also need to save the non time series outcomes
        self.netlogo.command(c_end)
        
        self.netlogo.command("file-close-all")
        self._handle_outcomes(fns)

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
        self.netlogo.kill_workspace()
        jpype.shutdownJVM()

    def _handle_outcomes(self, fns):
      
        for key, value in fns.iteritems():
            with open(value) as fh:
                result = fh.readline()
                result = result.strip()
                result = result.split()
                result = [float(entry) for entry in result]
                self.output[key] = np.asarray(result)
            os.remove(value)         
