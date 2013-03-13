
'''
Created on 25 mei 2011

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

convenience functions and classes to be used in combination with Vensim.
This module contains frequently used functions with error checking. For
more fine grained control, the :mod:`vensimDLLwrapper` can also be used 
directly.

'''
from __future__ import division
import types
import numpy as np

from expWorkbench import debug, info, warning, ModelStructureInterface,\
                         Outcome, CaseError, EMAWarning 

from vensimDLLwrapper import command, get_val
from vensimDLLwrapper import VensimError, VensimWarning
import vensimDLLwrapper 

__all__ = ['be_quiet',
           'load_model',
           'read_cin_file',
           'set_value',
           'run_simulation',
           'get_data',
           'VensimModelStructureInterface']


def be_quiet():
    '''
    this allows you to turn off the work in progress dialog that Vensim 
    displays during simulation and other activities, and also prevent the 
    appearance of yes or no dialogs.
    
    defaults to 2, suppressing all windows, for more fine grained control, use
    :mod:`vensimDLLwrapper` directly .
    '''
    vensimDLLwrapper.be_quiet(2)
    

def load_model(file):
    '''
    load the model 
    
    :param file: the location of the .vpm file to be loaded.
    :exception: raises a :class:`~EMAExceptions.VensimError` if the model 
                cannot be loaded.
    
    .. note: only works for .vpm files
    
    '''
    debug("executing COMMAND: SIMULATE>SPECIAL>LOADMODEL|"+file)
    try:
        command(r"SPECIAL>LOADMODEL|"+file)
    except VensimWarning as w:
        warning(str(w))
        raise VensimError("vensim file not found")

def read_cin_file(file):
    '''
    read a .cin file
    
    :param file: location of the .cin file.
    :exception: raises a :class:`~EMAExceptions.VensimWarning` if the cin file
                cannot be read.
    '''
    debug("executing COMMAND: SIMULATE>READCIN|"+file)
    try:
        command(r"SIMULATE>READCIN|"+file)
    except VensimWarning as w:
        debug(str(w))
        raise w

def set_value(variable, value):
    '''
    set the value of a variable to value
    
    current implementation only works for lookups and normal values. In case
    of a list, a lookup is assumed, else a normal value is assumed. 
    See the DSS reference supplement, p. 58 for details.

    
    :param variable: name of the variable to set.
    :param value: the value for the variable. 
                  **note**: the value can be either a list, or an float/integer. 
                  If it is a list, it is assumed the variable is a lookup.
    '''
    
    if type(value) == types.ListType:
        command(r"SIMULATE>SETVAL|"+variable+"("+ str(value)[1:-1] + ")")
    else:
        try:
            command(r"SIMULATE>SETVAL|"+variable+"="+str(value))
        except VensimWarning:
            warning('variable: \'' +variable+'\' not found')


def run_simulation(file):
    ''' 
    Convenient function to run a model and store the results of the run in 
    the specified .vdf file. The specified output file will be overwritten 
    by default

    :param file: the location of the outputfile
    :exception: raises a :class:`~EMAExceptions.VensimError` if running 
                the model failed in some way. 
                
    '''

    try:
        debug(" executing COMMAND: SIMULATE>RUNNAME|"+file+"|O")
        command("SIMULATE>RUNNAME|"+file+"|O")
        debug(r"MENU>RUN|o")
        command(r"MENU>RUN|o")
    except VensimWarning as w:
        warning((str(w)))
        raise VensimError(str(w))
        

def get_data(filename, varname, step=1):
    ''' 
    Retrieves data from simulation runs or imported data sets. 
    
    
    :param filename: the name of the .vdf file that contains the data
    :param varname: the name of the variable to retrieve data on
    :param step: steps used in slicing. Defaults to 1, meaning the full
                 recored time series is returned.
    :return: an array with the values for varname over the simulation
    
    '''
    
    vval = []
    try:
        vval, tval = vensimDLLwrapper.get_data(filename, varname)    
    except VensimWarning as w:
        warning(str(w))
        
    return vval

  
class VensimModelStructureInterface(ModelStructureInterface):
    '''
    This is a convenience extension of :class:`~model.ModelStructureInterface` 
    that can be used as a base class for performing EMA on Vensim models. This 
    class will handle starting Vensim, loading a model, setting parameters
    on the model, running the model, and retrieving the results. To this end
    it implements:
    
    * `__init__`
    * `model_init`
    * `run_model`
    
    For the simplest case, it is sufficient to only specify _`_init__` in more
    detail. That is, specify the uncertainties and the outcomes. For
    more elaborate cases, for example when using different policies, it might
    be necessary to overwrite or extent `model_init`, while for dealing with
    lookups etc. it might be necessary to also extent `run_model`. The 
    examples folder contains examples of each of these extensions. 
    
    .. note:: This class relies on the Vensim DLL, thus a complete installation 
              of Vensim DSS is needed. 
    
    '''
    
    #: attribute that can be set when one wants to load a cin file
    cin_file = None
    
    model_file = None
    '''
    The path to the vensim model to be loaded.
    
    **note:** The model file should be a `.vpm` file
    
    '''
    
    #: default name of the results file (default: 'Current.vdf')
    result_file = r'\Current.vdf'

    #: attribute used for getting a slice of the results array instead of the
    #: full array. This can cut down the amount of data saved. Alternatively,
    #: one can specify in Vensim the time steps for saving results
    step = 1 

    
    def __init__(self, working_directory, name):
        """interface to the model
        
        :param working_directory: working_directory for the model. 
        :param name: name of the modelInterface. The name should contain only
                     alphanumerical characters. 
        
        .. note:: Anything that is relative to `self.working_directory`
                  should be specified in `model_init` and not
                  in `__init__`. Otherwise, the code will not work when running
                  it in parallel. The reason for this is that the working
                  directory is being updated by parallelEMA to the worker's 
                  separate working directory prior to calling `model_init`.
        """
        super(VensimModelStructureInterface, self).__init__(working_directory, 
                                                            name)
        self.outcomes.append(Outcome('TIME' , time=True))
        
        self.outcomes = list(self.outcomes)
        
        debug("vensim interface init completed")
        

    def model_init(self, policy, kwargs):
        """
        Init of the model, The provided implementation here assumes
        that `self.model_file`  is set correctly. In case of using different
        vensim models for different policies, it is recomended to extent
        this method, extract the model file from the policy dict, set 
        `self.model_file` to this file and then call this implementation through
        calling `super`.
        
        :param policy: a dict specifying the policy. In this 
                       implementation, this argument is ignored. 
        :param kwargs: additional keyword arguments. In this implementation 
                       this argument is ignored.
        """

        load_model(self.working_directory+self.model_file) #load the model
        debug("model initialized successfully")

        be_quiet() # minimize the screens that are shown
        
        try:
            initialTime  = get_val('INITIAL TIME')
            finalTime = get_val('FINAL TIME')
            timeStep = get_val('TIME STEP')
            savePer = get_val('SAVEPER')
             
            if savePer > 0:
                timeStep = savePer
            
            self.runLength = int((finalTime - initialTime)/timeStep +1)
        except VensimWarning:
            raise EMAWarning(str(VensimWarning))
    
    def run_model(self, case):
        """
        Method for running an instantiated model structure. 
        the provided implementation assumes that the keys in the 
        case match the variable names in the Vensim model. 
        
        If lookups are to be set specify their transformation from 
        uncertainties to lookup values in the extension of this method, 
        then call this one using super with the updated case dict.
        
        if you want to use cin_files, set the cin_file, or cin_files in
        the extension of this method to `self.cin_file`.
        
        :param case: the case to run
        
        
        .. note:: setting parameters should always be done via run_model.
                  The model is reset to its initial values automatically after
                  each run.  
        
        """
                
        if self.cin_file:
            try:
                read_cin_file(self.working_directory+self.cin_file)
            except VensimWarning as w:
                debug(str(w))
            else:
                debug("cin file read successfully")
            
        for key, value in case.items():
            set_value(key, value)
        debug("model parameters set successfully")
        
        debug("run simulation, results stored in " + self.working_directory+self.result_file)
        try:
            run_simulation(self.working_directory+self.result_file)
        except VensimError:
            raise

        results = {}
        error = False
        for output in self.outcomes:
            debug("getting data for %s" %output.name)
            result = get_data(self.working_directory+self.result_file, 
                              output.name 
                              )
            debug("successfully retrieved data for %s" %output.name)
            if not result == []:
                if result.shape[0] != self.runLength:
                    got = result.shape[0]
                    a = np.zeros((self.runLength))
                    a[0:result.shape[0]] = result
                    result = a
                    error = True

            if not output.time:
                result = [-1]
            else:
                result = result[0::self.step]
            try:
                results[output.name] = result
            except ValueError as e:
                print "what"
                raise e
        self.output = results   
        if error:
            raise CaseError("run not completed, got %s, expected %s" %
                            (got, self.runLength), case)  
   
    def reset_model(self):
        """
        Method for reseting the model to its initial state before runModel 
        was called
        """
      
        self.output = None
        self.result_file =r'\Current.vdf'