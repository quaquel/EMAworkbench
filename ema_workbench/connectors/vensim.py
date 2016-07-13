'''

convenience functions and classes to be used in combination with Vensim. 

Most importantly, it specifies a generic ModelStructureInterface class
for controlling vensim models. In addition, this module contains frequently 
used functions with error checking. For more fine grained control, the 
:mod:`vensimDLLwrapper` can also be used directly.

'''
from __future__ import (absolute_import, print_function, division)

import decimal
import math
import types

import numpy as np

from . import vensimDLLwrapper
from .vensimDLLwrapper import (command, get_val, VensimError, VensimWarning)
from ..em_framework import (ModelStructureInterface, Outcome,
                            ParameterUncertainty, CategoricalUncertainty)
from ..em_framework.uncertainties import AbstractUncertainty
from ..util import debug, warning, EMAError, EMAWarning, CaseError

# Created on 25 mei 2011
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['be_quiet',
           'load_model',
           'read_cin_file',
           'set_value',
           'run_simulation',
           'get_data',
           'VensimModelStructureInterface',
           'LookupUncertainty']


def be_quiet():
    '''
    this allows you to turn off the work in progress dialog that Vensim 
    displays during simulation and other activities, and also prevent the 
    appearance of yes or no dialogs.
    
    defaults to 2, suppressing all windows, for more fine grained control, use
    :mod:`vensimDLLwrapper` directly .
    '''
    vensimDLLwrapper.be_quiet(2)
    

def load_model(file_name):
    '''
    load the model 
    
    Parameters
    ----------
    file_name : str
                file name of model, relative to working directory
    
    Raises
    -------
    VensimError if the model cannot be loaded.
    
    .. note: only works for .vpm files
    
    '''
    debug("executing COMMAND: SIMULATE>SPECIAL>LOADMODEL|"+file_name)
    try:
        command("SPECIAL>LOADMODEL|"+str(file_name))
    except VensimWarning as w:
        warning(str(w))
        raise VensimError("vensim file not found")

def read_cin_file(file_name):
    '''
    read a .cin file
    
    Parameters
    ----------
    file_name : str
                file name of cin file, relative to working directory
                
    Raises
    ------
    VensimWarning if the cin file cannot be read.
    
    '''
    debug("executing COMMAND: SIMULATE>READCIN|"+file_name)
    try:
        command(r"SIMULATE>READCIN|"+str(file_name))
    except VensimWarning as w:
        debug(str(w))
        raise w

def set_value(variable, value):
    '''
    set the value of a variable to value
    
    current implementation only works for lookups and normal values. In case
    of a list, a lookup is assumed, else a normal value is assumed. 
    See the DSS reference supplement, p. 58 for details.

    Parameters
    ----------
    variable : str
               name of the variable to set.
    value : int, float, or list
            the value for the variable. **note**: the value can be either a 
            list, or an float/integer. If it is a list, it is assumed the 
            variable is a lookup.
    '''
    variable = str(variable)
    
    if type(value) == types.ListType:
        value = [str(entry) for entry in value]
        command("SIMULATE>SETVAL|"+variable+"("+ str(value)[1:-1] + ")")
    else:
        try:
            command(r"SIMULATE>SETVAL|"+variable+"="+str(value))
        except VensimWarning:
            warning('variable: \'' +variable+'\' not found')


def run_simulation(file_name):
    ''' 
    Convenient function to run a model and store the results of the run in 
    the specified .vdf file. The specified output file will be overwritten 
    by default

    Parameters
    ----------
    file_name : str
                the file name of the output file relative to the working 
                directory
                
    Raises
    ------
    VensimError if running the model failed in some way. 
                
    '''

    file_name = str(file_name)

    try:
        debug(" executing COMMAND: SIMULATE>RUNNAME|"+file_name+"|O")
        command("SIMULATE>RUNNAME|"+file_name+"|O")
        debug(r"MENU>RUN|o")
        command(r"MENU>RUN|o")
    except VensimWarning as w:
        warning((str(w)))
        raise VensimError(str(w))
        

def get_data(filename, varname, step=1):
    ''' 
    Retrieves data from simulation runs or imported data sets. 
    
    Parameters
    ----------
    filename : str
               the name of the .vdf file that contains the data
    varname : str
              the name of the variable to retrieve data on
    step : int (optional)
           steps used in slicing. Defaults to 1, meaning the full recored time 
           series is returned.
    
    Returns
    -------
    numpy array with the values for varname over the simulation
    
    '''
    
    vval = []
    try:
        vval, _ = vensimDLLwrapper.get_data(str(filename), str(varname))    
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
    
    * `src`
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
        
        interface to the model
        
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
        
        .. note:: Anything that is relative to `self.working_directory`
          should be specified in `model_init` and not
          in `src`. Otherwise, the code will not work when running
          it in parallel. The reason for this is that the working
          directory is being updated by parallelEMA to the worker's 
          separate working directory prior to calling `model_init`.
                
        """
        super(VensimModelStructureInterface, self).__init__(working_directory, 
                                                            name)
        self.outcomes.append(Outcome('TIME' , time=True))
        
        self.outcomes = list(self.outcomes)
        
        self._lookup_uncertainties = []
        
        debug("vensim interface init completed")
        

    def model_init(self, policy, kwargs):
        """
        Init of the model, The provided implementation here assumes
        that `self.model_file`  is set correctly. In case of using different
        vensim models for different policies, it is recommended to extent
        this method, extract the model file from the policy dict, set 
        `self.model_file` to this file and then call this implementation 
        through calling `super`.
        
        Parameters
        ----------
        policy : dict
                 policy to be run.
        kwargs : dict
                 keyword arguments to be used by model_intit. This
                 gives users to the ability to pass any additional 
                 arguments. 
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
            
            self.run_length = int((finalTime - initialTime)/timeStep +1)
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
        
        Parameters
        ----------
        case : dict
               keyword arguments for running the model. The case is a dict with 
               the names of the uncertainties as key, and the values to which 
               to set these uncertainties. 
        
        
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

        for lookup_uncertainty in self._lookup_uncertainties:
            # ask the lookup to transform the retrieved uncertainties to the 
            # proper lookup value
            case[lookup_uncertainty.name] = lookup_uncertainty.transform(case)
  
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
            if result is not None:
                if result.shape[0] != self.run_length:
                    got = result.shape[0]
                    data = np.empty((self.run_length))
                    data[:] = np.NAN
                    data[0:result.shape[0]] = result
                    result = data
                    error = True

            if not output.time:
                result = result[-1]
            else:
                result = result[0::self.step]
            try:
                results[output.name] = result
            except ValueError as e:
                print("what")
                raise e
        self.output = results   
        if error:
            raise CaseError("run not completed, got %s, expected %s" %
                            (got, self.run_length), case)  


    def reset_model(self):
        """
        Method for reseting the model to its initial state before runModel 
        was called
        """
      
        self.output = None
        self.result_file =r'\Current.vdf'


    def _delete_lookup_uncertainties(self):
        '''
        deleting lookup uncertainties from the uncertainty list 
        '''
        self._lookup_uncertainties = self._lookup_uncertainties[:]
        self.uncertainties = [x for x in self.uncertainties if x not in 
                              self._lookup_uncertainties]
        
class LookupUncertainty(AbstractUncertainty):
    HEARNE1 = 'hearne1'
    HEARNE2 = 'hearne2'
    APPROX =  'approximation'
    CAT = 'categories'

    error_message = "unknown transform_type for lookup uncertainty {}"
    msi = None
    y_min = None
    y_max = None
    x_min = None
    x_max = None 
    x = []
    y = []
     
    def __init__(self, lookup_type, values, name, msi, ymin=None, ymax=None):
        '''

        Parameters
        ----------
        lookup_type : {'categories', 'hearne', 'approximation'}
                      the method to be used for alternative generation. 
        values : collection
                 the values for specifying the uncertainty from which to 
                 sample.
            If 'lookup_type' is "categories", a set of alternative lookup 
                functions to  be entered as tuples of x,y points.
                Example definition: 
                LookupUncertainty([[(0.0, 0.05), (0.25, 0.15), (0.5, 0.4), 
                                    (0.75, 1), (1, 1.25)], 
                                  [(0.0, 0.1), (0.25, 0.25), (0.5, 0.75), 
                                   (1, 1.25)],
                                  [(0.0, 0.0), (0.1, 0.2), (0.3, 0.6), 
                                   (0.6, 0.9), (1, 1.25)]], 
                                   "TF3", 'categories', self )
            if 'lookup_type' is "hearne1", a list of ranges for each parameter 
                Single-extreme piecewise functions
                m: maximum deviation from l of the distortion function
                p: the point that this occurs
                l: lower end point
                u: upper end point
            If 'lookup_type' is "hearne2", a list of ranges for each 
                parameter. Double extreme piecewise linear functions with 
                variable endpoints are used to distort the lookup functions. 
                These functions are defined by 6 parameters, being m1, m2, p1, 
                p2, l and u; and the uncertainty ranges for these 6 parameters 
                should  be given as the values of this lookup uncertainty if 
                Hearne's method is chosen. The meaning of these parameters is 
                simply:
                m1: maximum deviation (peak if positive, bottom if negative) of 
                 the distortion function from l in the first segment
                p1: where this peak occurs in the x axis
                m2: maximum deviation of the distortion function from l or u in 
                    the second segment
                p2: where the second peak/bottom occurs
                l : lower end point, namely the y value for x_min
                u : upper end point, namely the y value for x_max
                Example definition:
                LookupUncertainty([(-1, 2), (-1, 1), (0, 1), (0, 1), (0, 0.5), 
                                   (0.5, 1.5)], "TF2", 'hearne', self, 0, 2)
             If 'lookup_type' is "approximation", an analytical function 
                 approximation (a logistic function) will be used, instead of a 
                 lookup. This function also has 6 parameters whose ranges should 
                 be given:
                 A: the lower asymptote
                 K: the upper asymptote
                 B: the growth rate
                 Q: depends on the value y(0)
                 M: the time of maximum growth if Q=v
                Example definition:
                TODO:
        name : str
               name of the uncertainty
        msi : VensimModelStructureInterface instance
              model structure interface, to be used for adding new 
              parameter uncertainties
        min : float
              min value the lookup function can take, this argument is 
              not needed in case of CAT
        max : float
              max value the lookup function can take, this argument is 
              not needed in case of CAT
        
        '''
        super(LookupUncertainty, self).__init__(values, name)
        self.lookup_type = lookup_type
        self.y_min = ymin
        self.y_max = ymax
        self.error_message = self.error_message.format(self.name)
        self.transform_functions = {self.HEARNE1: self._hearne1,
                                    self.HEARNE2: self._hearne2,
                                    self.APPROX: self._approx,
                                    self.CAT: self._cat}
        
        if self.lookup_type == "categories":
            msi.uncertainties.append(CategoricalUncertainty(range(len(values)), 
                                                            "c-"+self.name))
            msi._lookup_uncertainties.append(self)  
        elif self.lookup_type == "hearne1":
            msi.uncertainties.append(ParameterUncertainty(values[0], 
                                                          "m-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[1], 
                                                          "p-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[2], 
                                                          "l-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[3], 
                                                          "u-"+self.name))
            msi._lookup_uncertainties.append(self)  
        elif self.lookup_type == "hearne2":
            msi.uncertainties.append(ParameterUncertainty(values[0], 
                                                          "m1-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[1], 
                                                          "m2-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[2], 
                                                          "p1-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[3], 
                                                          "p2-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[4], 
                                                          "l-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[5], 
                                                          "u-"+self.name))
            msi._lookup_uncertainties.append(self) 
        elif self.lookup_type == "approximation":
            msi.uncertainties.append(ParameterUncertainty(values[0], 
                                                          "A-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[1], 
                                                          "K-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[2], 
                                                          "B-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[3], 
                                                          "Q-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[4], 
                                                          "M-"+self.name))
            msi._lookup_uncertainties.append(self) 
        else: raise EMAError(self.error_message)
        
    
    def _get_initial_lookup(self, name):
        '''
        Helper function to retrieve the lookup function as defined in the
        vensim model. This lookup is transformed using a distortion function.
        
        Parameters
        ----------
        name : str
               name of variable in vensim model that contains the lookup
        
        '''
        
        a = vensimDLLwrapper.get_varattrib(name, 3)[0]
        elements = a.split('],', 1)
        b = elements[1][0:-1]
              
        list1 = []
        list2 = []
        number = []
        for c in b:
            if (c != '(') and (c != ')'):
                list1.append(c) 
         
        list1.append(',')      
        for c in list1:
            if c != ',':
                number.append(c)
            else:
                list2.append(float(''.join(number)))
                number[:] = []
        x = []
        y = []
        xT = True
        for i in list2:
            if xT:
                x.append(i)
                xT = False
            else:
                y.append(i)
                xT = True
        return (x, y)
    
    def _gen_log(self, t, A, K, B, Q, M):
        '''
        
        helper function implements a logistic function
        
        Parameters
        ----------
        t : float
        A : float
        K : float
        B : float
        Q : float
        M : float
        
        '''
        decimal.getcontext().prec = 3
        ex = math.exp(-B) 
        res = A+((K-A)/(1+Q*pow(ex,t)/pow(ex,M)))
        return res
        
    def transform(self, case):
        if not self.x:
            # first time transform is called
            self.x, self.y = self._get_initial_lookup(self.name)
            self.x_min = min(self.x)
            self.x_max = max(self.x)
        try: 
            func = self.transform_functions[self.lookup_type]
            
            return func(case)
        except KeyError:
            raise EMAError(self.error_message)

    def identity(self):
        '''
        helper method that returns the elements that define an uncertainty. 
        By default these are the name, the lower value of the range and the 
        upper value of the range.
         
        '''
        # TODO this identity function is tricky. Identity is dependent on
        # the exact transform lookup_type
        return (self.name, self.values[0], self.values[1])    
    
    def _hearne1(self, case):
        m = case['m-'+self.name]
        p = case['p-'+self.name]
        l = case['l-'+self.name]
        u = case['u-'+self.name]

        for char in ['m-', 'p-', 'l-', 'u-']:
            case.pop(char+self.name)
        
        df = []
        for i in self.x:
            if i < p:
                df.append(l + ((m/(p-self.x_min))*i))
            else:
                df.append(l + m - ((m+l-u)*(i-p)/(self.x_min-p)))
        new_lookup = []
        for i in range(len(self.x)):
            new_lookup.append((self.x[i], max(min(df[i]*
                                        self.y[i], self.y_max), self.y_min)))
        return new_lookup
    
    def _hearne2(self, case):
        m1 = case['m1-'+self.name]
        m2 = case['m2-'+self.name]
        p1 = case['p1-'+self.name]
        p2 = case['p2-'+self.name]
        l = case['l-'+self.name]
        u = case['u-'+self.name]

        for char in ['m1-', 'm2-', 'p1-', 'p2-', 'l-', 'u-']:
            case.pop(char+self.name)
        
        df = [] # distortion function
        for i in self.x:
            if i < p1:
                df.append(l + ((m1/(p1-self.x_min))*i))
            else:
                if i < p2:
                    df.append(l + m1 - ((m1-m2+l-u)*(i-p1)/(p2-p1)))
                else:
                    df.append(u + m2 - (m2*(i-p2)/(self.x_max-p2)))
        new_lookup = []
        for i in range(len(self.x)):
            new_lookup.append((self.x[i], max(min(df[i]*self.y[i], self.y_max), 
                                              self.y_min)))
        return new_lookup
    
    def _approx(self, case):
        A = case['A-'+self.name]
        K = case['K-'+self.name]
        B = case['B-'+self.name]
        Q = case['Q-'+self.name]
        M = case['M-'+self.name]
        
        for char in ['A-', 'K-', 'B-', 'Q-', 'M-']:
            case.pop(char+self.name)
        
        new_lookup = []
        if self.x_max > 10:
            for i in range(int(self.x_min), int(self.x_max+1)):
                new_lookup.append((i, max(min(
                        self._gen_log(i, A, K, B, Q, M), 
                        self.y_max), self.y_min))) 
        else:
            for i in range(int(self.x_min*10), 
                           int(self.x_max*10+1), 
                           max(int(self.x_max),1)):
                new_lookup.append((i/10, 
                                   max(min(self._gen_log(i/10, 
                                           A, K, B, Q, M), 
                                           self.y_max), 
                                           self.y_min)))
        return new_lookup
    
    def _cat(self, case):
        return  self.values[case.pop('c-'+self.name)] 