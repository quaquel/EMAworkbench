'''

this is a first draft for wrapping the vensim dll in a pythonic way

by default it is assumed the dll is readily available. If this generates an 
VensimError, you have to find the location of the dll and either copy it to
C:\Windows\System32 and/or C:\Windows\SysWOW64, or use::

    vensim = ctypes.windll.LoadLibrary('location of dll')

Typically, the dll can be found in ../AppData/Local/Vensim/vendll32.dll


'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import ctypes
import sys

import numpy as np

from ..util import info, EMAError, EMAWarning

try:
    WindowsError # @UndefinedVariable
except NameError:
    WindowsError = None

# Created on 21 okt. 2010
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

class VensimWarning(EMAWarning):
    '''
    base vensim warning
    '''
    pass

class VensimError(EMAError):
    '''
    base Vensim error
    '''
    pass



try:
    vensim_single = ctypes.windll.vendll32
except AttributeError:
    vensim_single = None
except WindowsError:
    vensim_single = None
    
try:
    vensim_double = ctypes.windll.LoadLibrary('C:\Windows\SysWOW64\VdpDLL32.dll')
except AttributeError:
    vensim_double = None
except WindowsError:
    vensim_double = None

if vensim_single and vensim_double:
    vensim = vensim_single
    info("both single and double precision vensim available, using single")
elif vensim_single:
    vensim = vensim_single
    info('using single precision vensim')
elif vensim_double:
    vensim = vensim_double
    info('using single precision vensim')
else:
    raise ImportError("vensim dll not found")
del sys


def be_quiet(quietflag):
    '''
    this allows you to turn off the work in progress dialog that Vensim 
    displays during simulation and other activities, and also prevent the 
    appearance of yes or no dialogs. 
    
    use 0 for normal interaction, 1 to prevent the appearance of any work 
    in progress windows, and 2 to also prevent the appearance of any 
    interrogative dialogs'
    '''
    if quietflag > 2:
        raise VensimError("incorrect value for quietflag")
    
    return vensim.vensim_be_quiet(quietflag)


def check_status():
    '''check status is used to check the current status of the Vensim DLL, for 
    details on the return values check DSS reference chapter 12'''
    
    return vensim.vensim_check_status()


def command(command):
    '''execute a command, for details see chapter 5 of the vensim DSS manual'''
    
    return_val = vensim.vensim_command(command.encode('utf-8'))
    if return_val == 0:
        raise VensimWarning("command failed "+command)
    return return_val


def continue_simulation(num_inter):
    '''This method continues the simulation for num_inter Time steps.
    
    Parameters
    ----------
    num_inter : int
                the number of TIME_STEP iterations that should be executed 
                during the continuation
    '''
    
    return_val = vensim.vensim_continue_simulation(num_inter)
    if return_val == -1:
        raise VensimWarning("floating point error has occurred")
    
    return return_val


def finish_simulation():
    '''completes a simulation started with start simulation'''
    
    return_val = vensim.vensim_finish_simulation()
    if return_val == 0:
        raise VensimWarning("failure to finish simulation")
    return return_val

    
def get_data(filename, variable_name, tname="Time"):
    ''' 
    Retrieves data from simulation runs or imported data sets. In contrast
    to the Vensim DLL, this method retrieves all the data, and not only the 
    data for the specified length. 
    
    Parameters
    ----------
    filename : str
               the name of the .vdf file that contains the data
    variable_name : str
             the name of the variable to retrieve data on
    tname : str
            the name of the time axis against which to pull the data, 
            by default this is Time
    
    Returns
    -------
    a tuple with an  for an array for varname and and array for tname.
    
    '''
    vval = (ctypes.c_float * 1)()  
    tval = (ctypes.c_float * 1)()  
    maxn = ctypes.c_int(0)
    
    filename = filename.encode('utf-8')
    varname = variable_name.encode('utf-8')
    tname = tname.encode('utf-8')
    
    return_val = vensim.vensim_get_data(filename, 
                                        varname, 
                                        tname, 
                                        vval, 
                                        tval, 
                                        maxn)
    
    if return_val == 0:
        raise VensimWarning("variable "+variable_name+" not found in dataset")
    
    vval = (ctypes.c_float * int(return_val))()  
    tval = (ctypes.c_float * int(return_val))()  
    maxn = ctypes.c_int(int(return_val))
    
    return_val = vensim.vensim_get_data(filename,\
                                         varname,\
                                         tname,\
                                         vval,\
                                         tval,\
                                         maxn)

    vval = np.ctypeslib.as_array(vval)
    tval = np.ctypeslib.as_array(tval)
    
    return vval, tval


def get_dpval(name, varval):
    '''
    use this to get the value of a variable during a simulation, as a game 
    is progressing, or during simulation setup. This function is only useful 
    if you are using the double precision Vensim DLL 
    
    currently not implemented
    '''
    
    raise NotImplementedError


def get_dpvecvals(vecoff, dpvals, veclen):
    '''
    This is the same as get_vecvals except it takes a double vector to store 
    values. This method is only meaningful in case of the double precision DLL 
    
    currently not implemented
    '''
    
    raise NotImplementedError

def get_info(infowanted):
    '''
    Use this function to get information about vensim, for details see DSS 
    reference chapter 12
    
    Parameters
    ----------
    infowanted : int
                 field that specifies the info wanted
    '''
    
    buf = ctypes.create_string_buffer(b"", 512)
    maxBuf = ctypes.c_int(512)    
    a = vensim.vensim_get_info(infowanted, buf, maxBuf)
    buf = ctypes.create_string_buffer(b"", int(a))
    maxBuf = ctypes.c_int(int(a))
    vensim.vensim_get_info(infowanted, buf, maxBuf)
    
    result = repr(buf.raw)
    result = result.strip()
    result = result.rstrip("'")
    result = result.lstrip("'")
    result = result.split(r"\x00")
    result = result[0:-2]
    return result


def get_sens_at_time(filename, varname, timename, attime, vals, maxn):
    '''
    Get results from a sensitivity run at a specific type and across 
    sensitivity runs.
    
    currently not implemented
    '''
    raise NotImplementedError


def get_substring():
    '''
    Utility function that is designed to make it easier to work with 
    get_varnames, get_info, and get_varattribs. 
    
    currently not implemented
    '''
    raise NotImplementedError


def get_val(name):
    '''
    This function returns the value of a variable during a simulation, as a 
    game is progressing, or during simulation setup
    
    Parameters
    ----------
    name : str
           the name of variable for which one wants to retrieve the value.
    
    '''
    value = ctypes.c_float(0)
    return_val = vensim.vensim_get_val(name.encode('utf-8'), ctypes.byref(value))
    if return_val == 0:
        raise VensimWarning("variable not found")
    
    return value.value


def get_varattrib(varname, attribute):
    '''
    This function can be used to access the attributes of a variable.
    
    Parameters
    ----------
    varname : str
              name for which you want attribute
    attribute : int
                attribute you want 
    
    Notes
    -----
    
    ====== =============
    number meaning
    ====== =============
    1      Units, 
    2      the comment, 
    3      the equation, 
    4      causes, 
    5      uses, 
    6      initial causes only, 
    7      active causes only, 
    8      the subscripts the variable has, 
    9      all combinations those subscripts create,
    10     the combination of subscripts that would be used by a graph tool, 
    11     the minimum value set in the equation editor, 
    12     the maximum and 
    13     the range, 
    14     the variable type (returned as "Level" etc) and 
    15     the main group of a variable
    ====== =============
    
    ''' 
    buf = ctypes.create_string_buffer("", 10)
    maxBuf = ctypes.c_int(10)
    
    bufferlength = vensim.vensim_get_varattrib(varname.encode('utf-8'), 
                                               attribute, 
                                               buf, 
                                               maxBuf)
    if bufferlength == -1:
        raise VensimWarning("variable not found")
    
    buf = ctypes.create_string_buffer("", int(bufferlength))
    maxBuf = ctypes.c_int(int(bufferlength))       
    vensim.vensim_get_varattrib(varname.encode('utf-8'), attribute, buf, maxBuf)
    
    result = repr(buf.raw)
    result = result.strip()
    result = result.rstrip("'")
    result = result.lstrip("'")
    result = result.split(r"\x00")
    result = [varname for varname in result if len(varname) != 0]
    
    return result


def get_varnames(filter='*', vartype=0):  # @ReservedAssignment
    '''
    This function returns variable names in the model a filter can be specified 
    in the same way as Vensim variable Selection filter  (use * for all), 
    vartype is an integer that specifies the types of variables you want to 
    see. 
    (see DSS reference chapter 12 for details) 
    
    Parameters
    ----------
    filter : str
             selection filter, use \* for all. 
    vartype : int
              variable type to retrieve. See table
    
    
    Returns
    -------
    a list with the variable names

    Notes
    -----
    ====== =============
    number meaning
    ====== =============
    0      all
    1      levels
    2      auxiliaries 
    3      data
    4      initial
    5      constant
    6      lookup
    7      group
    8      subscript
    9      constraint
    10     test input
    11     time base
    12     gaming
    ====== =============

    '''
    
    filter = ctypes.c_char_p(filter)  # @ReservedAssignment
    vartype = ctypes.c_int(vartype)
    buf = ctypes.create_string_buffer("", 512)
    maxBuf = ctypes.c_int(512)    

    a = vensim.vensim_get_varnames(filter, vartype, buf, maxBuf)
    buf = ctypes.create_string_buffer("", int(a))
    maxBuf = ctypes.c_int(int(a))
    vensim.vensim_get_varnames(filter, vartype, buf, maxBuf)
    
    varnames = repr(buf.raw)
    varnames = varnames.strip()
    varnames = varnames.rstrip("'")
    varnames = varnames.lstrip("'")
    varnames = varnames.split(r"\x00")
    varnames = [varname for varname in varnames if len(varname) != 0]

    return varnames


def get_varoff(varname):
    '''
    This function is intended for use with get_vecvals. By filling up a 
    vector of offsets you can speed the retrieval of multiple values 
    
    currently not implemented
    '''
    
    raise NotImplementedError

def get_vecvals(vecoff, vals, nvals):
    '''gets a vector of values at the current simulation time.
    
    currently not implemented
    '''
    
    raise NotImplementedError


def set_parent_window(window, r1, r2):
    '''
    This is used to set a window that will be the owner of an dialogs or 
    message boxes that Vensim presents.
    
    currently not implemented 
    '''
    
    raise NotImplementedError


def show_sketch(sketchnum, wantscroll, zoompercent, pwindow):
    '''
    Use this function to display a model diagram 
    
    currently not implemented
    '''
    
    raise NotImplementedError


def start_simulation(loadfirst, game, overwrite):
    '''
    Start a simulation that will be performed a bit at a time.
    
    Parameters
    ----------
    loadfirst : bool
                if True the run resulting from the simulationshould be loaded 
                first in the list of runs
    game : int 
           if 0 treat simulation as a normal simulation, if 1, start a new 
           game, if 2, continue with a game
    overwrite : bool
                if True, automatically overwrite existing files when simulation 
                starts
                      
    '''
    
    return_val =vensim.vensim_start_simulation(loadfirst, game, overwrite)
    if return_val == 0:
        raise VensimWarning("simulation not started")
    
    return return_val


def synthesim_vals(offset, tval, varval):
    ''' 
    This is a specialized function that uses memory managed by Vensim 
    to give access to values while SyntheSim is active.
    
    currently not implemented
    '''
    
    raise NotImplementedError


def tool_command(command, window, aswiptool):
    '''
    Perform a command that will cause output to be created, or the printing or 
    exporting of the contents of a currently displayed item.
    
    currently not implemented
    '''
    
    raise NotImplementedError


def contextAdd(wantcleanup):
    '''
    creates a new context for the server version of Vensim 
    
    currently not implemented
    '''
    
    raise NotImplementedError

  
def contextDrop(context):
    '''
    drops a context that was created by contextAdd
    
    currently not implemented
    '''
    
    raise NotImplementedError


def use_double_precision():
    '''
    convenience function for changing reference to dll to dll for double 
    precision. 
    
    
    In order to ensure that double precision is used when running in parallel,
    call this function at the top of the module in which you define the model
    interface.
    
    '''
    
    global vensim
    try:
        vensim = ctypes.windll.LoadLibrary('C:\Windows\SysWOW64\VdpDLL32.dll')
    except WindowsError:
        raise EMAError("double precision vensim dll not found")
            