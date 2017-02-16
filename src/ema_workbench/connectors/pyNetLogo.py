'''
Created on 21 mrt. 2013

@author: j.h. kwakkel

To do: check Mac support and handling of custom directories

'''
from __future__ import unicode_literals, absolute_import
import os
import re
import sys


import numpy as np
import pandas as pd
import jpype

from ..util.ema_logging import debug, info, warning

__all__ = ['NetLogoException',
           'NetLogoLink']
    
PYNETLOGO_HOME = os.path.dirname(os.path.abspath(__file__))

#Jar supports NetLogo 5.2 or 6.0
module_name = {'5':'netlogoLink_v52.NetLogoLink',
               '6':'netlogoLink_v6.NetLogoLink'}
jar_sep ={'win32':';',
          'darwin':':'}


def find_netlogo(path):
    '''find the most recent version of netlogo in the specified directory
    
    Parameters
    ----------
    path : str
    
    Returns
    -------
    str
    
    Raises
    ------
    IndexError if no netlogo version is found in path
    
    '''
    
    path = os.path.abspath(path)
    netlogo_versions = [entry for entry in os.listdir(path) if 'netlogo' in 
                        entry.lower()]
    
    # sort handles version numbers properly as long as pattern of 
    # directory name is not changed
    netlogo_versions.sort(reverse=True)
    
    return netlogo_versions[0]


def find_jars(path):
    '''find all jars in directory and return as list
    '''
    
    jars = []
    for root, _, files in os.walk(path):
        for file in files:  # @ReservedAssignment
            if file.endswith(".jar"):
                jars.append(os.path.join(root, file))
    return jars


def establish_netlogoversion(path):
    # TODO: python3 compliance
    pattern = re.compile(r'(?:(\d+)\.)?(?:(\d+)\.)?(\*|\d+)$')
    
    netlogo = os.path.basename(os.path.normpath(path))
    match = pattern.search(netlogo)
    version = match.group()
    
    debug('netlogo version is: '+version)
    
    main_version = version[0]
    return main_version


def find_netlogo_windows():
    netlogo = None
    if os.environ['PROGRAMW6432']:
        paths = [os.environ['PROGRAMFILES(X86)'],
                 os.environ['PROGRAMW6432']] 
        for path in paths:
            try:
                netlogo = find_netlogo(path)
            except IndexError:
                pass
            else:
                netlogo = os.path.join(path, netlogo)
    else:
        path = os.environ['PROGRAMFILES']
        try:
            netlogo = find_netlogo(path)
        except IndexError:
            pass
        else:
            netlogo = os.path.join(path, netlogo)
    
    return netlogo


def find_netlogo_mac():
    paths = ['/Applications', 
             os.path.join(os.getenv('HOME'), 'Applications')]
    netlogo = None
    for path in paths:
        try:
            netlogo = find_netlogo(path)
        except IndexError:
            pass
        else:
            netlogo = os.path.join(path, netlogo)
        
    return netlogo


def find_netlogo_linux():
    raise NotImplementedError

def get_netlogo_home():
    if sys.platform=='win32':
        netlogo_home = find_netlogo_windows()
    elif sys.platform=='darwin':
        netlogo_home = find_netlogo_mac()
    else:
        netlogo_home = find_netlogo_linux()
        
    debug('NETLOGO_HOME: '+netlogo_home)
        
    return netlogo_home
    

class NetLogoException(Exception):
    """Basic project specific exception """
    
    pass

    
class NetLogoLink():
    '''
    
    Create a link with NetLogo. Underneath, the NetLogo JVM is started
    through Jpype.
    
    
    :param gui: boolean, if true run NetLogo with gui, otherwise run in 
                headless mode. Defaults to false.
    :param thd: boolean, if true start NetLogo in 3d mode. Defaults to 
                false
    :param nl_dir: string - full path to NetLogo .exe directory, use if
                NetLogo is not in the default location
    :param nl_version: string - NetLogo version under nl_dir (5.2 or 6.0)
    
    '''
    
    def __init__(self, gui=False, thd=False, netlogo_home=None, 
                 netlogo_version=None, jvm_home=None):
        '''
        
        Create a link with NetLogo. Underneath, the NetLogo JVM is started
        through Jpype.
        
        
        :param gui: boolean, if true run NetLogo with gui, otherwise run in 
                    headless mode. Defaults to false.
        :param thd: boolean, if true start NetLogo in 3d mode. Defaults to 
                    false
        :param nl_dir: string - full path to NetLogo .exe directory, use if
                    NetLogo is not in the default location
        :param nl_version: string - NetLogo version under nl_dir (5.2 or 6.0)
        
        '''
        if not netlogo_home:
            netlogo_home = get_netlogo_home()
        if not netlogo_version:
            netlogo_version = establish_netlogoversion(netlogo_home)
        if not jvm_home:
            jvm_home = jpype.get_default_jvm_path()  
        
        
        if not jpype.isJVMStarted():
            jars = find_jars(netlogo_home)
            jars.append(os.path.join(PYNETLOGO_HOME, 'java', 'netlogoLink_combined.jar'))     
            joined_jars = jar_sep[sys.platform].join(jars)
            jarpath = '-Djava.class.path={}'.format(joined_jars)
            try:
                debug("starting jvm: {} {}".format(jvm_home, jarpath))
                jpype.startJVM(jvm_home, jarpath)
                debug("JVM started")  
            except RuntimeError as e:
                warning('failed to start JVM using jvm_home')
                raise e
            #jpype.java.lang.System.setProperty('user.dir', netlogo_home)
            if sys.platform=='darwin':
                debug('set state to headless on mac os')
                jpype.java.lang.System.setProperty("java.awt.headless", "true");            
            
        
        link = jpype.JClass(module_name[netlogo_version])
        debug('NetLogoLink class found')

        if sys.platform == 'darwin' and gui:
            info('on mac only Headless mode is supported')
            gui=False
        
        self.link = link(jpype.java.lang.Boolean(gui),jpype.java.lang.Boolean(thd))
        debug('NetLogoLink class instantiated')
        
            
    def load_model(self, path):
        '''
        
        Load a NetLogo model.
        
        :param path: the absolute path to the NetLogo model
        :raise: IOError in case the  model is not found
        :raise: NetLogoException wrapped around NetLogo exceptions. 
        
        '''
        try:
            self.link.loadModel(path)
        except jpype.JException(jpype.java.io.IOException) as ex:
            raise IOError(ex.message())
        except jpype.JException(jpype.java.org.nlogo.api.LogoException) as ex:
            raise NetLogoException(ex.message())
        except jpype.JException(jpype.java.org.nlogo.api.CompilerException) as ex:
            raise NetLogoException(ex.message())
        except jpype.JException(jpype.java.lang.InterruptedException) as ex:
            raise NetLogoException(ex.message())
        else:
            debug('loaded model successfully')


    def kill_workspace(self):
        '''
        
        Close NetLogo and shut down the JVM.
        
        '''
        
        self.link.killWorkspace()
        
        
    def command(self, netlogo_command):
        '''
        
        Execute the supplied command in NetLogo
        
        :param netlogo_command: a string with a valid netlogo command
        :raises: NetLogoException in case of either a LogoException or 
                CompilerException being raised by netlogo.
        
        '''
        
        try:
            self.link.command(netlogo_command)
        except jpype.JException(jpype.java.org.nlogo.api.LogoException) as ex:
            raise NetLogoException(ex.message())
        except jpype.JException(jpype.java.org.nlogo.api.CompilerException) as ex:
            raise NetLogoException(ex.message())
        except jpype.JException(jpype.java.org.nlogo.nvm.EngineException) as ex:
            raise NetLogoException(ex.message())

            
    def report(self, netlogo_reporter):
        '''
        
        Every reporter (commands which return a value) that can be called in 
        the NetLogo Command Center can be called with this method.
        
        :param netlogo_reporter: a valid netlogo reporter 
        :raises: NetlogoException
        
        '''
        
        try:
            result = self.link.report(netlogo_reporter)
            return self._cast_results(result)
        except jpype.JException(jpype.java.org.nlogo.api.LogoException) as ex:
            raise NetLogoException(ex.message())
        except jpype.JException(jpype.java.org.nlogo.api.CompilerException) as ex:
            raise NetLogoException(ex.message()) 
        except jpype.JException(jpype.java.lang.Exception) as ex:
            raise NetLogoException(ex.message()) 
        
        
    def patch_report(self, netlogo_reporter):
        '''
        
        Reports a Pandas dataframe with the same shape as the NetLogo world, with
        column labels and row indices following pxcor and pycor patch coordinates.
        The values of the dataframe correspond to patch attributes.
        
        :param netlogo_reporter: valid patch attribute 
        :raises: NetlogoException
        :returns: Dataframe containing NetLogo patch attributes
        
        '''

        try:
            extents = self.link.report('(list min-pxcor max-pxcor min-pycor max-pycor)')
            extents = self._cast_results(extents).astype(int)

            results_df = pd.DataFrame(index=range(extents[2],extents[3]+1,1),
                                      columns=range(extents[0],extents[1]+1,1))
            results_df.sort_index(ascending=False, inplace=True)
            if self.NL_VERSION == '6.0':
                resultsvec = self.link.report('map [[?1] -> [{0}] of ?1] sort patches'.format(netlogo_reporter))
            else:
                resultsvec = self.link.report('map [[{0}] of ?] sort patches'.format(netlogo_reporter))
            resultsvec = self._cast_results(resultsvec)
            results_df.ix[:,:] = resultsvec.reshape(results_df.shape)
            
            return results_df
            
        except jpype.JException(jpype.java.org.nlogo.api.LogoException) as ex:
            raise NetLogoException(ex.message())
        except jpype.JException(jpype.java.org.nlogo.api.CompilerException) as ex:
            raise NetLogoException(ex.message()) 
        except jpype.JException(jpype.java.lang.Exception) as ex:
            raise NetLogoException(ex.message())
            
    
    def patch_set(self, attribute, data):
        '''
        
        Set patch attributes from a Pandas dataframe
        
        :param attribute: valid NetLogo patch attribute
        :param data: Pandas dataframe with same dimensions as NetLogo world
        :raises: NetLogoException
        
        '''
    
        try:
            np.set_printoptions(threshold = np.prod(data.shape))
            datalist = '['+str(data.as_matrix().flatten()).strip('[ ')
            datalist = ' '.join(datalist.split())
            if self.NL_VERSION == '6.0':
                command = '(foreach map [[?1] -> [pxcor] of ?1] sort patches map [[?2] -> [pycor] of ?2] \
                            sort patches {0} [[?1 ?2 ?3 ] -> ask patch ?1 ?2 [set {1} ?3]])'.format(datalist, attribute)
            else:
                command = '(foreach map [[pxcor] of ?] sort patches map [[pycor] of ?] \
                            sort patches {0} [ask patch ?1 ?2 [set {1} ?3]])'.format(datalist, attribute)

            self.link.command(command)
            
        except jpype.JException(jpype.java.org.nlogo.api.LogoException) as ex:
            raise NetLogoException(ex.message())
        except jpype.JException(jpype.java.org.nlogo.api.CompilerException) as ex:
            raise NetLogoException(ex.message()) 
        except jpype.JException(jpype.java.lang.Exception) as ex:
            raise NetLogoException(ex.message())
        
        
    def repeat_command(self, netlogo_command, reps):
        '''
        
        Execute the supplied command in NetLogo a given number of times
        
        :param netlogo_command: a string with a valid NetLogo command
        :param reps: number of times to repeat commands
        :raises: NetLogoException in case of either a LogoException or 
                CompilerException being raised by NetLogo.
        
        '''
    
        try:
            commandstr = 'repeat {0} [{1}]'.format(reps, netlogo_command)
            self.link.command(commandstr)
        except jpype.JException(jpype.java.org.nlogo.api.LogoException) as ex:
            raise NetLogoException(ex.message())
        except jpype.JException(jpype.java.org.nlogo.api.CompilerException) as ex:
            raise NetLogoException(ex.message())
        except jpype.JException(jpype.java.org.nlogo.nvm.EngineException) as ex:
            raise NetLogoException(ex.message())
        

    def repeat_report(self, netlogo_reporter, reps):
        '''
        
        Execute a reporter (or multiple reporters if a list of strings is given as input)
        over a number of ticks.
        
        :param netlogo_reporter: valid NetLogo reporters (string or list of strings)
        :param reps: number of ticks for which to return values
        :raises: NetLogoException
        :returns: Dataframe of reported values indexed by ticks, with columns 
         for each reporter.
        
        '''
        
        if isinstance(netlogo_reporter, str):
            cols = [netlogo_reporter]
        elif isinstance(netlogo_reporter, list):
            cols = netlogo_reporter   
        else:
            raise NetLogoException("Unknown datatype")
        
        results_df = pd.DataFrame(columns=cols)
        
        for _ in range(reps):    
            for reporter in results_df.columns:
                try:
                    tick = self._cast_results(self.link.report('ticks'))
                    result = self.link.report(reporter)
                    results_df.loc[tick, reporter] = self._cast_results(result)
                except jpype.JException(jpype.java.org.nlogo.api.LogoException) as ex:
                    raise NetLogoException(ex.message())
                except jpype.JException(jpype.java.org.nlogo.api.CompilerException) as ex:
                    raise NetLogoException(ex.message()) 
                except jpype.JException(jpype.java.lang.Exception) as ex:
                    raise NetLogoException(ex.message()) 
            
            self.link.command('go')
                
        return results_df


    def _cast_results(self, results):
        '''
        
        Convert the results to the proper python data type. The NLResults
        object knows its datatype and has converter methods for each.
        
        :param results; the results from report
        :returns: a correct python version of the results
        
        '''
        
        java_dtype = results.type
        
        if java_dtype == "Boolean":
            results = results.getResultAsBoolean()
            if results == 1:
                return True
            else:
                return False
        elif java_dtype == "String":
            return results.getResultAsString()       
        elif java_dtype == "Integer":
            return results.getResultAsInteger()
        elif java_dtype == "Double":
            return results.getResultAsDouble()
        elif java_dtype == "BoolList":
            results = results.getResultAsBooleanArray()
            
            tr = []
            for entry in results:
                if entry == 1:
                    tr.append(True)
                else:
                    tr.append(False)
            return tr
        elif java_dtype == "StringList":
            return results.getResultAsStringArray()   
        elif java_dtype == "IntegerList":
            return results.getResultAsIntegerArray() 
        elif java_dtype == "DoubleList":
            return np.array(results.getResultAsDoubleArray())
        else:
            raise NetLogoException("Unknown datatype")