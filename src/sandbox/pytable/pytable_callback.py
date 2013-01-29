'''
Created on Apr 16, 2012

@author: localadmin

'''
import sys
import tables
import re
import datetime
import inspect

from tables.exceptions import NoSuchNodeError

from expWorkbench.util import AbstractCallback
from expWorkbench.EMAlogging import warning
from expWorkbench.uncertainties import CategoricalUncertainty,\
                                       ParameterUncertainty,\
                                       INTEGER
from expWorkbench import EMAlogging
from expWorkbench.EMAexceptions import EMAError


SVN_ID = '$Id: pytable_callback.py 820 2012-05-03 06:08:16Z jhkwakkel $'

# I moved some warnings from pytables upto the workbench. This is based
# on tables.path r. 102
class NullDevice():
    def write(self, s):
        pass

_pythonIdRE = re.compile('^[a-zA-Z_][a-zA-Z0-9_]*$')
warnInfo = (
"you will not be able to use natural naming to access this object; "
"using ``getattr()`` will still work, though" )


def format_svn_id(SVN_ID):
    '''
    Helper function for formatting the SVN_ID in a more readable form.
    This formatter keeps the file name, the revision number, and the date
    of the revision. 
    
    :param SVN_ID: the SVN_ID field of a module
    :returns: a formatted string representation of the SVN_ID
    
    '''
    
    svnidrep = r'^\$Id: (?P<filename>.+) (?P<revision>\d+) (?P<date>\d{4}-\d{2}-\d{1,2}) (?P<time>\d{2}:\d{2}:\d{2})Z (?P<user>\w+) \$$'
    mo = re.match(svnidrep, SVN_ID)
    svn_id = '%s - revision %s (%s)' % mo.group('filename', 'revision', 'date')
    return svn_id

def create_hdf5_ema__project(projectName='',
                             projectOwner='',
                             projectDescription='',
                             fileName=None):
    '''
    generic function that generates a project and sets up the project. 
    '''

    h5file = tables.openFile(filename=fileName, 
                    mode='w', 
                    title=projectName)
    h5file.root._v_attrs.project_name = projectName
    h5file.root._v_attrs.project_owner=projectOwner
    h5file.root._v_attrs.project_description=projectDescription
    h5file.root._v_attrs.date_created = datetime.datetime.now().date()
    
    h5file.root._v_attrs.created_with = format_svn_id(SVN_ID)
    
    for group in ['experiments', 'analysis']:
        group = h5file.createGroup(r'/', 
                            name=group)
        group._v_attrs.last_modified = datetime.datetime.now()
    h5file.flush()
    h5file.close()


class HDF5Callback(AbstractCallback):
    """ 
    New callback system, based on hdf5 and pytables
    """
    
    def __init__(self, 
                 uncs, 
                 outcomes, 
                 nrOfExperiments, 
                 reporting_interval=100,
                 fileName=None,
                 experimentName=None):
        '''
        
        :param fileName: if the filename exist, the file is opened. If the file
                         does not exist, a new file wih the same name is 
                         created. 
        :param experimentsName: name of the series of experiments                 
        
        
        '''
        if not fileName: 
            raise EMAError("no file name specified for hdf5 file")
        elif not experimentName:
            raise EMAError("no experiments name specified ")
        
        
        super(HDF5Callback, self).__init__(uncs, 
                                              outcomes, 
                                              nrOfExperiments, 
                                              reporting_interval)
        try:
            self.h5file = tables.openFile(filename=fileName,mode='r+')
        except IOError as e:
            EMAlogging.warning("file %s does not exist, a new file is created" %fileName)
            create_hdf5_ema__project(fileName=fileName)
            self.h5file = tables.openFile(filename=fileName, 
                                         mode='r+', 
                                         )

        # make a group for the new series of experiments
        # this raises an error if the group already exists
        self.experiments = self.h5file.createGroup('/experiments', experimentName)
        self.experiments._v_attrs.last_modified = datetime.datetime.now()

        #make table for experiments
        self.designs = self.make_designs_table(self.experiments, uncs)
        
        #some reflection
        stack = inspect.stack()
        first_entry = stack[1][0]
        ensemble = first_entry.f_locals['self']
        sampler = ensemble.sampler
        modelInterfaces = ensemble._modelStructures
      
        self.experiments._v_attrs.ensemble_svnid = format_svn_id(inspect.getmodule(ensemble).SVN_ID)
        for i, mi in enumerate(modelInterfaces):
            i = i+1
            self.experiments._v_attrs['mi_%s' %i] = mi.__class__.__name__
            self.experiments._v_attrs['mi_svnid_%s' %i] = format_svn_id(inspect.getmodule(mi).SVN_ID)

        self.designs._v_attrs.sampler = sampler.__class__.__name__
        self.designs._v_attrs.sampler_svnid = format_svn_id(inspect.getmodule(sampler).SVN_ID)
           
        self.design = self.designs.row
        
        self.outcomes = [outcome.name for outcome in outcomes]
        self.nrOfExperiments = nrOfExperiments

        
    def make_designs_table(self, group, uncs):
        #determine data types of uncertainties
        expDescription = {}
        self.categoricals = []
        for i, uncertainty in enumerate(uncs):
            name = uncertainty.name
            dataType = tables.FloatCol(pos=i+1) #@UndefinedVariable
            
            if isinstance(uncertainty, CategoricalUncertainty):
                dataType = tables.StringCol(16, pos=i+1) #@UndefinedVariable
                self.categoricals.append(name)
            elif isinstance(uncertainty, ParameterUncertainty) and\
                          uncertainty.dist==  INTEGER:
                dataType = tables.IntCol(pos=i+1) #@UndefinedVariable
            expDescription[name] = dataType
            
            if not _pythonIdRE.match(name):
                warning( " %r object name is not a valid Python identifier "
                       "it does not match the pattern ``%s``; %s"
                       % (name, _pythonIdRE.pattern, warnInfo))
            
        expDescription['model'] = tables.StringCol(16, pos=i+2)#@UndefinedVariable
        expDescription['policy'] =  tables.StringCol(16, pos=i+3)#@UndefinedVariable
           
        temp = sys.stderr 
        sys.stderr = NullDevice()
        experiments = self.h5file.createTable(group, 
                                           'designs', 
                                           expDescription)
        sys.stderr = temp   
        return experiments


    def __store_case(self, case, model, policy):
        for key, value in case.items():
            if key in self.categoricals:
                value =  repr(value)
            
            self.design[key] = value
        self.design['model'] = model
        self.design['policy'] = policy
        self.design.append()
            
    def __store_result(self, result):
        for outcome in self.outcomes:
            try:
                node = self.h5file.getNode(where=self.experiments,
                                           name=outcome)
            except NoSuchNodeError :
                if not _pythonIdRE.match(outcome):
                    warning( " %r object name is not a valid Python identifier "
                           "it does not match the pattern ``%s``; %s"
                           % (outcome, _pythonIdRE.pattern, warnInfo))
                
                try:
                    shapeResults = result[outcome].shape
                    if len(shapeResults) >0:
                        ncol = shapeResults[0] 
                    else:
                        ncol= 1
                except AttributeError:
                    #apparently the outcome is not an array but a scalar
                    ncol=1
                    
                temp = sys.stderr 
                sys.stderr = NullDevice()
                node = self.h5file.createCArray(where=self.experiments,
                                         name=outcome,
                                         atom=tables.Float32Atom(),#@UndefinedVariable
                                         shape=(self.nrOfExperiments, ncol)
                                         )
                sys.stderr = temp   
            
            node[self.i, :] = result[outcome]
    
    def __call__(self, case, policy, name, result ):
        '''
        Method responsible for storing results. This method calls 
        :meth:`super`, thus utilizing the logging provided there
        
        :param case: the case to be stored
        :param policy: the name of the policy being used
        :param name: the name of the model being used
        :param result: the result dict. This implementation assumes that
                       the values in this dict are numpy array instances. Two
                       types of instances are excepted: single values and
                       1-D arrays. 
        
        '''
        #store the case
        self.__store_case(case, name, policy.get('name'), )
        
        #store results
        self.__store_result(result)
        self.h5file.flush()
        
        super(HDF5Callback, self).__call__(case, policy, name, result)
        
    def get_results(self):
        return self.h5file