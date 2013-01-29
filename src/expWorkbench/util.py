'''
Created on 13 jan. 2011

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

This module provides various convenience functions and classes.

'''
from __future__ import division
import numpy as np
np = np
import cPickle
import os
import bz2
import math

from deap import creator, base

from ema_logging import info, debug, warning
from expWorkbench.uncertainties import CategoricalUncertainty,\
                                       ParameterUncertainty,\
                                       INTEGER
from expWorkbench import EMAError

SVN_ID = '$Id: util.py 1104 2013-01-24 16:43:50Z wlauping $'
__all__ = ['AbstractCallback',
           'DefaultCallback',
           'load_results',
           'save_results',
           'save_optimization_results',
           'load_optimization_results',
           'transform_old_cPickle_to_new_cPickle',
           'experiments_to_cases',
           'merge_results']

class AbstractCallback(object):
    '''
    Base class from which different call back classes can be derived.
    Callback is responsible for storing the results of the runs.
    
    '''
    
    i = 0
    reporting_interval = 100
    results = []
    
    def __init__(self, 
                 uncertainties, 
                 outcomes,
                 nrOfExperiments,
                 reporting_interval=100):
        '''
        
        :param uncertainties: list of :class:`~uncertianties.AbstractUncertainty` 
                              children
        :param outcomes: list of :class:`~outcomes.Outcome` instances
        :param nrOfExperiments: the total number of runs
        
        '''
        self.reporting_interval = reporting_interval
        pass    
    
    def __call__(self, case, policy, name, result):
        '''
        Method responsible for storing results. The implementation in this
        class only keeps track of how many runs have been completed and 
        logging this. 
        
        :param case: the case to be stored
        :param policy: the name of the policy being used
        :param name: the name of the model being used
        :param result: the result dict
        
        '''
        
        self.i+=1
        debug(str(self.i)+" cases completed")
        
        if self.i % self.reporting_interval == 0:
            info(str(self.i)+" cases completed")

    def get_results(self):
        """
        method for retrieving the results. Called after all experiments have 
        been completed
        """
        self.results
        

class OldCallback(AbstractCallback):
    """ 
    default callback system
    callback can be used in performExperiments as a means for specifying 
    the way in which the results should be handled. If no callback is 
    specified, this default implementation is used. This one can be 
    overwritten or replaced with a callback of your own design. For 
    example if you prefer to store the result in a database or write 
    them to a text file
    """
    
    def __call__(self, case, policy, name, result ):
        super(OldCallback, self).callback(case, policy, name, result)
        
        a = (case, policy, name)
        b = (a, result)
        self.results.append(b)

class DefaultCallback(AbstractCallback):
    """ 
    default callback system
    callback can be used in performExperiments as a means for specifying 
    the way in which the results should be handled. If no callback is 
    specified, this default implementation is used. This one can be 
    overwritten or replaced with a callback of your own design. For 
    example if you prefer to store the result in a database or write 
    them to a text file
    """
    
    i = 0
    cases = None
    policies = None
    names = None   
    results = {}
    
    def __init__(self, 
                 uncs, 
                 outcomes, 
                 nrOfExperiments, 
                 reporting_interval=100):
        super(DefaultCallback, self).__init__(uncs, 
                                              outcomes, 
                                              nrOfExperiments, 
                                              reporting_interval)
        self.i = 0
        self.cases = None
        self.policies = None
        self.names = None   
        self.results = {}
        
        self.outcomes = [outcome.name for outcome in outcomes]

        #determine data types of uncertainties
        self.dtypes = []
        self.uncertainties = []
        
        for uncertainty in uncs:
            name = uncertainty.name
            self.uncertainties.append(name)
            dataType = float
            
            if isinstance(uncertainty, CategoricalUncertainty):
                dataType = object
            elif isinstance(uncertainty, ParameterUncertainty) and\
                          uncertainty.dist==INTEGER:
                dataType = int
            self.dtypes.append((name, dataType))
        self.dtypes.append(('model', object))
        self.dtypes.append(('policy', object))
        try:
            self.cases = np.empty((nrOfExperiments,), dtype=self.dtypes)
        except ValueError:
            names_seen = set()
            for entry in self.dtypes:
                if entry[0] in names_seen:
                    ema_logging.warning("uncertainty %s occurs more then once" %(entry[0]))
                else:
                    names_seen.add(entry[0])
            raise EMAError(str(ValueError))
        self.nrOfExperiments = nrOfExperiments
        

    def __store_case(self, case, model, policy):
        case = [case.get(key) for key in self.uncertainties]
        case.append(model)
        case.append(policy)
        case = tuple(case)
        self.cases[self.i-1] = case
            
    def __store_result(self, result):
        for outcome in self.outcomes:
            try:
                self.results[outcome][self.i-1, :] = result[outcome]
            except KeyError:

                try:
                    shapeResults = result[outcome].shape
                    if len(shapeResults) >0:
                        ncol = shapeResults[0] 
                    else:
                        ncol= 1
                except AttributeError:
                    #apparently the outcome is not an array but a scalar
                    ncol=1
            
                
                self.results[outcome] = np.empty((self.nrOfExperiments, ncol))
                self.results[outcome][self.i-1, :] = result[outcome]
        
    
    def __call__(self, case, policy, name, result ):
        '''
        Method responsible for storing results. This method calls 
        :meth:`super` first, thus utilizing the logging provided there
        
        :param case: the case to be stored
        :param policy: the name of the policy being used
        :param name: the name of the model being used
        :param result: the result dict. This implementation assumes that
                       the values in this dict are numpy array instances. Two
                       types of instances are excepted: single values and
                       1-D arrays. 
        :return: a tuple with the cases structured array and the dict of 
                 result arrays. 
        
        '''
        
        super(DefaultCallback, self).__call__(case, policy, name, result)
                   
        #store the case
        self.__store_case(case, name, policy.get('name'), )
        
        #store results
        self.__store_result(result)
        
        
    def get_results(self):
        return self.cases, self.results


def load_results(file_name, zipped=True):
    '''
    load the specified bz2 file. the file is assumed to be saves
    using save_results.
    
    :param file: the path of the file
    :param zipped: load the pickled data from a zip file if True
    :return: the unpickled results
    :raises: IOError if file not found
    
    '''
    results = None
    file_name = os.path.abspath(file_name)
    debug("loading "+file_name)
    try:
        if zipped:
            file_handle = bz2.BZ2File(file_name, 'rb')
        else:
            file_handle = open(file_name, 'rb')
        
        results = cPickle.load(file_handle)
    except IOError:
        warning(file_name + " not found")
        raise
    
    return results
    

def save_results(results, file_name, zipped=True):
    '''
    save the results to the specified bz2 file. To facilitate transfer
    across different machines. the files are saved in binary format
        
    see also: http://projects.scipy.org/numpy/ticket/1284

    :param results: the return of run_experiments
    :param file: the path of the file
    :param zipped: save the pickled data to a zip file if True
    :raises: IOError if file not found

    '''
    file_name = os.path.abspath(file_name)
    debug("saving results to: " + file_name)
    try:
        if zipped:
            file_name = bz2.BZ2File(file_name, 'wb')
        else:
            file_name = open(file_name, 'wb')

        
        cPickle.dump(results, file_name, protocol=2)
    except IOError:
        warning(os.path.abspath(file_name) + " not found")
        raise
        


def results_to_tab(results, file_name):
    '''
    writes old style results to tab seperated
    '''
    
    fields = results[0][0][0].keys()
    outcomes = results[0][1].keys()
    file_handle = open(file_name, 'w')
    [file_handle.write(field + "\t") for field in fields]
    file_handle.write("policy\tmodel\t")
    [file_handle.write(field + "\t") for field in outcomes]
    file_handle.write("\n")
    
    for result in results:
        experiment = result[0]
        case = experiment[0]
        policy = experiment[1]['name']
        model = experiment[2]
        outcome = result[1]
        
        
        for field in fields:
            file_handle.write(str(case[field])+"\t")
        file_handle.write(policy+"\t")
        file_handle.write(model+"\t")
        for field in outcomes:
            file_handle.write(str(outcome[field][-1])+"\t")
        
        file_handle.write("\n")


def transform_old_cPickle_to_new_cPickle(file_name):
    data = cPickle.load(open(file_name, 'r'))
    
    uncertainties = []
    dtypes= []
    for name in  data[0][0][0].keys():
        uncertainties.append(name)
        dataType = float
        dtypes.append((name, dataType))
    dtypes.append(('model', object))
    dtypes.append(('policy', object))
    
    #setup the empty data structures
    cases = np.zeros((len(data),), dtype=dtypes)
    results = {}
    for key in data[0][1].keys():
        results[key] = np.zeros((len(data), len(data[0][1].get(key))))
        
    for i, entry in enumerate(data):
        case = entry[0][0]
        policy = entry[0][1].get('name')
        model = entry[0][2]
        result = entry[1]
        
        #handle the case
        case = [case.get(key) for key in uncertainties]
        case.append(model)
        case.append(policy)
        cases[i] = tuple(case)
        
        #handle the result
        for key, value in result.items():
            results[key][i, :] = value
    
    results = cases, results
    return results


def experiments_to_cases(experiments):
    '''
    
    This function transform a structured experiments array into a list
    of case dicts. This can then for example be used as an argument for 
    running :meth:`~model.SimpleModelEnsemble.perform_experiments`.
    
    :param experiments: a structured array containing experiments
    :return: a list of case dicts.
    
    '''
    #get the names of the uncertainties
    uncertainties = [entry[0] for entry in experiments.dtype.descr]
    
    #remove policy and model, leaving only the case related uncertainties
    try:
        uncertainties.pop(uncertainties.index('policy'))
        uncertainties.pop(uncertainties.index('model'))
    except:
        pass
    
    #make list of of tuples of tuples
    cases = []
    for i in range(experiments.shape[0]):
        case = []
        for uncertainty in uncertainties:
            entry = (uncertainty, experiments[uncertainty][i])
            case.append(entry)
        cases.append(tuple(case))
    
    #remove duplicate cases, reason for using tuples before
    cases = set(cases)
    
    #cast back to list of dicts
    tempcases = []
    for case in cases:
        tempCase = {}
        for entry in case:
            tempCase[entry[0]] = entry[1]
        tempcases.append(tempCase)
    cases = tempcases
    
    return cases

def experiments_to_cases_prim(experiments, designs):
    '''
    
    This function transform a structured experiments array into a list
    of case dicts. This can then for example be used as an argument for 
    running :meth:`~model.SimpleModelEnsemble.perform_experiments`.
    
    :param experiments: a structured array containing experiments
    :return: a list of case dicts.
    
    '''
    #get the names of the uncertainties
    uncertainties = [entry[0] for entry in designs.dtype.descr]
    
    #remove policy and model, leaving only the case related uncertainties
    try:
        uncertainties.pop(uncertainties.index('policy'))
        uncertainties.pop(uncertainties.index('model'))
    except:
        pass
    
    #make list of of tuples of tuples
    cases = []
    for i in range(len(experiments)):
        case = []
        j = 0
        for uncertainty in uncertainties:
            entry = (uncertainty, experiments[i][j])
            j += 1
            case.append(entry)
        cases.append(tuple(case))
    
    #remove duplicate cases, reason for using tuples before
    cases = set(cases)
    
    #cast back to list of dicts
    tempcases = []
    for case in cases:
        tempCase = {}
        for entry in case:
            tempCase[entry[0]] = entry[1]
        tempcases.append(tempCase)
    cases = tempcases
    
    return cases

def merge_results(results1, results2, downsample=None):
    '''
    convenience function for merging the return from 
    :meth:`~modelEnsemble.ModelEnsemble.perform_experiments`.
    
    The function merges results2 with results1. For the experiments,
    it generates an empty array equal to the size of the sum of the 
    experiments. As dtype is uses the dtype from the experiments in results1.
    The function assumes that the ordering of dtypes and names is identical in
    both results.  
    
    A typical use case for this function is in combination with 
    :func:`~util.experiments_to_cases`. Using :func:`~util.experiments_to_cases`
    one extracts the cases from a first set of experiments. One then
    performs these cases on a different model or policy, and then one wants to
    merge these new results with the old result for further analysis.  
    
    :param results1: first results to be merged
    :param results2: second results to be merged
    :param downsample: should be an integer, will be used in slicing the results
                       in order to avoid memory problems. 
    :return: the merged results
    
    
    '''

    #start of merging
    old_exp, old_res = results1
    new_exp, new_res = results2
    
    #merge experiments
    dtypes = old_exp.dtype
    
    merged_exp = np.empty((old_exp.shape[0]+new_exp.shape[0],),dtype= dtypes)
    merged_exp[0:old_exp.shape[0]] = old_exp
    merged_exp[old_exp.shape[0]::] = new_exp
    
    #only merge the results that are in both
    keys = old_res.keys()
    [keys.append(key) for key in new_res.keys()]
    keys = set(keys)
    info("intersection of keys: %s" % keys)
    
    #merging results
    merged_res = {}
    for key in keys:
        info("merge "+key)
        
        old_value = old_res.get(key)
        new_value = new_res.get(key)
        
        i = old_value.shape[0]+new_value.shape[0]
        j = old_value.shape[1]
        slice_value = 1
        if downsample:
            j = int(math.ceil(j/downsample))
            slice_value = downsample
            
        merged_value = np.empty((i,j))
        debug("merged shape: %s" % merged_value.shape)
        
        merged_value[0:old_value.shape[0], :] = old_value[:, ::slice_value]
        merged_value[old_value.shape[0]::, :] = new_value[:, ::slice_value]

        merged_res[key] = merged_value
    
    mr = (merged_exp, merged_res)
    return mr  

def load_optimization_results(file_name, weights, zipped=True):
    '''
    load the specified bz2 file. the file is assumed to be saves
    using save_results.
    
    :param file: the path of the file
    :param zipped: load the pickled data from a zip file if True
    :return: the unpickled results
    :raises: IOError if file not found
    :raises: EMAError if weights are not correct
    
    '''
    creator.create("Fitness", base.Fitness, weights=weights)
    creator.create("Individual", dict, 
                   fitness=creator.Fitness) #@UndefinedVariable
    
    file_name = os.path.abspath(file_name)
    debug("loading "+file_name)
    try:
        if zipped:
            file_name = bz2.BZ2File(file_name, 'rb')
        else:
            file_name = open(file_name, 'rb')
        
        results = cPickle.load(file_name)
        
        if results[0].weights != weights:
            raise EMAError("weights are %s, should be %s" % (weights, results[0].weights))
    except IOError:
        warning(file_name + " not found")
        raise
    
    return results

def save_optimization_results(results, file_name, zipped=True):
    '''
    save the results to the specified bz2 file. To facilitate transfer
    across different machines. the files are saved in binary format
        
    see also: http://projects.scipy.org/numpy/ticket/1284

    :param results: the return of run_experiments
    :param file: the path of the file
    :param zipped: save the pickled data to a zip file if True
    :raises: IOError if file not found

    '''

    save_results(results, file_name, zipped=zipped) 
