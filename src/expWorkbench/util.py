'''
Created on 13 jan. 2011

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

This module provides various convenience functions and classes.

'''
from __future__ import division

import cPickle
import os
import bz2
import math
import StringIO

import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
from matplotlib.mlab import rec2csv, csv2rec

from deap import creator, base

from ema_logging import info, debug, warning
from expWorkbench import EMAError
import tarfile

__all__ = ['load_results',
           'save_results',
           'save_optimization_results',
           'load_optimization_results',
           'experiments_to_cases',
           ]


def load_results(file_name):
    '''
    load the specified bz2 file. the file is assumed to be saves
    using save_results.
    
    :param file_name: the path of the file
    :raises: IOError if file not found


    '''
    
    outcomes = {}
    with tarfile.open(file_name, 'r') as z:
        # load experiments
        experiments = z.extractfile('experiments.csv')
        experiments = csv2rec(experiments)

        # load experiment metadata
        metadata = z.extractfile('experiments metadata.csv').readlines()
        metadata = [entry.strip() for entry in metadata]
        metadata = [tuple(entry.split(",")) for entry in metadata]
        metadata = np.dtype(metadata)

        # cast experiments to dtype and name specified in metadata        
        temp_experiments = np.zeros((experiments.shape[0],), dtype=metadata)
        for i, entry in enumerate(experiments.dtype.descr):
            dtype = metadata[i]
            name = metadata.descr[i][0]
            temp_experiments[name][:] = experiments[entry[0]].astype(dtype)
        experiments = temp_experiments
        
        # load outcome metadata
        metadata = z.extractfile('outcomes metadata.csv').readlines()
        metadata = [entry.strip() for entry in metadata]
        metadata = [tuple(entry.split(",")) for entry in metadata]
        metadata = {entry[0]: entry[1:] for entry in metadata}

        # load outcomes
        for outcome, shape in metadata.iteritems():
            shape = list(shape)
            shape[0] = shape[0][1:]
            shape[-1] = shape[-1][0:-1]
            shape = tuple([int(entry) for entry in shape])
            
            if len(shape)>2:
                nr_files = shape[-1]
                
                data = np.empty(shape)
                for i in range(nr_files):
                    values = z.extractfile("{}_{}.csv".format(outcome, i))
                    values = read_csv(values, index_col=False, header=None).values
                    data[:,:,i] = values

            else:
                data = z.extractfile("{}.csv".format(outcome))
                data = read_csv(data, index_col=False, header=None).values
                
            outcomes[outcome] = data
            
    info("results loaded succesfully from {}".format(file_name))
    return experiments, outcomes


def save_results(results, file_name):
    '''
    save the results to the specified tar.gz file. The results are stored as 
    csv files. There is an experiments.csv, and a csv for each outcome. In 
    addition, there is a metadata csv which contains the datatype information
    for each of the columns in the experiments array.

    :param results: the return of run_experiments
    :param file_name: the path of the file
    :raises: IOError if file not found

    '''

    def add_file(tararchive, string_to_add, filename):
        tarinfo = tarfile.TarInfo(filename)
        tarinfo.size = len(string_to_add)
        
        z.addfile(tarinfo, StringIO.StringIO(string_to_add))  
    
    def save_numpy_array(fh, data):
        data = pd.DataFrame(data)
        data.to_csv(fh, header=False, index=False)
        
    experiments, outcomes = results
    with tarfile.open(file_name, 'w:gz') as z:
        # write the experiments to the zipfile
        experiments_file = StringIO.StringIO()
        rec2csv(experiments, experiments_file, withheader=True)
        add_file(z, experiments_file.getvalue(), 'experiments.csv')
        
        # write experiment metadata
        dtype = experiments.dtype.descr
        dtype = ["{},{}".format(*entry) for entry in dtype]
        dtype = "\n".join(dtype)
        add_file(z, dtype, 'experiments metadata.csv')
        
        # write outcome metadata
        outcome_names = outcomes.keys()
        outcome_meta = ["{},{}".format(outcome, outcomes[outcome].shape) 
                        for outcome in outcome_names]
        outcome_meta = "\n".join(outcome_meta)
        add_file(z, outcome_meta, "outcomes metadata.csv")
        
        
        # outcomes
        for key, value in outcomes.iteritems():
            fh = StringIO.StringIO()
            
            nr_dim = len(value.shape)
            if nr_dim==3:
                for i in range(value.shape[2]):
                    data = value[:,:,i]
                    save_numpy_array(fh, data)
                    fh = fh.getvalue()
                    fn = '{}_{}.csv'.format(key, i)
                    add_file(z, fh, fn)
                    fh = StringIO.StringIO()
            else:
                save_numpy_array(fh, value)
                fh = fh.getvalue()
                add_file(z, fh, '{}.csv'.format(key))
  
    info("results saved successfully to {}".format(file_name))
    

def oldcsv_load_results(file_name):
    '''
    load the specified bz2 file. the file is assumed to be saves
    using save_results.
    
    :param file_name: the path of the file
    :raises: IOError if file not found


    '''
    
    outcomes = {}
    with tarfile.open(file_name, 'r') as z:
        # load experiments
        experiments = z.extractfile('experiments.csv')
        experiments = csv2rec(experiments)

        # load metadata
        metadata = z.extractfile('experiments metadata.csv').readlines()
        metadata = [entry.strip() for entry in metadata]
        metadata = [tuple(entry.split(",")) for entry in metadata]
        metadata = np.dtype(metadata)

        # cast experiments to dtype and name specified in metadata        
        temp_experiments = np.zeros((experiments.shape[0],), dtype=metadata)
        for i, entry in enumerate(experiments.dtype.descr):
            dtype = metadata[i]
            name = metadata.descr[i][0]
            temp_experiments[name][:] = experiments[entry[0]].astype(dtype)
        experiments = temp_experiments

        # load outcomes
        fhs = z.getnames()
        fhs.remove('experiments.csv')
        fhs.remove('experiments metadata.csv')
        for fh in fhs:
            root = os.path.splitext(fh)[0]
            data = z.extractfile(fh)
            first_line = data.readline()
            shape = first_line.split(":")[1].strip()[1:-1]
            shape = shape.split((','))
            shape = tuple([int(entry) for entry in shape if len(entry)>0])
            data = np.loadtxt(data, delimiter=',')
            data = data.reshape(shape)
            
            outcomes[root] = data
            
    info("results loaded succesfully from {}".format(file_name))
    return experiments, outcomes

  
def pickled_load_results(file_name, zipped=True):
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
    

def pickled_save_results(results, file_name, zipped=True):
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
