'''

This module provides various convenience functions and classes.

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

# import cPickle
from io import BytesIO, StringIO
import math
import os
import sys
import tarfile

from matplotlib.mlab import rec2csv
import numpy as np
from numpy.lib import recfunctions

import pandas as pd
from pandas.io.parsers import read_csv

from .ema_logging import info, debug, warning
from .ema_exceptions import EMAError

PY3 = sys.version_info[0] == 3
if PY3:
    WriterFile =  StringIO
else:
    WriterFile =  BytesIO

# Created on 13 jan. 2011
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['load_results',
           'save_results',
           'experiments_to_cases',
           'merge_results'
           ]


def load_results(file_name):
    '''
    load the specified bz2 file. the file is assumed to be saves
    using save_results.
    
    Parameters
    ----------    
    file_name : str
                the path to the file
                
    Raises
    ------
    IOError if file not found

    '''
    file_name = os.path.abspath(file_name)
    outcomes = {}
    with tarfile.open(file_name, 'r:gz', encoding="UTF8") as z:
        # load x
        experiments = z.extractfile('experiments.csv')
        if not (hasattr(experiments, 'read')):
            raise EMAError(repr(experiments))
        
        df = pd.read_csv(experiments)
        experiments = df.to_records(index=False)
#         experiments = recfunctions.drop_fields(experiments, ['index'])

        # load experiment metadata
        metadata = z.extractfile('experiments metadata.csv').readlines()
        
        metadata_temp = []
        for entry in metadata:
            entry = entry.decode('UTF-8')
            entry = entry.strip()
            entry = entry.split(",")
            entry = [str(item) for item in entry]
            entry = tuple(entry)
            metadata_temp.append(entry)
        metadata = metadata_temp    
        
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
        metadata = [entry.decode('UTF-8') for entry in metadata]
        metadata = [entry.strip() for entry in metadata]
        metadata = [tuple(entry.split(",")) for entry in metadata]
        metadata = {entry[0]: entry[1:] for entry in metadata}

        # load outcomes
        for outcome, shape in metadata.items():
            shape = list(shape)
            shape[0] = shape[0][1:]
            shape[-1] = shape[-1][0:-1]
            
            temp_shape = []
            for entry in shape:
                if entry:
                    try:
                        temp_shape.append(int(entry))
                    except ValueError:
                        try:
                            temp_shape.append(int(long(entry))) #@UndefinedVariable
                        except NameError: # we are on python3
                            temp_shape.append(int(entry[0:-1]))
            shape = tuple(temp_shape)
            
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
                data = np.reshape(data, shape)
                
            outcomes[outcome] = data
            
    info("results loaded succesfully from {}".format(file_name))
    return experiments, outcomes


def save_results(results, file_name):
    '''
    save the results to the specified tar.gz file. The results are stored as 
    csv files. There is an x.csv, and a csv for each outcome. In 
    addition, there is a metadata csv which contains the datatype information
    for each of the columns in the x array.

    Parameters
    ----------    
    results : tuple
              the return of run_experiments
    file_name : str
                the path of the file
    
    Raises
    ------
    IOError if file not found

    '''
    file_name = os.path.abspath(file_name)

    def add_file(tararchive, string_to_add, filename):
        tarinfo = tarfile.TarInfo(filename)
        tarinfo.size = len(string_to_add)
        
        fh = BytesIO(string_to_add.encode('UTF-8'))
        
        z.addfile(tarinfo, fh)  
    
    def save_numpy_array(fh, data):
        data = pd.DataFrame(data)
        data.to_csv(fh, header=False, index=False, encoding='UTF-8')
        
    experiments, outcomes = results
    with tarfile.open(file_name, 'w:gz') as z:
        # write the x to the zipfile
        experiments_file = WriterFile()
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
        for key, value in outcomes.items():
            fh = WriterFile()
            
            nr_dim = len(value.shape)
            if nr_dim==3:
                for i in range(value.shape[2]):
                    data = value[:,:,i]
                    save_numpy_array(fh, data)
                    fh = fh.getvalue()
                    fn = '{}_{}.csv'.format(key, i)
                    add_file(z, fh, fn)
                    fh = WriterFile()
            else:
                save_numpy_array(fh, value)
                fh = fh.getvalue()
                add_file(z, fh, '{}.csv'.format(key))
  
    info("results saved successfully to {}".format(file_name))
    

def experiments_to_cases(experiments):
    '''
    
    This function transform a structured x array into a list
    of case dicts. This can then for example be used as an argument for 
    running :meth:`~em_framework.model_ensemble.ModelEnsemble.perform_experiments`.
    
    Parameters
    ----------    
    experiments : numpy structured array
                  a structured array containing experiments
    
    Returns
    -------
    a list of case dicts.
    
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
    cache = set()
    for i in range(experiments.shape[0]):
        case = {}
        case_tuple = []
        for uncertainty in uncertainties:
            entry =  experiments[uncertainty][i]
            case[uncertainty] = entry
            case_tuple.append(entry)
        
        case_tuple = tuple(case_tuple)
        if case_tuple not in cache:
            cases.append(case)
            cache.add((case_tuple))
    
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
    
    Parameters
    ----------
    results1 : tuple
               first results to be merged
    results2 : tuple
               second results to be merged
    downsample : int 
                 should be an integer, will be used in slicing the results
                 in order to avoid memory problems. 

    Returns
    -------
    the merged results
    
    
    '''

    #start of merging
    old_exp, old_res = results1
    new_exp, new_res = results2
    
    #merge x
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


def get_ema_project_home_dir():
    try:
        config_file_name = "expworkbench.cfg"
        directory = os.path.dirname(__file__)
        fn = os.path.join(directory, config_file_name)

        config = configparser.ConfigParser()
        parsed = config.read(fn)

        if parsed:
            info('config loaded from {}'.format(parsed[0]))
        else:
            info('no config file found')


        home_dir = config.get('ema_project_home', 'home_dir')
        return home_dir
    except:
        return os.getcwd()
