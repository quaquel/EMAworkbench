'''

This module provides various convenience functions and classes.

'''

import configparser
from io import BytesIO, StringIO
import json
import os
import tarfile

import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv

from . import EMAError, get_module_logger



# Created on 13 jan. 2011
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['load_results',
           'save_results',
           'merge_results',
		   'process_replications'
           ]
_logger = get_module_logger(__name__)


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



def load_results_old(file_name):
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

        experiments = pd.read_csv(experiments)

        # load experiment metadata
        metadata = z.extractfile('experiments metadata.csv').readlines()

        for entry in metadata:
            entry = entry.decode('UTF-8')
            entry = entry.strip()
            entry = entry.split(",")
            name, dtype = [str(item) for item in entry]
            
            
            
            try:
                dtype = np.dtype(dtype)
            except TypeError:
                dtype = pd.api.types.pandas_dtype(dtype)
            
            if pd.api.types.is_object_dtype(dtype):
                experiments[name] = experiments[name].astype('category')

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
                        temp_shape.append(int(entry[0:-1]))
            shape = tuple(temp_shape)

            if len(shape) > 2:
                nr_files = shape[-1]

                data = np.empty(shape)
                for i in range(nr_files):
                    values = z.extractfile("{}_{}.csv".format(outcome, i))
                    values = read_csv(values, index_col=False,
                                      header=None).values
                    data[:, :, i] = values

            else:
                data = z.extractfile("{}.csv".format(outcome))
                data = read_csv(data, index_col=False, header=None).values
                data = np.reshape(data, shape)

            outcomes[outcome] = data

    _logger.info("results loaded succesfully from {}".format(file_name))
    return experiments, outcomes


def save_results(results, file_name):
    '''
    save the results to the specified tar.gz file. 
    
    The way in which results are stored depends. Experiments are saved
    as csv. Outcomes depend on the outcome type. Scalar, and <3D arrays are
    saved as csv files. Higher dimensional arrays are stored as .npy files.
    
    Parameters
    ----------
    results : tuple
              the return of perform_experiments
    file_name : str
                the path of the file

    Raises
    ------
    IOError if file not found

    '''
    VERSION = 0.1
    file_name = os.path.abspath(file_name)

    def add_file(tararchive, stream, filename):
        string_to_add = stream.getvalue()
        
        tarinfo = tarfile.TarInfo(filename)
        tarinfo.size = len(string_to_add)

        fh = BytesIO(string_to_add.encode('UTF-8'))

        z.addfile(tarinfo, fh)

    experiments, outcomes = results
    with tarfile.open(file_name, 'w:gz') as z:
        # store experiments
        stream = StringIO()
        experiments.to_csv(stream, header=True,
                           encoding='UTF-8', index=False)
        add_file(z, stream, 'experiments.csv')

        # store outcomes
        outcomes_metadata = []
        for key, value in outcomes.items():
            stream, filename = key.to_disk(value)
            add_file(z, stream, filename)
            outcomes_metadata.append((key.__class__.__name__, key.name,
                                      filename))

        # store metadata
        metadata = {'version': VERSION,
                    'outcomes': outcomes_metadata,
                    'experiments': {k:v.name for k, v in
                                    experiments.dtypes.to_dict().items()}}

        stream = StringIO()
        json.dump(metadata, stream)
        add_file(z, stream, "metadata.json")

    _logger.info(f"results saved successfully to {file_name}")



def merge_results(results1, results2):
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

    Returns
    -------
    the merged results


    '''

    # start of merging
    exp1, res1 = results1
    exp2, res2 = results2

    # merge x
    merged_exp = pd.concat([exp1, exp2], axis=0)
    merged_exp.reset_index(drop=True, inplace=True)

    # only merge the results that are in both
    keys = set(res1.keys()).intersection(set(res2.keys()))
    _logger.info("intersection of keys: %s" % keys)

    # merging results
    merged_res = {}
    for key in keys:
        _logger.info("merge " + key)

        value1 = res1.get(key)
        value2 = res2.get(key)
        merged_value = np.concatenate([value1, value2])
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
            _logger.info('config loaded from {}'.format(parsed[0]))
        else:
            _logger.info('no config file found')

        home_dir = config.get('ema_project_home', 'home_dir')
        return home_dir
    except BaseException:
        return os.getcwd()


def process_replications(data, aggregation_func = np.mean):
    '''
    Convenience function for processing the replications of a stochastic
    model's outcomes.

    The default behavior is to take the mean of the replications. This reduces
    the dimensionality of the outcomes from
    (experiments * replications * outcome_shape) to
    (experiments * outcome_shape), where outcome_shape is 0-d for scalars,
    1-d for time series, and 2-d for arrays.

    The function can take either the outcomes (dictionary: keys are outcomes
    of interest, values are arrays of data) or the results (tuple: experiments
    as DataFrame, outcomes as dictionary) of a set of simulation experiments.

    Parameters
    ----------
    data : dict, tuple
        outcomes or results of a set of experiments
    aggregation_func : callabale, optional
        aggregation function to be applied, defaults to np.mean.

    Returns
    -------
    dict, tuple 
    
    
    '''

    if isinstance(data, dict):
        #replications are the second dimension of the outcome arrays
        outcomes_processed = {key:aggregation_func(data[key],axis=1) for key
                              in data.keys()}
        return outcomes_processed
    elif (isinstance(data, tuple) and
            isinstance(data[0], pd.DataFrame) and
            isinstance(data[1], dict)):
        experiments, outcomes = data #split results
        outcomes_processed = {key:aggregation_func(outcomes[key], axis=1) for
                              key in outcomes.keys()}
        results_processed = (experiments.copy(deep=True), outcomes_processed)
        return results_processed

    else:
        raise EMAError(
            f"data should be a dict or tuple, but is a {type(data)}".format())
