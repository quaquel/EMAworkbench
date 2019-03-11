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
import os
import sys
import tarfile

import numpy as np

import pandas as pd
from pandas.io.parsers import read_csv

from . import EMAError, get_module_logger
import ema_workbench

PY3 = sys.version_info[0] == 3
if PY3:
    WriterFile = StringIO
else:
    WriterFile = BytesIO

# Created on 13 jan. 2011
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['load_results',
           'save_results',
           'experiments_to_scenarios',
           'merge_results'
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
            if np.dtype(dtype) == object:
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
    save the results to the specified tar.gz file. The results are
    stored as csv files. There is an x.csv, and a csv for each
    outcome. In addition, there is a metadata csv which contains
    the datatype information for each of the columns in the x array.

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

        experiments.to_csv(experiments_file, header=True,
                           encoding='UTF-8', index=False)

        add_file(z, experiments_file.getvalue(), 'experiments.csv')

        # write experiment metadata
        metadatafile = WriterFile()
        experiments.dtypes.to_csv(metadatafile, header=False)
        add_file(z, metadatafile.getvalue(), 'experiments metadata.csv')

        # write outcome metadata
        outcome_names = outcomes.keys()
        outcome_meta = ["{},{}".format(outcome, outcomes[outcome].shape)
                        for outcome in outcome_names]
        outcome_meta = "\n".join(outcome_meta)
        add_file(z, outcome_meta, "outcomes metadata.csv")

        # outcomes
        for key, value in outcomes.items():
            fh = WriterFile()

            if value.ndim == 3:
                for i in range(value.shape[2]):
                    data = value[:, :, i]
                    save_numpy_array(fh, data)
                    fh = fh.getvalue()
                    fn = '{}_{}.csv'.format(key, i)
                    add_file(z, fh, fn)
                    fh = WriterFile()
            else:
                save_numpy_array(fh, value)
                fh = fh.getvalue()
                add_file(z, fh, '{}.csv'.format(key))

    _logger.info("results saved successfully to {}".format(file_name))


def experiments_to_scenarios(experiments, model=None):
    '''

    This function transform a structured experiments array into a list
    of Scenarios.

    If model is provided, the uncertainties of the model are used.
    Otherwise, it is assumed that all non-default columns are
    uncertainties.

    Parameters
    ----------
    experiments : numpy structured array
                  a structured array containing experiments
    model : ModelInstance, optional

    Returns
    -------
    a list of Scenarios

    '''
    # get the names of the uncertainties
    if model is None:
        uncertainties = [entry[0] for entry in experiments.dtype.descr]

        # remove policy and model, leaving only the case related uncertainties
        try:
            uncertainties.pop(uncertainties.index('policy'))
            uncertainties.pop(uncertainties.index('model'))
            uncertainties.pop(uncertainties.index('scenario_id'))
        except BaseException:
            pass
    else:
        uncertainties = [u.name for u in model.uncertainties]

    # make list of of tuples of tuples
    cases = []
    cache = set()
    for i in range(experiments.shape[0]):
        case = {}
        case_tuple = []
        for uncertainty in uncertainties:
            entry = experiments[uncertainty][i]
            case[uncertainty] = entry
            case_tuple.append(entry)

        case_tuple = tuple(case_tuple)
        if case_tuple not in cache:
            cases.append(case)
            cache.add((case_tuple))

    scenarios = [ema_workbench.em_framework.parameters.Scenario(
        **entry) for entry in cases]

    return scenarios


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
