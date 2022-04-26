"""

This module provides various convenience functions and classes.

"""

import configparser
import json
import os
import tarfile
from io import BytesIO

import numpy as np
import pandas as pd

from . import EMAError, get_module_logger


# Created on 13 jan. 2011
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ["load_results", "save_results", "merge_results", "process_replications"]
_logger = get_module_logger(__name__)


def load_results(file_name):
    """
    load the specified bz2 file. the file is assumed to be saves
    using save_results.

    Parameters
    ----------
    file_name : str
                the path to the file

    Raises
    ------
    IOError if file not found

    """
    from ..em_framework.outcomes import AbstractOutcome, register

    file_name = os.path.abspath(file_name)

    with tarfile.open(file_name, "r:gz", encoding="UTF8") as archive:
        try:
            f = archive.extractfile("metadata.json")
        except KeyError:
            # old style data file
            results = load_results_old(archive)
            _logger.info(f"results loaded successfully from {file_name}")
            return results

        metadata = json.loads(f.read().decode())

        # load experiments
        f = archive.extractfile("experiments.csv")
        experiments = pd.read_csv(f)

        for name, dtype in metadata["experiments"].items():
            try:
                dtype = np.dtype(dtype)
            except TypeError:
                dtype = pd.api.types.pandas_dtype(dtype)

            if pd.api.types.is_object_dtype(dtype):
                experiments[name] = experiments[name].astype("category")

        # load outcomes
        outcomes = {}
        known_outcome_classes = {
            entry.__name__: entry for entry in AbstractOutcome.get_subclasses()
        }
        for (outcome_type, name, filename) in metadata["outcomes"]:
            outcome = known_outcome_classes[outcome_type](name)

            values = register.deserialize(name, filename, archive)
            outcomes[name] = values

    _logger.info(f"results loaded successfully from {file_name}")
    return experiments, outcomes


def load_results_old(archive):
    """
    load the specified bz2 file. the file is assumed to be saves
    using save_results.

    Parameters
    ----------
    file_name : TarFile

    Raises
    ------
    IOError if file not found

    """
    from ..em_framework.outcomes import ScalarOutcome, ArrayOutcome, register

    outcomes = {}

    # load x
    experiments = archive.extractfile("experiments.csv")
    if not (hasattr(experiments, "read")):
        raise EMAError(repr(experiments))

    experiments = pd.read_csv(experiments)

    # load experiment metadata
    metadata = archive.extractfile("experiments metadata.csv").readlines()

    for entry in metadata:
        entry = entry.decode("UTF-8")
        entry = entry.strip()
        entry = entry.split(",")
        name, dtype = (str(item) for item in entry)

        try:
            dtype = np.dtype(dtype)
        except TypeError:
            dtype = pd.api.types.pandas_dtype(dtype)

        if pd.api.types.is_object_dtype(dtype):
            experiments[name] = experiments[name].astype("category")

    # load outcome metadata
    metadata = archive.extractfile("outcomes metadata.csv").readlines()
    metadata = [entry.decode("UTF-8") for entry in metadata]
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
                values = archive.extractfile(f"{outcome}_{i}.csv")
                values = pd.read_csv(values, index_col=False, header=None).values
                data[:, :, i] = values

        else:
            data = archive.extractfile(f"{outcome}.csv")
            data = pd.read_csv(data, index_col=False, header=None).values
            data = np.reshape(data, shape)

        outcomes[outcome] = data

    # reformat outcomes from generic dict to new style OutcomesDict
    outcomes_new = {}
    for k, v in outcomes.items():
        if v.ndim == 1:
            outcome = ScalarOutcome(k)
        else:
            outcome = ArrayOutcome(k)

        outcomes_new[outcome.name] = v

    return experiments, outcomes_new


def save_results(results, file_name):
    """
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

    """
    from ..em_framework.outcomes import register

    VERSION = 0.1
    file_name = os.path.abspath(file_name)

    def add_file(tararchive, stream, filename):
        stream.seek(0)
        tarinfo = tarfile.TarInfo(filename)
        tarinfo.size = len(stream.getbuffer())
        tararchive.addfile(tarinfo, stream)

    experiments, outcomes = results
    with tarfile.open(file_name, "w:gz") as z:
        # store experiments
        stream = BytesIO()
        stream.write(
            experiments.to_csv(header=True, encoding="UTF-8", index=False).encode()
        )
        add_file(z, stream, "experiments.csv")

        # store outcomes
        outcomes_metadata = []
        for key, value in outcomes.items():
            klass = register.outcomes[key]
            stream, filename = register.serialize(key, value)
            add_file(z, stream, filename)
            outcomes_metadata.append((klass.__name__, key, filename))

        # store metadata
        metadata = {
            "version": VERSION,
            "outcomes": outcomes_metadata,
            "experiments": {k: v.name for k, v in experiments.dtypes.to_dict().items()},
        }

        stream = BytesIO()
        stream.write(json.dumps(metadata).encode())
        add_file(z, stream, "metadata.json")

    _logger.info(f"results saved successfully to {file_name}")


def merge_results(results1, results2):
    """
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


    """

    # start of merging
    exp1, res1 = results1
    exp2, res2 = results2

    # merge x
    merged_exp = pd.concat([exp1, exp2], axis=0)
    merged_exp.reset_index(drop=True, inplace=True)

    # only merge the results that are in both
    keys = set(res1.keys()).intersection(set(res2.keys()))
    _logger.info(f"intersection of keys: {keys}")

    # merging results
    merged_res = {}
    for key in keys:
        _logger.info(f"merge {key}")

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
            _logger.info(f"config loaded from {parsed[0]}")
        else:
            _logger.info("no config file found")

        home_dir = config.get("ema_project_home", "home_dir")
        return home_dir
    except BaseException:
        return os.getcwd()


def process_replications(data, aggregation_func=np.mean):
    """
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


    """

    if isinstance(data, dict):
        # replications are the second dimension of the outcome arrays
        outcomes_processed = {
            key: aggregation_func(data[key], axis=1) for key in data.keys()
        }
        return outcomes_processed
    elif (
        isinstance(data, tuple)
        and isinstance(data[0], pd.DataFrame)
        and isinstance(data[1], dict)
    ):
        experiments, outcomes = data  # split results
        outcomes_processed = {
            key: aggregation_func(outcomes[key], axis=1) for key in outcomes.keys()
        }
        results_processed = (experiments.copy(deep=True), outcomes_processed)
        return results_processed

    else:
        raise EMAError(
            f"data should be a dict or tuple, but is a {type(data)}".format()
        )
