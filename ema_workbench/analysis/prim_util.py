'''Collection of utility functions used by PRIM module

'''

from enum import Enum
import math

import numpy as np

from . import scenario_discovery_util as sdutil

from ..util.ema_logging import (get_module_logger)


_logger = get_module_logger(__name__)


# Created on 6 Feb 2019
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


class PrimException(Exception):
    '''Base exception class for prim related exceptions'''
    pass


class PRIMObjectiveFunctions(Enum):
    LENIENT2 = 'lenient2'
    LENIENT1 = 'lenient1'
    ORIGINAL = 'original'


def get_quantile(data, quantile):
    '''
    quantile calculation modeled on the implementation used in sdtoolkit

    Parameters
    ----------
    data : nd array like
           dataset for which quantile is needed
    quantile : float
               the desired quantile

    '''
    assert quantile > 0
    assert quantile < 1

    data = np.sort(data)

    i = (len(data) - 1) * quantile
    index_lower = int(math.floor(i))
    index_higher = int(math.ceil(i))

    value = 0

    if quantile > 0.5:
        # upper
        while (data[index_lower] == data[index_higher]) & (index_lower > 0):
            index_lower -= 1
        value = (data[index_lower] + data[index_higher]) / 2
    else:
        # lower
        while (data[index_lower] == data[index_higher]) & \
              (index_higher < len(data) - 1):
            index_higher += 1
        value = (data[index_lower] + data[index_higher]) / 2

    return value


class CurEntry(object):
    '''a descriptor for the current entry on the peeling and pasting
    trajectory'''

    def __init__(self, name):
        self.name = name

    def __get__(self, instance, _):
        return instance.peeling_trajectory[self.name][instance._cur_box]

    def __set__(self, instance, value):
        raise PrimException("this property cannot be assigned to")


def calculate_qp(data, x, y, Hbox, Tbox, box_lim, initial_boxlim):
    '''Helper function for calculating quasi p-values'''
    if data.size == 0:
        return [-1, -1]

    u = data.name
    dtype = data.dtype

    unlimited = initial_boxlim[u]

    if np.issubdtype(dtype, np.number):
        qp_values = []
        for direction, (limit, unlimit) in enumerate(zip(data,
                                                         unlimited)):
            if unlimit != limit:
                temp_box = box_lim.copy()
                temp_box.at[direction, u] = unlimit
                qp = sdutil._calculate_quasip(x, y, temp_box,
                                              Hbox, Tbox)
            else:
                qp = -1
            qp_values.append(qp)
    else:
        temp_box = box_lim.copy()
        temp_box.loc[:, u] = unlimited
        qp = sdutil._calculate_quasip(x, y, temp_box,
                                      Hbox, Tbox)
        qp_values = [qp, -1]

    return qp_values


def rotate_subset(experiments, y):
    '''
    rotate a subset

    Parameters
    ----------
    experiments_subset : DataFrame
    y : ndarray

    Returns
    -------
    rotation_matrix
        DataFrame
    rotated_experiments
        DataFrame

    '''
    mean = np.mean(experiments, axis=0)
    std = np.std(experiments, axis=0)
    std[std == 0] = 1  # in order to avoid a devision by zero
    experiments = (experiments - mean) / std

    # normalize the data
    subset_experiments = experiments.loc[y, :]

    # determine the rotation
    rotation_matrix = determine_rotation(subset_experiments)

    # apply the rotation
    rotated_experiments = np.dot(experiments, rotation_matrix)
    return rotation_matrix, rotated_experiments


def determine_rotation(experiments):
    '''
    Determine the rotation for the specified experiments


    Parameters
    ----------
    experiments : pd.DataFrame

    Returns
    -------
    ndarray

    '''
    covariance = np.cov(experiments.T)

    eigen_vals, eigen_vectors = np.linalg.eig(covariance)

    indices = np.argsort(eigen_vals)
    indices = indices[::-1]
    eigen_vectors = eigen_vectors[:, indices]
    eigen_vals = eigen_vals[indices]

    # make the eigen vectors unit length
    for i in range(eigen_vectors.shape[1]):
        eigen_vectors[:, i] / \
            np.linalg.norm(eigen_vectors[:, i]) * np.sqrt(eigen_vals[i])

    return eigen_vectors


def determine_dimres(box, issignificant=True,
                     significance_threshold=0.1):
    def is_significant(v):
        for entry in v:
            if (entry >= 0) & (entry <= significance_threshold):
                return True

    all_dims = set()
    for qp in box.qp:
        if issignificant:
            dims = [k for k, v in qp.items() if is_significant(v)]
        else:
            dims = qp.keys()
        all_dims.update(dims)
    return all_dims


def box_to_tuple(box):
    names = box.columns
    sorted(names)
    tupled_box = []
    for name in names:
        values = box[name]
        formatted = []
        for entry in values:
            if isinstance(entry, set):
                entry = tuple(entry)
            formatted.append(entry)
        tupled_box += tuple(formatted)
    tupled_box = tuple(tupled_box)
    return tupled_box


class NotSeen(object):
    def __init__(self):
        self.seen = set()

    def __call__(self, box):
        tupled = box_to_tuple(box)
        if tupled in self.seen:
            return False
        else:
            self.seen.add(tupled)
            return True


def is_significant(box, i, alpha=0.05):
    qp = box.qp[i]
    return not any([value > alpha for values in qp.values()
                    for value in values])


def is_pareto_efficient(data):
    fronts = calc_fronts(data)
    return fronts == 0


def calc_fronts(M):
    # taken from
    # https://stackoverflow.com/questions/41740596/pareto-frontier-indices-using-numpy
    i_dominates_j = np.all(M[:, None] >= M, axis=-
                           1) & np.any(M[:, None] > M, axis=-1)
    remaining = np.arange(len(M))
    fronts = np.empty(len(M), int)
    frontier_index = 0
    while remaining.size > 0:
        dominated = np.any(
            i_dominates_j[remaining[:, None], remaining], axis=0)
        fronts[remaining[~dominated]] = frontier_index

        remaining = remaining[dominated]
        frontier_index += 1
    return fronts
