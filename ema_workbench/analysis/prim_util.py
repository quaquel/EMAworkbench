"""Collection of utility functions used by PRIM module."""

import math
from enum import Enum

import numpy as np

from ..util.ema_logging import get_module_logger
from . import scenario_discovery_util as sdutil

_logger = get_module_logger(__name__)


# Created on 6 Feb 2019
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


class PrimException(Exception):  # noqa: N818
    """Base exception class for prim related exceptions."""


class PRIMObjectiveFunctions(Enum):
    """Enum for the prim objectives functions."""

    LENIENT2 = "lenient2"
    LENIENT1 = "lenient1"
    ORIGINAL = "original"


def get_quantile(data, quantile):
    """Quantile calculation modeled on the implementation used in sdtoolkit.

    Parameters
    ----------
    data : nd array like
           dataset for which quantile is needed
    quantile : float
               the desired quantile

    """
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
        while (data[index_lower] == data[index_higher]) & (
            index_higher < len(data) - 1
        ):
            index_higher += 1
        value = (data[index_lower] + data[index_higher]) / 2

    return value


class CurEntry:
    """a descriptor for the current entry on the peeling and pasting trajectory."""

    def __init__(self, dtype):
        """Init."""
        self.dtype = dtype

    def __set_name__(self, owner, name):  # noqa: D105
        self.name = name

    def __get__(self, instance, _):  # noqa: D105
        return instance.peeling_trajectory[self.name][instance._cur_box]

    def __set__(self, instance, value):  # noqa: D105
        raise PrimException("this property cannot be assigned to")


def calculate_qp(data, x, y, Hbox, Tbox, box_lim, initial_boxlim):  # noqa: N803
    """Helper function for calculating quasi p-values."""
    if data.size == 0:
        return [-1, -1]

    u = data.name
    dtype = data.dtype

    unlimited = initial_boxlim[u]

    if np.issubdtype(dtype, np.number):
        qp_values = []
        for direction, (limit, unlimit) in enumerate(zip(data, unlimited)):
            if unlimit != limit:
                temp_box = box_lim.copy()
                temp_box.at[direction, u] = unlimit
                qp = sdutil._calculate_quasip(x, y, temp_box, Hbox, Tbox)
            else:
                qp = -1
            qp_values.append(qp)
    else:
        temp_box = box_lim.copy()
        temp_box.loc[:, u] = unlimited
        qp = sdutil._calculate_quasip(x, y, temp_box, Hbox, Tbox)
        qp_values = [qp, -1]

    return qp_values


def rotate_subset(experiments, y):
    """Rotate a subset.

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

    """
    mean = np.mean(experiments, axis=0)
    std = np.std(experiments, axis=0)
    std[std == 0] = 1  # in order to avoid a division by zero
    experiments = (experiments - mean) / std

    # normalize the data
    subset_experiments = experiments.loc[y, :]

    # determine the rotation
    rotation_matrix = determine_rotation(subset_experiments)

    # apply the rotation
    rotated_experiments = np.dot(experiments, rotation_matrix)
    return rotation_matrix, rotated_experiments


def determine_rotation(experiments):
    """Determine the rotation for the specified experiments.

    Parameters
    ----------
    experiments : pd.DataFrame

    Returns
    -------
    ndarray

    """
    covariance = np.cov(experiments.T)

    eigen_vals, eigen_vectors = np.linalg.eig(covariance)

    indices = np.argsort(eigen_vals)
    indices = indices[::-1]
    eigen_vectors = eigen_vectors[:, indices]
    eigen_vals = eigen_vals[indices]

    # make the eigen vectors unit length
    for i in range(eigen_vectors.shape[1]):
        eigen_vectors[:, i] / np.linalg.norm(eigen_vectors[:, i]) * np.sqrt(
            eigen_vals[i]
        )

    return eigen_vectors


def determine_dimres(box, issignificant=True, significance_threshold=0.1):
    """Helper function for determining the restricted dimensions."""

    def is_significant(v):
        for entry in v:
            if (entry >= 0) & (entry <= significance_threshold):
                return True

    all_dims = set()
    for qp in box.p_values:
        if issignificant:
            dims = [k for k, v in qp.items() if is_significant(v)]
        else:
            dims = qp.keys()
        all_dims.update(dims)
    return all_dims


def box_to_tuple(box):
    """Helper function for converting box limits to tuple."""
    names = box.columns
    sorted(names)
    tupled_box = []
    for name in names:
        values = box[name]
        formatted = []
        for entry in values:
            if isinstance(entry, set):
                entry = tuple(entry)  # noqa: PLW2901
            formatted.append(entry)
        tupled_box += tuple(formatted)
    tupled_box = tuple(tupled_box)
    return tupled_box


class NotSeen:
    """Helper class."""

    def __init__(self):
        """Init."""
        self.seen = set()

    def __call__(self, box):  # noqa: D102
        tupled = box_to_tuple(box)
        if tupled in self.seen:
            return False
        else:
            self.seen.add(tupled)
            return True


def is_significant(box, i, alpha=0.05):
    """Check if quasi-p values are significant."""
    qp = box.p_values[i]
    return not any(value > alpha for values in qp.values() for value in values)


def is_pareto_efficient(data):
    """Check if a given datapoint is pareto efficient."""
    fronts = calc_fronts(data)
    return fronts == 0


def calc_fronts(boxes):
    """Non dominated sort."""
    # taken from
    # https://stackoverflow.com/questions/41740596/pareto-frontier-indices-using-numpy
    i_dominates_j = np.all(boxes[:, None] >= boxes, axis=-1) & np.any(
        boxes[:, None] > boxes, axis=-1
    )
    remaining = np.arange(len(boxes))
    fronts = np.empty(len(boxes), int)
    frontier_index = 0
    while remaining.size > 0:
        dominated = np.any(i_dominates_j[remaining[:, None], remaining], axis=0)
        fronts[remaining[~dominated]] = frontier_index

        remaining = remaining[dominated]
        frontier_index += 1
    return fronts
