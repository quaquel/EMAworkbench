''' helper module that implements the PCA preprocessing that might be used 
in combination with PRIM

'''

# Created on 12 Jan 2019
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

import numpy as np
import pandas as pd


__all__ = ['pca_preprocess']


def pca_preprocess(experiments, y, subsets=None, exclude=set()):
    '''perform PCA to preprocess experiments before running PRIM

    Pre-process the data by performing a pca based rotation on it.
    This effectively turns the algorithm into PCA-PRIM as described
    in `Dalal et al (2013) <http://www.sciencedirect.com/science/article/pii/S1364815213001345>`_

    Parameters
    ----------
    experiments : DataFrame
    y : ndarray
        one dimensional binary array
    subsets : dict, optional
              expects a dictionary with group name as key and a list of
              uncertainty names as values. If this is used, a constrained
              PCA-PRIM is executed
    exclude : list of str, optional
              the uncertainties that should be excluded from the rotation

    Returns
    -------
    rotated_experiments
        DataFrame
    rotation_matrix
        DataFrame
    
    Raises
    ------
    RuntimeError
        if mode is not binary (i.e. y is not a binary classification).
        if X contains non numeric columns


    '''
    # experiments to rotate
    x = experiments.drop(exclude, axis=1)
    
    #
    if not x.select_dtypes(exclude=np.number).empty:
        raise RuntimeError("X includes non numeric columns")
    if not set(np.unique(y)) == {0, 1}:
        raise RuntimeError('y should only contain 0s and 1s')

    # if no subsets are provided all uncertainties with non dtype object
    # are in the same subset, the name of this is r, for rotation
    if not subsets:
        subsets = {"r": x.columns.values.tolist()}
    else:
        # TODO:: should we check on double counting in subsets?
        #        should we check all uncertainties are in x?
        pass
        
    # prepare the dtypes for the new rotated experiments recarray
    new_columns = []
    new_dtypes = []
    for key, value in subsets.items():
        # the names of the rotated columns are based on the group name
        # and an index
        subset_cols = ["{}_{}".format(key, i) for i in range(len(value))]
        new_columns.extend(subset_cols)
        new_dtypes.extend((float,)*len(value))

    # make a new empty experiments dataframe
    rotated_experiments = pd.DataFrame(index=experiments.index.values)

    for name, dtype in zip(new_columns, new_dtypes):
        rotated_experiments[name] = pd.Series(dtype=dtype)

    # put the uncertainties with object dtypes already into the new
    for entry in exclude:
        rotated_experiments[name] = experiments[entry]

    # iterate over the subsets, rotate them, and put them into the new
    # experiments dataframe
    rotation_matrix = np.zeros((x.shape[1], )*2)
    column_names = []
    row_names = []

    j = 0
    for key, value in subsets.items():
        x_subset = x[value]
        subset_rotmat, subset_experiments = rotate_subset(x_subset, y)
        rotation_matrix[j:j + len(value), j:j + len(value)] = subset_rotmat
        row_names.extend(value)
        j += len(value)

        for i in range(len(value)):
            name = "%s_%s" % (key, i)
            rotated_experiments[name] = subset_experiments[:, i]
            [column_names.append(name)]

    rotation_matrix = pd.DataFrame(rotation_matrix, index=row_names,
                                   columns=column_names)

    return rotated_experiments, rotation_matrix


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

