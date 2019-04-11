'''
This module provides time series clustering functionality using
complex invariant distance

'''
import itertools
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import sklearn.cluster as cluster

from ..util import get_module_logger


#
# Created on  11 Apr 2019
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
#


__all__ = ['calculate_cid',
           'plot_dendrogram',
           'apply_agglomerative_clustering']

_logger = get_module_logger(__name__)


def CID(xi, xj, ce_i, ce_j):
    return np.linalg.norm(xi - xj) * (max(ce_i, ce_j) / min(ce_i, ce_j))


def calculate_cid(data, condensed_form=False):
    '''calculate the complex invariant distance between all rows


    Parameters
    ----------
    data : 2d ndarray
    condensed_form : bool, optional

    Returns
    -------
    distances
        a 2D ndarray with the distances between all time series, or condensed
        form similar to scipy.spatial.distance.pdist¶



    '''
    ce = np.sqrt(np.sum(np.diff(data, axis=1)**2, axis=1))

    indices = np.arange(0, data.shape[0])
    cid = np.zeros((data.shape[0], data.shape[0]))

    for i, j in itertools.combinations(indices, 2):
        xi = data[i, :]
        xj = data[j, :]
        ce_i = ce[i]
        ce_j = ce[j]

        distance = CID(xi, xj, ce_i, ce_j)
        cid[i, j] = distance
        cid[j, i] = distance

    if not condensed_form:
        return cid
    else:
        return sp.spatial.distance.squareform(cid)


def plot_dendrogram(distances):
    '''plot dendrogram for distances
    '''

    if distances.ndim == 2:
        distances = sp.spatial.distance.squareform(distances)
    linked = sp.cluster.hierarchy.linkage(distances)  # @UndefinedVariable

    fig = plt.figure()
    sp.cluster.hierarchy.dendrogram(linked,  # @UndefinedVariable
                                    orientation='top',
                                    distance_sort='descending',
                                    show_leaf_counts=True)
    return fig


def apply_agglomerative_clustering(distances, n_clusters, linkage='average'):
    '''apply agglomerative clustering to the distances

    Parameters
    ----------
    distances : ndarray
    n_clusters : int
    linkage : {'average', 'complete', 'single'}

    Returns
    -------
    1D ndarray with cluster assignment

    '''

    c = cluster.AgglomerativeClustering(n_clusters=n_clusters,
                                        affinity='precomputed',
                                        linkage=linkage)
    clusters = c.fit_predict(distances)
    return clusters
