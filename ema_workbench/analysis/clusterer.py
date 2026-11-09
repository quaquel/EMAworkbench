"""Time series clustering functionality using complex invariant distance.

For details see `Steinmann et al (2020) <https://doi.org/10.1016/j.techfore.2020.120052>`_

"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sklearn.cluster as cluster

from ..util import get_module_logger

#
# Created on  11 Apr 2019
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
#


__all__ = [
    "apply_agglomerative_clustering",
    "calculate_cid",
    "normalize_distances",
    "plot_dendrogram",
]

_logger = get_module_logger(__name__)


def CID(xi, xj, ce_i, ce_j):
    """Calculate Complex Invariant Distance between two points."""
    return np.linalg.norm(xi - xj) * (max(ce_i, ce_j) / min(ce_i, ce_j))


def calculate_cid(
    data: np.ndarray,
    condensed_form: bool = False,
    chunk_size: int = 1000,
    rel_floor: float = 1e-3,
) -> np.ndarray:
    """Calculate the complex invariant distance between all rows.

    CID(a, b) = ED(a, b) * max(CE(a), CE(b)) / min(CE(a), CE(b)), with
    ED the Euclidean distance and CE(a) = sqrt(sum_t (a_{t+1} - a_t)^2) the
    complexity estimate of a series (Batista et al., 2014).

    Parameters
    ----------
    data : 2d ndarray
           one time series per row, all on the same time grid
    condensed_form : bool, optional
                     if True, return the condensed form, similar to
                     scipy.spatial.distance.pdist
    chunk_size : int, optional
                 number of rows processed at once; bounds peak memory
    rel_floor : float, optional
                complexity estimates are floored at rel_floor times the
                median positive complexity estimate. This keeps distances
                involving flat (constant) series finite, while keeping them
                large relative to non-flat series.

    Returns
    -------
    distances
        a 2D ndarray with the distances between all time series, or condensed
        form similar to scipy.spatial.distance.pdist

    """
    data = np.asarray(data, dtype=float)
    n = data.shape[0]

    ce = np.sqrt(np.sum(np.diff(data, axis=1) ** 2, axis=1))
    positive = ce[ce > 0]
    floor = rel_floor * np.median(positive) if positive.size else 1.0
    ce = np.maximum(ce, floor)

    sq_norms = np.sum(data**2, axis=1)
    cid = np.empty((n, n))
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        # squared Euclidean distance via |a|^2 + |b|^2 - 2ab, clipped for round-off
        ed2 = (
            sq_norms[start:stop, None]
            + sq_norms[None, :]
            - 2 * data[start:stop] @ data.T
        )
        ed = np.sqrt(np.maximum(ed2, 0))
        ce_block = ce[start:stop, None]
        cf = np.maximum(ce_block, ce[None, :]) / np.minimum(ce_block, ce[None, :])
        cid[start:stop] = ed * cf

    np.fill_diagonal(cid, 0)
    cid = np.minimum(cid, cid.T)  # exact symmetry despite round-off

    if not condensed_form:
        return cid
    else:
        return sp.spatial.distance.squareform(cid, checks=False)


def normalize_distances(
    distances: np.ndarray,
    method: Literal["exact", "sample"] = "exact",
    sample_size: int = 200_000,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """Divide distances by the median pairwise distance.

    After normalization, a distance of 1 means 'a typical pairwise distance'.
    This makes distances comparable across models and outcomes, which matters
    when distances are traded off against other quantities. The median rather
    than the mean is used because distances such as CID are heavy tailed.

    Parameters
    ----------
    distances : ndarray
                square (n, n) distance matrix with zero diagonal, or the
                condensed form
    method : {"exact", "sample"}, optional
             "exact" uses all n(n-1)/2 distinct pairs. "sample" uses
             sample_size random pairs, which bounds memory and time for very
             large n (relative error ~0.1-0.4% at the default sample size).
             For condensed input the exact median is always cheap and is used.
    sample_size : int, optional
                  number of pairs drawn when method="sample"
    seed : int, optional
           random seed for method="sample"

    Returns
    -------
    tuple
        the normalized distances (same form as the input) and the median
        distance that was divided out

    """
    distances = np.asarray(distances, dtype=float)

    if distances.ndim == 1:
        scale = float(np.median(distances))
    else:
        n = distances.shape[0]
        if distances.shape != (n, n):
            raise ValueError("distances must be square or condensed")
        if n < 2:
            raise ValueError("need at least two items to normalize distances")
        if method == "exact":
            # gather the strict upper triangle row by row; avoids np.triu_indices,
            # whose two index arrays are 4x larger than the values themselves
            upper = np.concatenate([distances[i, i + 1 :] for i in range(n - 1)])
            scale = float(np.median(upper, overwrite_input=True))
        elif method == "sample":
            rng = np.random.default_rng(seed)
            i = rng.integers(0, n, sample_size)
            j = rng.integers(0, n, sample_size)
            keep = i != j  # self distances are 0 and would bias the median
            scale = float(np.median(distances[i[keep], j[keep]]))
        else:
            raise ValueError(f"method must be 'exact' or 'sample', not {method!r}")

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"median pairwise distance is {scale}; items are (nearly) identical "
            "or distances contain non-finite values"
        )
    return distances / scale, scale


def plot_dendrogram(distances):
    """Plot dendrogram for distances."""
    if distances.ndim == 2:
        distances = sp.spatial.distance.squareform(distances)
    linked = sp.cluster.hierarchy.linkage(distances)  # @UndefinedVariable

    fig = plt.figure()
    sp.cluster.hierarchy.dendrogram(
        linked,  # @UndefinedVariable
        orientation="top",
        distance_sort="descending",
        show_leaf_counts=True,
    )
    return fig


def apply_agglomerative_clustering(
    distances, n_clusters, metric="precomputed", linkage="average"
):
    """Apply agglomerative clustering to the distances.

    Parameters
    ----------
    distances : ndarray
    n_clusters : int
    metric : str, optional. The distance metric to use. The default is 'precomputed'. For a list of available metrics, see the documentation of scipy.spatial.distance.pdist.
    linkage : {'average', 'complete', 'single'}

    Returns
    -------
    1D ndarray with cluster assignment

    """
    c = cluster.AgglomerativeClustering(
        n_clusters=n_clusters, metric=metric, linkage=linkage
    )
    clusters = c.fit_predict(distances)
    return clusters
