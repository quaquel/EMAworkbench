"""Time series clustering functionality using complex invariant distance.

For details see `Steinmann et al (2020) <https://doi.org/10.1016/j.techfore.2020.120052>`_

"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sklearn.cluster as cluster
from numpy.typing import DTypeLike

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
    dtype: DTypeLike = np.float64,
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
                     scipy.spatial.distance.pdist.
    chunk_size : int, optional
                 number of rows processed at once; bounds the memory needed
                 on top of the result
    rel_floor : float, optional
                complexity estimates are floored at rel_floor times the
                median positive complexity estimate. This keeps distances
                involving flat (constant) series finite, while keeping them
                large relative to non-flat series.
    dtype : floating dtype, optional
            dtype of the returned distances. Distances are always computed in
            float64; np.float32 halves the memory of the result, e.g. 3.6 GB
            instead of 7.2 GB for the square form of 30,000 series.

    Returns
    -------
    distances
        a 2D ndarray with the distances between all time series, or condensed
        form similar to scipy.spatial.distance.pdist

    Notes
    -----
    Only the upper triangle is computed, block by block, and mirrored into the
    lower triangle, so the result is exactly symmetric without a second
    n by n array. Peak memory is the result plus a few chunk_size by n blocks.

    """
    data = np.asarray(data, dtype=float)
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.floating):
        raise ValueError(f"dtype must be a floating dtype, not {dtype}")
    n = data.shape[0]

    ce = np.sqrt(np.sum(np.diff(data, axis=1) ** 2, axis=1))
    positive = ce[ce > 0]
    floor = rel_floor * np.median(positive) if positive.size else 1.0
    ce = np.maximum(ce, floor)
    sq_norms = np.sum(data**2, axis=1)

    if condensed_form:
        cid = np.empty(n * (n - 1) // 2, dtype=dtype)
    else:
        cid = np.empty((n, n), dtype=dtype)

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        # distances between rows start:stop and rows start:n, i.e. the
        # upper triangle of this block row; |a|^2 + |b|^2 - 2ab is clipped
        # for round-off
        ed2 = (
            sq_norms[start:stop, None]
            + sq_norms[None, start:]
            - 2 * data[start:stop] @ data[start:].T
        )
        ed = np.sqrt(np.maximum(ed2, 0))
        ce_block = ce[start:stop, None]
        ce_rest = ce[None, start:]
        block = ed * (np.maximum(ce_block, ce_rest) / np.minimum(ce_block, ce_rest))

        # the diagonal square of the block must be symmetric before mirroring
        size = stop - start
        square = block[:, :size]
        square[...] = np.minimum(square, square.T)
        np.fill_diagonal(square, 0)

        if condensed_form:
            # row i holds the distances to rows i + 1, ..., n - 1
            for i in range(start, stop):
                offset = i * n - i * (i + 1) // 2
                cid[offset : offset + n - i - 1] = block[i - start, i - start + 1 :]
        else:
            cid[start:stop, start:] = block
            cid[start:, start:stop] = block.T

    return cid


def normalize_distances(
    distances: np.ndarray,
    method: Literal["exact", "sample"] = "exact",
    sample_size: int = 200_000,
    seed: int | None = None,
    copy: bool = True,
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
                condensed form. A floating dtype (e.g. float32) is kept.
    method : {"exact", "sample"}, optional
             "exact" uses all n(n-1)/2 distinct pairs. "sample" uses
             sample_size random pairs, which bounds memory and time for very
             large n (relative error ~0.1-0.4% at the default sample size).
             For condensed input the exact median is always cheap and is used.
    sample_size : int, optional
                  number of pairs drawn when method="sample"
    seed : int, optional
           random seed for method="sample"
    copy : bool, optional
           if False and distances is a floating ndarray, divide it in place
           instead of returning a new array, which saves a full copy

    Returns
    -------
    tuple
        the normalized distances (same form as the input) and the median
        distance that was divided out

    """
    distances = np.asarray(distances)
    if not np.issubdtype(distances.dtype, np.floating):
        distances = distances.astype(float)

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
            del upper
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
    if copy:
        return distances / scale, scale
    distances /= scale
    return distances, scale


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
