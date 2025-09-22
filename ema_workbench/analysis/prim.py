"""A scenario discovery oriented implementation of PRIM.

The implementation of prim provided here is data type aware, so
categorical variables will be handled appropriately. It also uses a
non-standard objective function in the peeling and pasting phase of the
algorithm. This algorithm looks at the increase in the mean divided
by the amount of data removed. So essentially, it uses something akin
to the first order derivative of the original objective function.

The implementation is designed for interactive use in combination with
the jupyter notebook.

"""

import abc
import contextlib
import copy
import itertools
import numbers
import warnings
from collections.abc import Sequence
from operator import itemgetter
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from sklearn.metrics import root_mean_squared_error as rmse

try:
    import altair as alt
except ImportError:
    alt = None
    warnings.warn(
        "altair based interactive inspection not available", ImportWarning, stacklevel=2
    )

from ..util import INFO, EMAError, get_module_logger, temporary_filter
from . import scenario_discovery_util as sdutil
from .prim_util import (
    CurEntry,
    NotSeen,
    PrimException,
    PRIMObjectiveFunctions,
    calculate_qp,
    determine_dimres,
    get_quantile,
    is_pareto_efficient,
    is_significant,
    rotate_subset,
)

# Created on 22 feb. 2013
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence
RNGLike = np.random.Generator | np.random.BitGenerator


__all__ = [
    "PRIMObjectiveFunctions",
    "Prim",
    "PrimBox",
    "pca_preprocess",
    "run_constrained_prim",
]
_logger = get_module_logger(__name__)

PRECISION = ".2f"


def pca_preprocess(
    experiments: pd.DataFrame,
    y: np.ndarray,
    subsets: dict | None = None,
    exclude: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform PCA to preprocess experiments before running PRIM.

    Pre-process the data by performing a pca based rotation on it.
    This effectively turns the algorithm into PCA-PRIM as described
    in `Dalal et al. (2013) <https://www.sciencedirect.com/science/article/pii/S1364815213001345>`_

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


    """
    # experiments to rotate
    exclude = set() if exclude is None else exclude

    x = experiments.drop(exclude, axis=1)

    #
    non_numerical_columns = x.select_dtypes(exclude=np.number)
    if not non_numerical_columns.empty:
        raise ValueError(
            f"X includes non numeric columns: {non_numerical_columns.columns.values.tolist()}"
        )
    if not set(np.unique(y)) == {0, 1}:
        raise ValueError(
            f"y should only contain 0s and 1s, currently y contains {set(np.unique(y))}."
        )

    # if no subsets are provided all uncertainties with non dtype object
    # are in the same subset, the name of this is r, for rotation
    if not subsets:
        subsets = {"r": x.columns.values.tolist()}
    else:
        # TODO:: should we check on double counting in subsets?
        #        should we check all uncertainties are in x?
        #        also, subsets cannot be a single uncertainty,
        pass

    # prepare the dtypes for the new rotated experiments dataframe
    new_columns = []
    new_dtypes = []
    for key, value in subsets.items():
        # the names of the rotated columns are based on the group name
        # and an index
        subset_cols = [f"{key}_{i}" for i in range(len(value))]
        new_columns.extend(subset_cols)
        new_dtypes.extend((float,) * len(value))

    # make a new empty experiments dataframe
    rotated_experiments = pd.DataFrame(index=experiments.index.values)

    for name, dtype in zip(new_columns, new_dtypes):
        rotated_experiments[name] = pd.Series(dtype=dtype)

    # put the uncertainties with object dtypes already into the new
    for entry in exclude:
        rotated_experiments[entry] = experiments[entry]

    # iterate over the subsets, rotate them, and put them into the new
    # experiments dataframe
    rotation_matrix = np.zeros((x.shape[1],) * 2)
    column_names = []
    row_names = []

    j = 0
    for key, value in subsets.items():
        x_subset = x[value]
        subset_rotmat, subset_experiments = rotate_subset(x_subset, y)
        rotation_matrix[j : j + len(value), j : j + len(value)] = subset_rotmat
        row_names.extend(value)
        j += len(value)

        for i in range(len(value)):
            name = f"{key}_{i}"
            rotated_experiments[name] = subset_experiments[:, i]
            column_names.append(name)

    rotation_matrix = pd.DataFrame(
        rotation_matrix, index=row_names, columns=column_names
    )

    return rotated_experiments, rotation_matrix


def run_constrained_prim(
    experiments: pd.DataFrame, y: np.ndarray, issignificant: bool = True, **kwargs
) -> "PrimBox":
    """Run PRIM repeatedly while constraining the maximum number of dimensions available in x.

    Improved usage of PRIM as described in `Kwakkel (2019) <https://onlinelibrary.wiley.com/doi/full/10.1002/ffo2.8>`_.

    Parameters
    ----------
    experiments : DataFrame
    y : numpy array
    issignificant : bool, optional
                    if True, run prim only on subsets of dimensions
                    that are significant for the initial PRIM on the
                    entire dataset.
    **kwargs : any additional keyword arguments are passed on to PRIM


    Returns
    -------
    PrimBox instance

    """
    frontier = []
    merged_lims = []
    merged_qp = []

    alg = Prim(experiments, y, **kwargs)
    boxn = alg.find_box()

    dims = determine_dimres(boxn, issignificant=issignificant)

    # run prim for all possible combinations of dims
    subsets = []
    for n in range(1, len(dims) + 1):
        for subset in itertools.combinations(dims, n):
            subsets.append(subset)
    _logger.info(f"going to run PRIM {len(subsets)} times")

    boxes = [boxn]
    for subset in subsets:
        with temporary_filter(__name__, INFO):
            x = experiments.loc[:, subset].copy()
            alg = Prim(x, y, **kwargs)
            box = alg.find_box()
            boxes.append(box)

    box_init = boxn.prim.box_init
    not_seen = NotSeen()

    for box in boxes:
        peeling = box.peeling_trajectory
        lims = box.box_lims

        logical = np.ones(box.peeling_trajectory.shape[0], dtype=bool)

        for i in range(box.peeling_trajectory.shape[0]):
            lim = lims[i]

            boxlim = box_init.copy()
            for column in lim:
                boxlim[column] = lim[column]

            boolean = is_significant(box, i) & not_seen(boxlim)

            logical[i] = boolean
            if boolean:
                merged_lims.append(boxlim)
                merged_qp.append(box.p_values[i])
        frontier.append(peeling[logical])

    frontier = pd.concat(frontier)

    # remove dominated boxes
    peeling_trajectory = frontier.reset_index(drop=True)
    data = peeling_trajectory.iloc[:, [0, 1, -1]].copy().values
    data[:, 2] *= -1
    logical = is_pareto_efficient(data)

    # resort to ensure sensible ordering
    pt = peeling_trajectory[logical]
    pt = pt.reset_index(drop=True)
    pt = pt.sort_values(["res_dim", "coverage"], ascending=[True, False])

    box_lims = [lim for lim, entry in zip(merged_lims, logical) if entry]
    qps = [qp for qp, entry in zip(merged_qp, logical) if entry]
    sorted_lims = []
    sorted_qps = []
    for entry in pt.index:
        sorted_lims.append(box_lims[int(entry)])
        sorted_qps.append(qps[int(entry)])

    # ensuring index has normal order starting from 0
    pt = pt.reset_index(drop=True)
    pt.id = pt.index

    # create PrimBox
    box = PrimBox(boxn.prim, boxn.prim.box_init, boxn.prim.yi)
    box.peeling_trajectory = pt
    box.box_lims = sorted_lims
    box.p_values = sorted_qps
    return box


class BasePrimBox(abc.ABC):
    _frozen = False

    mean = CurEntry(
        float
    )  # fixme, can't we use __get_name__ to get rid of name in CurEntry init?
    res_dim = CurEntry(int)
    mass = CurEntry(float)

    def __init__(self, prim: "BasePrim", box_lims: pd.DataFrame, indices: np.ndarray):
        """Init.

        Parameters
        ----------
        prim : Prim instance
        box_lims : DataFrame
        indices : ndarray


        """
        self.prim = prim

        # peeling and pasting trajectory
        columns = {
            "id": pd.Series(dtype=int),
            "n": pd.Series(dtype=int),  # items in box
        }
        for klass in type(self).__mro__:
            for name, value in vars(klass).items():
                if isinstance(value, CurEntry):
                    columns[name] = pd.Series(dtype=value.dtype)
        self.peeling_trajectory = pd.DataFrame(columns)

        self.box_lims = []
        self.p_values = []
        self.yi_initial = indices[:]
        self._cur_box = -1
        self.yi = None

    def __getattr__(self, name: str):
        """Small override for attribute access of box_lims.

        Used here to give box_lim same behaviour as coverage, density,
        mean, res_dim, and mass. That is, it will return the box lim
        associated with the currently selected box.
        """
        if name == "box_lim":
            return self.box_lims[self._cur_box]
        else:
            raise AttributeError

    @property
    @abc.abstractmethod
    def stats(self): ...

    def inspect(
        self,
        i: int | None = None,
        style: Literal["table", "graph", "data"] = "table",
        ax=None,
        **kwargs,
    ):
        """Write the stats and box limits of the user specified box to standard out.

        If it is not provided, the last box will be printed

        Parameters
        ----------
        i : int or list of ints, optional
            the index of the box, defaults to currently selected box
        style : {'table', 'graph', 'data'}
                the style of the visualization. 'table' prints the stats and
                boxlim. 'graph' creates a figure. 'data' returns a list of
                tuples, where each tuple contains the stats and the box_lims.
        ax : axes or list of axes instances, optional
             used in conjunction with `graph` style, allows you to control the axes on which graph is plotted
             if i is list, axes should be list of equal length. If axes is None, each i_j in i will be plotted
             in a separate figure.

        additional kwargs are passed to the helper function that
        generates the table or graph

        """
        if style not in {"table", "graph", "data"}:
            raise ValueError(
                f"style must be one of 'table', 'graph', or 'data', not {style}"
            )

        if i is None:
            i = [self._cur_box]
        elif isinstance(i, int):
            i = [i]

        if isinstance(ax, mpl.axes.Axes):
            ax = [ax]

        if not all(isinstance(x, int) for x in i):
            raise TypeError(f"i must be an integer or list of integers, not {type(i)}")

        if (ax is not None) and style == "graph":
            if len(ax) != len(i):
                raise ValueError(
                    f"the number of axes ({len(ax)}) does not match the number of boxes to inspect ({len(i)})"
                )
            else:
                return [
                    self._inspect(i_j, style=style, ax=ax, **kwargs)
                    for i_j, ax in zip(i, ax)
                ]
        else:
            return [self._inspect(entry, style=style, **kwargs) for entry in i]

    def _inspect(
        self,
        i: int | None = None,
        style: Literal["table", "graph", "data"] = "table",
        **kwargs,
    ):
        """Helper method for inspecting one or more boxes on the peeling trajectory.

        Parameters
        ----------
        i, int
        style, {'table', 'graph', 'data'}

        additional kwargs are passed to the helper function that
        generates the table or graph

        """
        uncs = sdutil._determine_restricted_dims(self.box_lims[i], self.box_lims[0])

        match style:
            case "table":
                return self._inspect_table(i, uncs)
            case "graph":
                try:
                    ax = kwargs.pop("ax")
                except KeyError:
                    fig, ax = plt.subplots()
                return self._inspect_graph(i, uncs, ax=ax, **kwargs)
            case "data":
                return self._inspect_data(i, uncs)
            case _:
                raise ValueError(
                    f"style must be one of 'graph', 'table' or 'data', not {style}."
                )

    def _inspect_data(
        self, i: int, uncs: list[str] | np.ndarray
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Helper method for inspecting boxes.

        This one returns a tuple with a series with overall statistics, and a
        DataFrame containing the boxlims and qp values

        """
        p_values = self.p_values[i]

        # make the descriptive statistics for the box
        stats = self.peeling_trajectory.iloc[i]

        # make the box definition
        columns = pd.MultiIndex.from_product(
            [[f"box {i}"], ["min", "max", "qp value", "qp value"]]
        )
        box_lim = pd.DataFrame(np.zeros((len(uncs), 4)), index=uncs, columns=columns)

        for unc in uncs:
            values = self.box_lims[i][unc]
            box_lim.loc[unc] = values.values.tolist() + p_values[unc]
            box_lim.iloc[:, 2::] = box_lim.iloc[:, 2::].replace(-1, np.nan)

        return stats, box_lim

    def _inspect_table(self, i: int, uncs: list[str]):
        """Helper method for visualizing box statistics in table form."""
        # make the descriptive statistics for the box
        stats, box_lim = self._inspect_data(i, uncs)

        print(stats)
        print()
        print(box_lim)
        print()

    def select(self, i: int):
        """Select an entry from the peeling and pasting trajectory.

        The prim box will be updated to this selected box.

        Parameters
        ----------
        i : int
            the index of the box to select.

        """
        if self._frozen:
            raise PrimException(
                "box has been frozen because PRIM "
                "has found at least one more recent "
                "box"
            )

        res_dim = sdutil._determine_restricted_dims(
            self.box_lims[i], self.prim.box_init
        )

        indices = sdutil._in_box(
            self.prim.x.loc[self.prim.yi_remaining, res_dim], self.box_lims[i][res_dim]
        )
        self.yi = self.prim.yi_remaining[indices]
        self._cur_box = i

    def drop_restriction(self, uncertainty: str = "", i: int | None = None):
        """Drop the restriction on the specified dimension for box i.

        Parameters
        ----------
        i : int, optional
            defaults to the currently selected box, which
            defaults to the latest box on the trajectory
        uncertainty : str


        Replace the limits in box i with a new box where
        for the specified uncertainty the limits of the initial box are
        being used. The resulting box is added to the peeling trajectory.

        """
        if i is None:
            i = self._cur_box

        new_box_lim = self.box_lims[i].copy()
        new_box_lim.loc[:, uncertainty] = self.box_lims[0].loc[:, uncertainty]
        indices = sdutil._in_box(
            self.prim.x.loc[self.prim.yi_remaining, :], new_box_lim
        )
        indices = self.prim.yi_remaining[indices]
        self.update(new_box_lim, indices)

    @abc.abstractmethod
    def update(self, box_lims: pd.DataFrame, indices: np.ndarray):
        """Update the box to the provided box limits.

        Parameters
        ----------
        box_lims: DataFrame
                  the new box_lims
        indices: ndarray
                 the indices of y that are inside the box

        """
        ...

    def show_ppt(self):
        """Show the peeling and pasting trajectory in a figure."""
        return sdutil.plot_ppt(self.peeling_trajectory)

    def show_pairs_scatter(
        self,
        i: int | None = None,
        dims: list[str] | None = None,
        diag: Literal["kde", "cdf", "regression"] | None = "kde",
        upper: Literal["scatter", "hexbin", "hist", "contour"] | None = "scatter",
        lower: Literal["scatter", "hexbin", "hist", "contour"] | None = "contour",
        fill_subplots: bool = True,
        legend=True,
    ) -> sns.PairGrid:
        """Make a pair wise scatter plot of all the restricted dimensions.

        Color denotes whether a given point is of
        interest or not and the boxlims superimposed on top.

        Parameters
        ----------
        i : int, optional
        dims : list of str, optional
               dimensions to show, defaults to all restricted dimensions
        diag : {"kde", "cdf", "regression"}
               Plot diagonal as kernel density estimate ('kde'),
               cumulative density function ('cdf'), or regression ('regression')
        upper, lower: string, optional
               Use either 'scatter', 'contour', or 'hist' (bivariate
               histogram) plots for upper and lower triangles. Upper triangle
               can also be 'none' to eliminate redundancy. Legend uses
               lower triangle style for markers.
        fill_subplots: Boolean, optional
                       if True, subplots are resized to fill their respective axes.
                       This removes unnecessary whitespace, but may be undesirable
                       for some variable combinations.

        Returns
        -------
        seaborn PairGrid

        """
        if i is None:
            i = self._cur_box

        if dims is None:
            dims = sdutil._determine_restricted_dims(
                self.box_lims[i], self.prim.box_init
            )

        if diag not in {"kde", "cdf", "regression"}:
            raise ValueError(
                f"diag_kind should be one of DiagKind.KDE or DiagKind.CDF, not {diag}"
            )

        return sdutil.plot_pair_wise_scatter(
            self.prim.x.iloc[self.yi_initial, :],
            self.prim.y[self.yi_initial],
            self.box_lims[i],
            self.prim.box_init,
            dims,
            diag=diag,
            upper=upper,
            lower=lower,
            fill_subplots=fill_subplots,
            legend=legend,
        )

    def write_ppt_to_stdout(self):
        """Write the peeling and pasting trajectory to stdout."""
        print(self.peeling_trajectory)
        print("\n")


class PrimBox(BasePrimBox):
    """A class that holds information for a specific box.

    Attributes
    ----------
    coverage : float
               coverage of currently selected box
    density : float
               density of currently selected box
    mean : float
           mean of currently selected box
    res_dim : int
              number of restricted dimensions of currently selected box
    mass : float
           mass of currently selected box
    peeling_trajectory : DataFrame
                         stats for each box in peeling trajectory
    box_lims : list
               list of box lims for each box in peeling trajectory


    by default, the currently selected box is the last box on the
    peeling trajectory, unless this is changed via
    :meth:`PrimBox.select`.

    """

    coverage = CurEntry(float)
    density = CurEntry(float)
    k = CurEntry(int)

    def __init__(self, prim: "BasePrim", box_lims: pd.DataFrame, indices: np.ndarray):
        """Init.

        Parameters
        ----------
        prim : Prim instance
        box_lims : DataFrame
        indices : ndarray


        """
        super().__init__(prim, box_lims, indices)
        self._resampled = []

        # indices van data in box
        self.update(box_lims, indices)

    @property
    def stats(self):
        """Return stats of this box."""
        return {k: getattr(self, k) for k in ["coverage", "density", "mass", "res_dim"]}

    def inspect_tradeoff(self):
        """Inspecting tradeoff using altair."""
        # TODO::
        # make legend with res_dim color code a selector as well?
        # https://medium.com/dataexplorations/focus-generating-an-interactive-legend-in-altair-9a92b5714c55

        boxes = []
        nominal_vars = []
        quantitative_dims = set(
            self.prim.x_float_colums.tolist() + self.prim.x_int_columns.tolist()
        )
        nominal_dims = set(self.prim.x_nominal_columns)

        box_zero = self.box_lims[0]

        for i, (entry, qp) in enumerate(zip(self.box_lims, self.p_values)):
            qp = pd.DataFrame(qp, index=["qp_lower", "qp_upper"])  # noqa: PLW2901
            dims = qp.columns.tolist()
            quantitative_res_dim = [e for e in dims if e in quantitative_dims]
            nominal_res_dims = [e for e in dims if e in nominal_dims]

            # handle quantitative
            df = entry

            box = df[quantitative_res_dim]
            box.index = ["x1", "x2"]
            box = box.T
            box["name"] = box.index
            box["id"] = int(i)
            box["minimum"] = box_zero[quantitative_res_dim].T.iloc[:, 0]
            box["maximum"] = box_zero[quantitative_res_dim].T.iloc[:, 1]
            box = box.join(qp.T)
            boxes.append(box)

            # handle nominal
            for dim in nominal_res_dims:
                # TODO:: qp values
                items = df[dim].values[0]
                for j, item in enumerate(items):
                    # we need to have tick labeling to be dynamic?
                    # adding it to the dict won't work, creates horrible figure
                    # unless we can force a selection?
                    name = f"{dim}, {qp.loc[qp.index[0], dim]: .2g}"
                    tick_info = {
                        "name": name,
                        "n_items": len(items) + 1,
                        "item": item,
                        "id": int(i),
                        "x": j / len(items),
                    }
                    nominal_vars.append(tick_info)

        boxes = pd.concat(boxes)
        nominal_vars = pd.DataFrame(nominal_vars)

        width = 400
        height = width

        point_selector = alt.selection_point(fields=["id"])

        peeling = self.peeling_trajectory.copy()
        peeling["id"] = peeling.index

        chart = (
            alt.Chart(peeling)
            .mark_circle(size=75)
            .encode(
                x=alt.X("coverage:Q", scale=alt.Scale(domain=(0, 1.1))),
                y=alt.Y("density:Q", scale=alt.Scale(domain=(0, 1.1))),
                color=alt.Color(
                    "res_dim:O",
                    scale=alt.Scale(
                        range=sns.color_palette("YlGnBu", n_colors=8).as_hex()
                    ),
                ),
                opacity=alt.condition(point_selector, alt.value(1), alt.value(0.4)),
                tooltip=[
                    alt.Tooltip("id:Q"),
                    alt.Tooltip("coverage:Q", format=".2"),
                    alt.Tooltip("density:Q", format=".2"),
                    alt.Tooltip("res_dim:O"),
                ],
            )
            .add_params(point_selector)
            .properties(width=width, height=height)
        )

        base = (
            alt.Chart(boxes)
            .encode(
                x=alt.X(
                    "x_lower:Q",
                    axis=alt.Axis(grid=False, title="box limits", labels=False),
                    scale=alt.Scale(domain=(0, 1), padding=0.1),
                ),
                x2="x_upper:Q",
                y=alt.Y("name:N", scale=alt.Scale(padding=1.0)),
            )
            .transform_calculate(
                x_lower="(datum.x1-datum.minimum)/(datum.maximum-datum.minimum)",
                x_upper="(datum.x2-datum.minimum)/(datum.maximum-datum.minimum)",
            )
            .transform_filter(point_selector)
            .properties(width=width)
        )

        lines = base.mark_rule()

        texts1 = (
            base.mark_text(baseline="top", dy=5, align="left")
            .encode(text=alt.Text("text:O"))
            .transform_calculate(
                text=(
                    "datum.qp_lower>0?"
                    'format(datum.x1, ".2")+" ("+format(datum.qp_lower, ".1~g")+")" :'
                    'format(datum.x1, ".2")'
                )
            )
        )

        texts2 = (
            base.mark_text(baseline="top", dy=5, align="right")
            .encode(text=alt.Text("text:O"), x="x_upper:Q")
            .transform_calculate(
                text=(
                    "datum.qp_upper>0?"
                    'format(datum.x2, ".2")+" ("+format(datum.qp_upper, ".1")+")" :'
                    'format(datum.x2, ".2")'
                )
            )
        )

        data = pd.DataFrame([{"start": 0, "end": 1}])
        rect = alt.Chart(data).mark_rect(opacity=0.05).encode(x="start:Q", x2="end:Q")

        # TODO:: for qp can we do something with the y encoding here and
        # connecting this to a selection?
        # seems tricky, no clear way to control the actual labels
        # or can we use the text channel identical to the above?
        nominal = (
            alt.Chart(nominal_vars)
            .mark_point()
            .encode(x="x:Q", y="name:N")
            .transform_filter(point_selector)
            .properties(width=width)
        )

        texts3 = nominal.mark_text(baseline="top", dy=5, align="center").encode(
            text="item:N"
        )

        layered = alt.layer(lines, texts1, texts2, rect, nominal, texts3)

        return chart & layered

    def resample(
        self,
        i: int | None = None,
        iterations: int = 10,
        p: float = 1 / 2,
        rng: RNGLike | SeedLike | None = None,
    ) -> pd.DataFrame:
        """Calculate resample statistics for candidate box i.

        Parameters
        ----------
        i : int, optional
        iterations : int, optional
        p : float, optional
        rng : seed or random number generator, optional


        Returns
        -------
        DataFrame

        """
        rng = np.random.default_rng(rng)

        if i is None:
            i = self._cur_box

        x = self.prim.x.loc[self.yi_initial, :]
        y = self.prim.y[self.yi_initial]

        if len(self._resampled) < iterations:
            with temporary_filter(__name__, INFO, "find_box"):
                for j in range(len(self._resampled), iterations):
                    _logger.info(f"resample {j}")
                    index = rng.choice(x.index, size=int(x.shape[0] * p), replace=False)
                    x_temp = x.loc[index, :].reset_index(drop=True)
                    y_temp = y[index]

                    box = Prim(
                        x_temp,
                        y_temp,
                        peel_alpha=self.prim.peel_alpha,
                        paste_alpha=self.prim.paste_alpha,
                    ).find_box()
                    self._resampled.append(box)

        counters = []
        for _ in range(2):
            counter = dict.fromkeys(x.columns, 0.0)
            counters.append(counter)

        coverage = self.peeling_trajectory.coverage[i]
        density = self.peeling_trajectory.density[i]

        for box in self._resampled:
            coverage_index = (box.peeling_trajectory.coverage - coverage).abs().idxmin()
            density_index = (box.peeling_trajectory.density - density).abs().idxmin()
            for counter, index in zip(counters, [coverage_index, density_index]):
                for unc in box.p_values[index]:
                    counter[unc] += 1 / iterations

        scores = (
            pd.DataFrame(
                counters,
                index=["reproduce coverage", "reproduce density"],
                columns=box.box_lim.columns,
            ).T
            * 100
        )
        return scores.sort_values(
            by=["reproduce coverage", "reproduce density"], ascending=False
        )

    def _inspect_graph(
        self,
        i: int,
        uncs: list[str],
        ticklabel_formatter: str = "{} ({})",
        boxlim_formatter: str = "{: .2g}",
        table_formatter: str = "{:.3g}",
        ax=None,
    ) ->plt.Figure:
        """Helper method for visualizing box statistics in graph form."""
        return sdutil.plot_box(
            self.box_lims[i],
            self.p_values[i],
            self.prim.box_init,
            uncs,
            self.peeling_trajectory.loc[i, ["coverage", "density"]].to_dict(),
            ax,
            ticklabel_formatter=ticklabel_formatter,
            boxlim_formatter=boxlim_formatter,
            table_formatter=table_formatter,
        )

    def update(self, box_lims: pd.DataFrame, indices: np.ndarray):
        """Update the box to the provided box limits.

        Parameters
        ----------
        box_lims: DataFrame
                  the new box_lims
        indices: ndarray
                 the indices of y that are inside the box

        """
        self.yi = indices
        self.box_lims.append(box_lims)

        # peeling trajectory
        i = self.peeling_trajectory.shape[0]
        y = self.prim.y[self.yi]
        coi = np.sum(y)

        restricted_dims = sdutil._determine_restricted_dims(
            self.box_lims[-1], self.prim.box_init
        )

        data = {
            "coverage": coi / np.sum(self.prim.y),
            "density": coi / y.shape[0],
            "mean": np.mean(y),
            "res_dim": restricted_dims.shape[0],
            "mass": y.shape[0] / self.prim.n,
            "id": i,
            "n": y.shape[0],
            "k": coi,
        }
        new_row = pd.DataFrame([data])

        self.peeling_trajectory = pd.concat(
            [self.peeling_trajectory, new_row], ignore_index=True, sort=True
        )

        # boxlims
        qp = self._calculate_quasi_p(i, restricted_dims)
        self.p_values.append(qp)
        self._cur_box = len(self.peeling_trajectory) - 1

    def show_tradeoff(
        self, cmap=mpl.cm.viridis, annotated: bool = False
    ) -> plt.Figure:  # @UndefinedVariable
        """Visualize the trade-off between coverage and density.

        Color is used to denote the number of restricted dimensions.

        Parameters
        ----------
        cmap : valid matplotlib colormap
        annotated : bool, optional. Shows point labels if True.

        Returns
        -------
        a Figure instance

        """
        return sdutil.plot_tradeoff(
            self.peeling_trajectory, cmap=cmap, annotated=annotated
        )

    def _calculate_quasi_p(self, i: int, restricted_dims: list[str]) -> dict[str, float|np.float64]:
        """Helper function for calculating quasi-p values as discussed in Bryant and Lempert (2010).

        This is a one-sided binomial test.

        Parameters
        ----------
        i : int
            the specific box in the peeling trajectory for which the
            quasi-p values are to be calculated.
        restricted_dims : list of str

        Returns
        -------
        dict

        """
        box_lim = self.box_lims[i]
        box_lim = box_lim[restricted_dims]

        # total nr. of cases in box
        Tbox = self.peeling_trajectory.loc[i, "n"]  # noqa: N806

        # total nr. of cases of interest in box
        Hbox = self.peeling_trajectory.loc[i, "k"]  # noqa: N806

        x = self.prim.x.loc[self.prim.yi_remaining, restricted_dims]
        y = self.prim.y[self.prim.yi_remaining]

        # TODO use apply on df?

        qp_values = box_lim.apply(
            calculate_qp,
            axis=0,
            result_type="expand",
            args=[x, y, Hbox, Tbox, box_lim, self.box_lims[0]],
        )

        # TODO:: this has knock on consequences
        # TODO:: elsewhere in the code (e.g. all box visualizations
        # TODO:: as well as in CART etc.)
        # qp_values = qp_values.replace(-1, np.nan)
        qp_values = qp_values.to_dict(orient="list")
        return qp_values


class RegressionPrimBox(BasePrimBox):
    """Prim box for regression version of the Prim Algorithm."""
    rmse = CurEntry(float)

    def __init__(self, prim: "BasePrim", box_lims: pd.DataFrame, indices: np.ndarray):
        """Init.

        Parameters
        ----------
        prim : Prim instance
        box_lims : DataFrame
        indices : ndarray


        """
        super().__init__(prim, box_lims, indices)
        self.update(box_lims, indices)

    @property
    def stats(self):
        return {k: getattr(self, k) for k in ["rmse", "mean", "mass", "res_dim"]}

    def _inspect_graph(
        self,
        i: int,
        uncs: list[str],
        ticklabel_formatter: str = "{} ({})",
        boxlim_formatter: str = "{: .2g}",
        table_formatter: str = "{:.3g}",
        ax=None,
    )->plt.Figure:
        """Helper method for visualizing box statistics in graph form."""
        return sdutil.plot_box(
            self.box_lims[i],
            self.p_values[i],
            self.prim.box_init,
            uncs,
            self.peeling_trajectory.loc[i, ["rmse", "mean"]].to_dict(),
            ax,
            ticklabel_formatter=ticklabel_formatter,
            boxlim_formatter=boxlim_formatter,
            table_formatter=table_formatter,
        )

    def show_pairs_scatter(
        self,
        i: int | None = None,
        dims: list[str] | None = None,
        diag: Literal["kde", "cdf", "regression"] | None = "regression",
        upper: Literal["scatter", "hexbin", "hist", "contour"] | None = "hexbin",
        lower: Literal["scatter", "hexbin", "hist", "contour"] | None = "scatter",
        fill_subplots: bool = True,
        legend=False,
    )-> sns.PairGrid:
        return super().show_pairs_scatter(
            i=i,
            dims=dims,
            diag=diag,
            upper=upper,
            lower=lower,
            fill_subplots=fill_subplots,
            legend=legend,
        )

    def _calculate_ks(self, i: int, restricted_dims: list[str]|np.ndarray) -> dict[str, float]:
        """Helper function for calculating ks values.

        The KS is based on comparing the distribution of y within the box with
        the distribution of y with a given limit removed.

        Parameters
        ----------
        i : int
            the specific box in the peeling trajectory for which the
            ks values are to be calculated.
        restricted_dims : list of str

        Returns
        -------
        dict

        """
        box_lim = self.box_lims[i]
        box_lim = box_lim[restricted_dims]

        x = self.prim.x.loc[self.prim.yi_remaining, restricted_dims]
        y = self.prim.y[self.prim.yi_remaining]
        y_with_limits = self.prim.y[self.yi]

        def calculate_ks(
            data,
            x: pd.DataFrame,
            y: np.ndarray,
            box_lim: pd.DataFrame,
            initial_boxlim: pd.DataFrame,
        ):
            """Helper function for calculating quasi p-values."""
            if data.size == 0:
                return [-1, -1]

            u = data.name
            dtype = data.dtype

            unlimited = initial_boxlim[u]

            if np.issubdtype(dtype, np.number):
                ks_values = []
                for direction, (limit, unlimit) in enumerate(zip(data, unlimited)):
                    ks = -1
                    if unlimit != limit:
                        temp_box = box_lim.copy()
                        temp_box.at[direction, u] = unlimit

                        logical = sdutil._in_box(x, temp_box)
                        y_without_limit = y[logical]

                        ks = sp.stats.kstest(y_with_limits, y_without_limit).pvalue
                    ks_values.append(ks)
            else:
                temp_box = box_lim.copy()
                temp_box.loc[:, u] = unlimited
                logical = sdutil._in_box(x, temp_box)
                y_without_limit = y[logical]

                ks = sp.stats.kstest(y_with_limits, y_without_limit).pvalue

                ks_values = [ks, -1]

            return ks_values

        ks_values = box_lim.apply(
            calculate_ks,
            axis=0,
            result_type="expand",
            args=[x, y, box_lim, self.box_lims[0]],
        )

        ks_values = ks_values.to_dict(orient="list")
        return ks_values

    def update(self, box_lims: pd.DataFrame, indices: np.ndarray):
        """Update the box to the provided box limits.

        Parameters
        ----------
        box_lims: DataFrame
                  the new box_lims
        indices: ndarray
                 the indices of y that are inside the box

        """
        self.yi = indices
        self.box_lims.append(box_lims)

        # peeling trajectory
        i = self.peeling_trajectory.shape[0]
        y = self.prim.y[self.yi]

        restricted_dims = sdutil._determine_restricted_dims(
            self.box_lims[-1], self.prim.box_init
        )

        a = np.zeros(y.shape)
        a[:] = np.mean(y)

        data = {
            "rmse": rmse(a, y),
            "mean": np.mean(y),
            "res_dim": restricted_dims.shape[0],
            "mass": y.shape[0] / self.prim.n,
            "id": i,
            "n": y.shape[0],
        }
        new_row = pd.DataFrame([data])

        self.peeling_trajectory = pd.concat(
            [self.peeling_trajectory, new_row], ignore_index=True, sort=True
        )

        # boxlims
        ks = self._calculate_ks(i, restricted_dims)
        self.p_values.append(ks)

        self._cur_box = len(self.peeling_trajectory) - 1


class BasePrim(sdutil.OutputFormatterMixin):
    """Abstract base class for the prim algorithm."""

    def __init__(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
        obj_function: Literal[
            PRIMObjectiveFunctions.LENIENT1,
            PRIMObjectiveFunctions.LENIENT2,
            PRIMObjectiveFunctions.ORIGINAL,
        ] = PRIMObjectiveFunctions.LENIENT1,
        peel_alpha: float = 0.05,
        paste_alpha: float = 0.05,
        mass_min: float = 0.05,
        update_function: str = "default",
    ):
        """Init. """
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if y.shape[0] != x.shape[0]:
            raise ValueError(f"len(y) != len(x): {y.shape[0]} != {x.shape[0]}")

        # preprocess x
        x = x.copy()
        with contextlib.suppress(KeyError):
            x.drop(columns="scenario", inplace=True)
        x = x.reset_index(drop=True)

        x_float = x.select_dtypes([np.float64, float])
        self.x_float = x_float.values
        self.x_float_colums = x_float.columns.values

        x_int = x.select_dtypes([np.int32, np.int64, int])
        self.x_int = x_int.values
        self.x_int_columns = x_int.columns.values

        self.x_numeric_columns = np.concatenate(
            [self.x_float_colums, self.x_int_columns]
        )

        x_nominal = x.select_dtypes(exclude=np.number)

        # filter out dimensions with only single value
        for column in x_nominal.columns.values:
            if np.unique(x[column]).shape == (1,):
                x = x.drop(column, axis=1)
                _logger.info(
                    f"column {column} dropped from analysis because it has only one category"
                )

        x_nominal = x.select_dtypes(exclude=np.number)
        self.x_nominal = x_nominal.values
        self.x_nominal_columns = x_nominal.columns.values

        self.n_cols = x.columns.shape[0]

        for column in self.x_nominal_columns:
            x[column] = x[column].astype("category")

        self.x = x
        self.y = y

        self._update_yi_remaining = self._update_functions[update_function]

        # store the remainder of the parameters
        self.paste_alpha = paste_alpha
        self.peel_alpha = peel_alpha
        self.mass_min = mass_min
        self.obj_func = self._obj_functions[obj_function]

        # set the indices
        self.yi = x.index.values

        # how many data points do we have
        self.n = self.y.shape[0]

        # initial box that contains all data
        self.box_init = sdutil._make_box(self.x)

        # make a list in which the identified boxes can be put
        self._boxes = []
        self._update_yi_remaining(self)
        self._prim_box_klass = None
        self._maximization = True

    @property
    def boxes(self):
        """Return all boxes."""
        boxes = [box.box_lim for box in self._boxes]

        if not boxes:
            return [self.box_init]
        return boxes

    @property
    def stats(self) -> list[dict[str, numbers.Number]]:
        """Return all stats."""
        return [box.stats for box in self._boxes]

    def find_box(self) -> BasePrimBox | None:
        """Execute one iteration of the PRIM algorithm.

        That is, find one box, starting from the current state of Prim.
        """
        # set the indices
        self._update_yi_remaining(self)

        # make boxes already found immutable
        for box in self._boxes:
            box._frozen = True

        if self.yi_remaining.shape[0] == 0:
            _logger.info("no data remaining, exiting")
            return

        self._log_progress()

        # make a new box that contains all the remaining data points
        box = self._prim_box_klass(self, self.box_init, self.yi_remaining[:])

        #  perform peeling phase
        box = self._peel(box)
        _logger.debug("peeling completed")

        # perform pasting phase
        box = self._paste(box)
        _logger.debug("pasting completed")

        _logger.info(" ".join([f"{k}: {v}," for k, v in box.stats.items()]))
        self._boxes.append(box)
        return box

    @abc.abstractmethod
    def _log_progress(self): ...

    def _update_yi_remaining_default(self):
        """Update yi_remaining."""
        # set the indices
        logical = np.ones(self.yi.shape[0], dtype=bool)
        for box in self._boxes:
            logical[box.yi] = False
        self.yi_remaining = self.yi[logical]

    def _peel(self, box:pd.DataFrame):
        """Executes the peeling phase of the PRIM algorithm.

        Delegates peeling to data type specific helper methods.

        """
        mass_old = box.yi.shape[0] / self.n

        x_float = self.x_float[box.yi]
        x_int = self.x_int[box.yi]
        x_nominal = self.x_nominal[box.yi]

        # identify all possible peels
        possible_peels = []

        for x, columns, dtype in [
            (x_float, self.x_float_colums, "float"),
            (x_int, self.x_int_columns, "int"),
            (x_nominal, self.x_nominal_columns, "object"),
        ]:
            for j, u in enumerate(columns):
                peels = self._peels[dtype](self, box, u, j, x)
                [possible_peels.append(entry) for entry in peels]

        if not possible_peels:
            # there is no peel identified, so return box
            return box

        # determine the scores for each peel in order
        # to identify the next candidate box
        scores = []
        for entry in possible_peels:
            i, box_lim = entry
            obj = self.obj_func(self, self.y[box.yi], self.y[i])
            non_res_dim = self.n_cols - sdutil._determine_nr_restricted_dims(
                box_lim, self.box_init
            )
            score = (obj, non_res_dim, box_lim, i)
            scores.append(score)

        scores.sort(key=itemgetter(0, 1), reverse=self._maximization)
        entry = scores[0]

        obj_score = entry[0]
        box_new, indices = entry[2:]

        mass_new = self.y[indices].shape[0] / self.n

        if (mass_new >= self.mass_min) & (mass_new < mass_old) & (obj_score > 0 if self._maximization else obj_score < 0):
            box.update(box_new, indices)
            return self._peel(box)
        else:
            # else return received box
            return box

    def _real_peel(self, box: BasePrimBox, u: str, j: int, x: np.ndarray):
        """Returns two candidate new boxes by peeling upper and lower limit.

        Parameters
        ----------
        box : a PrimBox instance
        u : str
            the uncertainty for which to peel
        j : int
            column for which to peel
        x : ndarray

        Returns
        -------
        tuple
            two box lims and the associated indices

        """
        peels = []
        for direction in ["upper", "lower"]:
            xj = x[:, j]

            peel_alpha = self.peel_alpha

            i = 0
            if direction == "upper":
                peel_alpha = 1 - self.peel_alpha
                i = 1

            box_peel = get_quantile(xj, peel_alpha)

            match direction:
                case "lower":
                    logical = xj >= box_peel
                    indices = box.yi[logical]
                case "upper":
                    logical = xj <= box_peel
                    indices = box.yi[logical]
            temp_box = copy.deepcopy(box.box_lims[-1])
            temp_box.loc[i, u] = box_peel
            peels.append((indices, temp_box))

        return peels

    def _discrete_peel(self, box, u, j, x):
        """Returns two candidate new boxes, peel along upper and lower dimension.

        Parameters
        ----------
        box : a PrimBox instance
        u : str
            the uncertainty for which to peel
        j : int
            column for which to peel
        x : ndarray

        Returns
        -------
        tuple
            two box lims and the associated indices

        """
        peels = []
        for direction in ["upper", "lower"]:
            peel_alpha = self.peel_alpha
            xj = x[:, j]
            box_lim = box.box_lims[-1]

            i = 0
            if direction == "upper":
                peel_alpha = 1 - self.peel_alpha
                i = 1

            box_peel = get_quantile(xj, peel_alpha)
            box_peel = int(box_peel)

            # determine logical associated with peel value
            if direction == "lower":
                if box_peel == box_lim.loc[i, u]:
                    logical = (xj > box_lim.loc[i, u]) & (xj <= box_lim.loc[i + 1, u])
                else:
                    logical = (xj >= box_peel) & (xj <= box_lim.loc[i + 1, u])
            if direction == "upper":
                if box_peel == box_lim.loc[i, u]:
                    logical = (xj < box_lim.loc[i, u]) & (xj >= box_lim.loc[i - 1, u])
                else:
                    logical = (xj <= box_peel) & (xj >= box_lim.loc[i - 1, u])

            # determine value of new limit given logical
            if xj[logical].shape[0] == 0:
                new_limit = np.max(xj) if direction == "upper" else np.min(xj)
            else:
                new_limit = (
                    np.max(xj[logical]) if direction == "upper" else np.min(xj[logical])
                )

            indices = box.yi[logical]
            temp_box = copy.deepcopy(box_lim)
            temp_box.loc[i, u] = new_limit
            peels.append((indices, temp_box))

        return peels

    def _categorical_peel(self, box, u, j, x):
        """Returns candidate new boxes for each possible removal of a single  category.

        So. if the box[u] is a categorical variable
        with 4 categories, this method will return 4 boxes.

        Parameters
        ----------
        box : a PrimBox instance
        u : str
            the uncertainty for which to peel
        j : int
            column for which to peel
        x : ndarray

        Returns
        -------
        tuple
            a list of box lims and the associated indices

        """
        entries = box.box_lims[-1].loc[0, u]

        if len(entries) > 1:
            peels = []
            for entry in entries:
                bools = []

                temp_box = box.box_lims[-1].copy()
                peel = copy.deepcopy(entries)
                peel.discard(entry)
                temp_box[u] = [peel, peel]

                if type(next(iter(entries))) not in (str, float, int, bool):
                    for element in x[:, j]:
                        if element != entry:
                            bools.append(True)
                        else:
                            bools.append(False)
                    logical = np.asarray(bools, dtype=bool)
                else:
                    logical = x[:, j] != entry
                indices = box.yi[logical]
                peels.append((indices, temp_box))
            return peels
        else:
            # no peels possible, return empty list
            return []

    def _paste(self, box:pd.DataFrame):
        """Executes the pasting phase of the PRIM.

        Delegates pasting to data type specific helper methods.
        """
        mass_old = box.yi.shape[0] / self.n

        # need to break this down by dtype
        restricted_dims = sdutil._determine_restricted_dims(
            box.box_lims[-1], self.box_init
        )
        res_dim = set(restricted_dims)

        x = self.x.loc[self.yi_remaining, :]

        # identify all possible pastes
        possible_pastes = []
        for columns, dtype in [
            (self.x_float_colums, "float"),
            (self.x_int_columns, "int"),
            (self.x_nominal_columns, "object"),
        ]:
            for _, u in enumerate(columns):
                if u not in res_dim:
                    continue
                _logger.debug(f"pasting {u}")
                pastes = self._pastes[dtype](self, box, u, x, restricted_dims)
                [possible_pastes.append(entry) for entry in pastes]
            if not possible_pastes:
                # there is no peel identified, so return box
                return box

        # determine the scores for each peel in order
        # to identify the next candidate box
        scores = []
        for entry in possible_pastes:
            i, box_lim = entry
            obj = self.obj_func(self, self.y[box.yi], self.y[i])
            non_res_dim = len(x.columns) - sdutil._determine_nr_restricted_dims(
                box_lim, self.box_init
            )
            score = (obj, non_res_dim, box_lim, i)
            scores.append(score)

        scores.sort(key=itemgetter(0, 1), reverse=self._maximization)
        entry = scores[0]
        obj_score, _, box_new, indices = entry
        mass_new = self.y[indices].shape[0] / self.n

        mean_old = np.mean(self.y[box.yi])
        mean_new = np.mean(self.y[indices])

        if (
            (mass_new >= self.mass_min)
            & (mass_new > mass_old)
            & (obj_score > 0 if self._maximization else obj_score < 0)
            & (mean_new > mean_old if self._maximization else mean_old < mean_new)
        ):
            box.update(box_new, indices)
            return self._paste(box)
        else:
            # else return received box
            return box

    def _real_paste(self, box, u, x, resdim):
        """Returns two candidate new boxes, pasted along upper and lower dimension.

        Parameters
        ----------
        box : a PrimBox instance
        u : str
            the uncertainty for which to peel
        x : ndarray


        Returns
        -------
        tuple
            two box lims and the associated indices

        """
        pastes = []
        boxlim = box.box_lims[-1]

        for i, direction in enumerate(["lower", "upper"]):
            box_paste = boxlim.copy()
            # box containing data candidate for pasting
            paste_box = boxlim.copy()

            minimum, maximum = self.box_init[u].values

            if direction == "lower":
                paste_box.loc[:, u] = minimum, box_paste.loc[0, u]

                indices = sdutil._in_box(x[resdim], paste_box[resdim])
                data = x.loc[indices, u]

                paste_value = minimum
                if data.size > 0:
                    paste_value = get_quantile(data, 1 - self.paste_alpha)

                assert paste_value <= boxlim.loc[i, u]
            else:  # direction == 'upper':
                paste_box.loc[:, u] = paste_box.loc[1, u], maximum

                indices = sdutil._in_box(x[resdim], paste_box[resdim])
                data = x.loc[indices, u]

                paste_value = maximum
                if data.size > 0:
                    paste_value = get_quantile(data, self.paste_alpha)

                assert paste_value >= box.box_lims[-1].loc[i, u]

            dtype = box_paste[u].dtype
            if dtype == np.int32:
                paste_value = np.int(paste_value)

            box_paste.loc[i, u] = paste_value
            logical = sdutil._in_box(x[resdim], box_paste[resdim])
            indices = self.yi_remaining[logical]

            pastes.append((indices, box_paste))

        return pastes

    def _categorical_paste(self, box, u, x, resdim):
        """Return a list of pastes, equal to the number of classes currently not on the box lim.

        Parameters
        ----------
        box : a PrimBox instance
        u : str
            the uncertainty for which to peel
        x : ndarray


        Returns
        -------
        tuple
            a list of box lims and the associated indices


        """
        box_lim = box.box_lims[-1]

        c_in_b = box_lim.loc[0, u]
        c_t = self.box_init.loc[0, u]

        if len(c_in_b) < len(c_t):
            pastes = []
            possible_cs = c_t - c_in_b
            for entry in possible_cs:
                paste = copy.deepcopy(c_in_b)
                paste.add(entry)

                box_paste = box_lim.copy()
                box_paste.loc[:, u] = [paste, paste]

                indices = sdutil._in_box(x[resdim], box_paste[resdim])
                indices = self.yi_remaining[indices]
                pastes.append((indices, box_paste))
            return pastes
        else:
            # no pastes possible, return empty list
            return []

    def _lenient1_obj_func(self, y_old, y_new):
        r"""The default objective function used by prim.

        Instead of the original objective function, This function can cope with
        continuous, integer, and categorical uncertainties. The basic
        idea is that the gain in mean is divided by the loss in mass.

        .. math::

            obj = \frac
                 {\text{ave} [y_{i}\mid x_{i}\in{B-b}] - \text{ave} [y\mid x\in{B}]}
                 {|n(y_{i})-n(y)|}

        where :math:`B-b` is the set of candidate new boxes, :math:`B`
        the old box and :math:`y` are the y values belonging to the old
        box. :math:`n(y_{i})` and :math:`n(y)` are the cardinality of
        :math:`y_{i}` and :math:`y` respectively. So, this objective
        function looks for the difference between  the mean of the old
        box and the new box, divided by the change in the  number of
        data points in the box. This objective function offsets a
        problem in case of categorical data where the normal objective
        function often results in boxes mainly based on the categorical
        data.

        """
        mean_old = np.mean(y_old)
        mean_new = np.mean(y_new) if y_new.shape[0] > 0 else 0

        delta = mean_new - mean_old

        obj = 0
        if mean_old != mean_new:
            if y_old.shape[0] > y_new.shape[0]:
                obj = delta / (y_old.shape[0] - y_new.shape[0])
            elif y_old.shape[0] < y_new.shape[0]:
                obj = delta / (y_new.shape[0] - y_old.shape[0])
            else:
                raise PrimException(
                    f"""mean is different {mean_old} vs {mean_new}, while shape is the same,
                                       this cannot be the case"""
                )
        return obj

    def _lenient2_obj_func(self, y_old, y_new):
        """Friedman and fisher 14.6."""
        mean_old = np.mean(y_old)
        mean_new = np.mean(y_new) if y_new.shape[0] > 0 else 0

        delta = mean_new - mean_old

        obj = 0
        if mean_old != mean_new:
            if y_old.shape == y_new.shape:
                raise PrimException(
                    f"""mean is different {mean_old} vs {mean_new}, while shape is the same,
                                       this cannot be the case"""
                )

            change_mass = abs(y_old.shape[0] - y_new.shape[0])
            mass_new = y_new.shape[0]

            obj = mass_new * delta / change_mass

        return obj

    def _original_obj_func(self, y_old, y_new):
        """The original objective function: the mean of the data inside the box."""
        if y_new.shape[0] > 0:
            return np.mean(y_new)
        else:
            return -1

    def _assert_mode(self, y, mode, update_function):
        if mode == sdutil.RuleInductionType.BINARY:
            return set(np.unique(y)) == {0, 1}
        return False if update_function == "guivarch" else True  # noqa: SIM211

    def _assert_dtypes(self, keys, dtypes):
        """Helper function that checks whether none of the provided keys has a dtype object as value."""
        for key in keys:
            if dtypes[key][0] == np.dtype(object):
                raise EMAError(f"{key} has dtype object and can thus not be rotated")
        return True

    _peels = {"object": _categorical_peel, "int": _discrete_peel, "float": _real_peel}

    _pastes = {"object": _categorical_paste, "int": _real_paste, "float": _real_paste}

    # dict with the various objective functions available
    # todo:: move functions themselves to ENUM?
    _obj_functions = {
        PRIMObjectiveFunctions.LENIENT2: _lenient2_obj_func,
        PRIMObjectiveFunctions.LENIENT1: _lenient1_obj_func,
        PRIMObjectiveFunctions.ORIGINAL: _original_obj_func,
    }


class Prim(BasePrim):
    """Patient rule induction algorithm.

    The implementation of Prim is tailored to interactive use in the
    context of scenario discovery.

    Parameters
    ----------
    x : DataFrame
        the independent variables
    y : 1d ndarray
        the dependent variable
    obj_function : {LENIENT1, LENIENT2, ORIGINAL}
                   the objective function used by PRIM. Defaults to a
                   lenient objective function based on the gain of mean
                   divided by the loss of mass.
    peel_alpha : float, optional
                 parameter controlling the peeling stage (default = 0.05).
    paste_alpha : float, optional
                  parameter controlling the pasting stage (default = 0.05).
    mass_min : float, optional
               minimum mass of a box (default = 0.05).
    update_function = {'default', 'guivarch'}, optional
                      controls behavior of PRIM after having found a
                      first box. use either the default behavior were
                      all points are removed, or the procedure
                      suggested by Guivarch et al (2016)
                      doi:10.1016/j.envsoft.2016.03.006 to simply set
                      all points to be no longer of interest (only
                      valid in binary mode).

    See Also
    --------
    :mod:`cart`


    """

    def __init__(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
        obj_function: Literal[
            PRIMObjectiveFunctions.LENIENT1,
            PRIMObjectiveFunctions.LENIENT2,
            PRIMObjectiveFunctions.ORIGINAL,
        ] = PRIMObjectiveFunctions.LENIENT1,
        peel_alpha: float = 0.05,
        paste_alpha: float = 0.05,
        mass_min: float = 0.05,
        update_function: Literal["default", "guivarch"] = "default",
    ):
        """Init."""
        super().__init__(
            x,
            y,
            obj_function=obj_function,
            peel_alpha=peel_alpha,
            paste_alpha=paste_alpha,
            mass_min=mass_min,
            update_function=update_function,
        )
        self._prim_box_klass = PrimBox

        # how many cases of interest do we have?
        self.t_coi = np.sum(y)

    def _log_progress(self):
        """Helper method to log progress."""
        _logger.info(
            f"{self.yi_remaining.shape[0]} points remaining, containing {np.sum(self.yi_remaining)} cases of interest"
        )

    def _update_yi_remaining_guivarch(self):
        """Update yi_remaining.

        Used the modified version from Guivarch et al (2016) doi:10.1016/j.envsoft.2016.03.006

        """
        # set the indices
        for box in self._boxes:
            self.y[box.yi] = 0

        self.yi_remaining = self.yi

    _update_functions = {
        "default": BasePrim._update_yi_remaining_default,
        "guivarch": _update_yi_remaining_guivarch,
    }


class RegressionPrim(BasePrim):
    """Prim for regression.

    Parameters
    ----------
    x : DataFrame
    the independent variables
    y : 1d ndarray
        the dependent variable
    obj_function : {LENIENT1, LENIENT2, ORIGINAL}
    peel_alpha : float, optional
                 parameter controlling the peeling stage (default = 0.05).
    paste_alpha : float, optional
                  parameter controlling the pasting stage (default = 0.05).
    mass_min : float, optional
                minimum mass of a box (default = 0.05).
    update_function : {'default', 'guivarch'}, optional
                    controls behavior of PRIM after having found a first box. use either the
                    default behavior were the all points are removed, or the procedure
                    suggested by guivarch et al (2016)
    maximization: bool

    """

    def __init__(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
        obj_function: Literal[
            PRIMObjectiveFunctions.LENIENT1,
            PRIMObjectiveFunctions.LENIENT2,
            PRIMObjectiveFunctions.ORIGINAL,
        ] = PRIMObjectiveFunctions.LENIENT1,
        peel_alpha: float = 0.05,
        paste_alpha: float = 0.05,
        mass_min: float = 0.05,
        update_function: str = "default",
        maximization: bool = True,
    ):
        """Init."""
        # is there a meaning of guivarch in regression mode
        super().__init__(
            x,
            y,
            obj_function=obj_function,
            peel_alpha=peel_alpha,
            paste_alpha=paste_alpha,
            mass_min=mass_min,
            update_function=update_function,
        )

        self._prim_box_klass = RegressionPrimBox
        self._maximization = maximization # fixme this is not working correctly

    def _log_progress(self):
        """Helper method to log progress."""
        _logger.info(
            f"{self.yi_remaining.shape[0]} points remaining, with mean of {np.mean(self.yi_remaining)}"
        )

    _update_functions = {"default": BasePrim._update_yi_remaining_default}
