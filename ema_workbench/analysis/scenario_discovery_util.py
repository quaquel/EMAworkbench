"""Scenario discovery utilities used by both :mod:`cart` and :mod:`prim`."""

import abc
import enum
import itertools
from typing import Literal

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from mpl_toolkits.axes_grid1 import host_subplot  # @UnresolvedImports

from .plotting_util import COLOR_LIST, make_legend

# Created on May 24, 2015
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ["RuleInductionType"]


class RuleInductionType(enum.Enum):
    """Enum for types of rule induction."""

    REGRESSION = "regression"
    """constant indicating regression mode"""

    BINARY = "binary"
    """constant indicating binary classification mode. This is the most
    common used mode in scenario discovery"""

    CLASSIFICATION = "classification"
    """constant indicating classification mode"""


def _get_sorted_box_lims(boxes, box_init):
    """Sort the uncertainties for each box in boxes based on a normalization given box_init.

    Unrestricted dimensions are dropped. The sorting is based on the normalization
    of the first box in boxes.

    Parameters
    ----------
    boxes : list of DataFrames
    box_init : DataFrmae

    Returns
    -------
    tuple
        with the sorted boxes, and the list of restricted uncertainties

    """
    # determine the uncertainties that are being restricted
    # in one or more boxes
    uncs = set()
    for box in boxes:
        us = _determine_restricted_dims(box, box_init)
        uncs = uncs.union(us)
    uncs = np.asarray(list(uncs))

    # normalize the range for the first box
    box_lim = boxes[0]
    nbl = _normalize(box_lim, box_init, uncs)
    box_size = nbl[:, 1] - nbl[:, 0]

    # sort the uncertainties based on the normalized size of the
    # restricted dimensions
    uncs = uncs[np.argsort(box_size)]
    box_lims = list(boxes)

    return box_lims, uncs.tolist()


def _make_box(x):
    """Make a box that encompasses all the data.

    Parameters
    ----------
    x : DataFrame

    Returns
    -------
    DataFrame


    """
    # x.select_dtypes(np.number)

    def limits(x):
        if pd.api.types.is_numeric_dtype(x.dtype):  # @UndefinedVariable
            return pd.Series([x.min(), x.max()])
        else:
            return pd.Series([set(x), set(x)])

    return x.apply(limits)


def _normalize(box_lim, box_init, uncertainties):
    """Normalize the given box lim to the unit interval.

    The limits are  derived from box init for the specified uncertainties.

    Categorical uncertainties are normalized based on fractionated. So
    value specifies the fraction of categories in the box_lim.

    Parameters
    ----------
    box_lim : DataFrame
    box_init :  DataFrame
    uncertainties : list of strings
                    valid names of columns that exist in both DataFrames

    Returns
    -------
    ndarray
        a numpy array of the shape (2, len(uncertainties) with the
        normalized box limits.


    """
    # normalize the range for the first box
    norm_box_lim = np.zeros((len(uncertainties), box_lim.shape[0]))

    for i, u in enumerate(uncertainties):
        dtype = box_lim[u].dtype
        if dtype == np.dtype(object):
            nu = len(box_lim.loc[0, u]) / len(box_init.loc[0, u])
            nl = 0
        else:
            lower, upper = box_lim.loc[:, u]
            dif = box_init.loc[1, u] - box_init.loc[0, u]
            a = 1 / dif
            b = -1 * box_init.loc[0, u] / dif
            nl = a * lower + b
            nu = a * upper + b
        norm_box_lim[i, :] = nl, nu
    return norm_box_lim


def _determine_restricted_dims(
    box_limits: pd.DataFrame, box_init: pd.DataFrame
) -> np.ndarray:
    """Returns a list of dimensions that is restricted.

    Parameters
    ----------
    box_limits : pd.DataFrame
    box_init : pd.DataFrame

    Returns
    -------
    np.ndarray

    """
    cols = box_init.columns.values
    restricted_dims = cols[
        np.logical_not(np.all(box_init.values == box_limits.values, axis=0))
    ]
    #     restricted_dims = [column for column in box_init.columns if not
    #            np.all(box_init[column].values == box_limits[column].values)]
    return restricted_dims


def _determine_nr_restricted_dims(
    box_lims: pd.DataFrame, box_init: pd.DataFrame
) -> int:
    """Determine the number of restricted dimensions of a box.

    Parameters
    ----------
    box_lims : DataFrame
               a specific box limit
    box_init : DataFrame
               the initial box containing all data points


    Returns
    -------
    int

    """
    return _determine_restricted_dims(box_lims, box_init).shape[0]


def _compare(a, b):
    """Compare two boxes to see which restrictions are the same or not."""
    dtypes_desc = a.dtype.descr
    logical = np.ones((len(dtypes_desc)), dtype=bool)
    for i, entry in enumerate(dtypes_desc):
        name = entry[0]
        logical[i] = (
            logical[i] & (a[name][0] == b[name][0]) & (a[name][1] == b[name][1])
        )
    return logical


def _in_box(x, boxlim):
    """Check which data is inside the boxlims.

    Returns the boolean index indicated which data points are inside
    and which are outside the given box_lims

    Parameters
    ----------
    x : pd.DataFrame
    boxlim : pd.DataFrame

    Returns
    -------
    ndarray
        boolean 1D array

    Raises
    ------
    Attribute error if not numbered columns are not pandas
    category dtype

    """
    x_numbered = x.select_dtypes(np.number)
    boxlim_numbered = boxlim.select_dtypes(np.number)
    logical = (boxlim_numbered.loc[0, :].values <= x_numbered.values) & (
        x_numbered.values <= boxlim_numbered.loc[1, :].values
    )
    logical = logical.all(axis=1)

    # TODO:: how to speed this up
    for column, values in x.select_dtypes(exclude=np.number).items():
        entries = boxlim.loc[0, column]
        if values.dtype == np.dtype(np.bool):
            l = x[column] == entries
            logical = logical & l
        else:
            not_present  = set(values.cat.categories.values) - entries

            if not_present:
                # what other options do we have here....
                l = pd.isnull(x[column].cat.remove_categories(list(entries)))  # noqa: E741
                logical = l & logical
    return logical


def _calculate_quasip(x, y, box, Hbox, Tbox):  # noqa: N803
    """Calculate quasi-p values.

    Parameters
    ----------
    x : DataFrame
    y : np.array
    box : DataFrame
    Hbox : int
    Tbox : int

    """
    logical = _in_box(x, box)
    yi = y[logical]

    # total nr. of cases in box with one restriction removed
    Tj = yi.shape[0]  # noqa: N806

    # total nr. of cases of interest in box with one restriction
    # removed
    Hj = np.sum(yi)  # noqa: N806

    p = Hj / Tj

    Hbox = int(Hbox)  # noqa: N806
    Tbox = int(Tbox)  # noqa: N806

    # force one sided
    qp = sp.stats.binomtest(Hbox, Tbox, p, alternative="greater")  # @UndefinedVariable

    return qp.pvalue


def plot_pair_wise_scatter(
    x: pd.DataFrame,
    y: np.array,
    boxlim: pd.DataFrame,
    box_init: pd.DataFrame,
    restricted_dims: list[str],
    diag: Literal["kde", "cdf", "regression"] | None = "kde",
    upper: Literal["scatter", "hexbin", "hist", "contour"] | None = "scatter",
    lower: Literal["scatter", "hexbin", "hist", "contour"] | None = "hist",
    fill_subplots=True,
    legend=True,
)-> sns.PairGrid:
    """Helper function for pair wise scatter plotting.

    Parameters
    ----------
    x : DataFrame
        the experiments
    y : numpy array
        the outcome of interest
    box_lim : DataFrame
              a boxlim
    box_init : DataFrame
    restricted_dims : collection of strings
                      list of uncertainties that define the boxlims
    diag : string, optional
           Plot diagonal as kernel density estimate ('kde'),
           cumulative density function ('cdf'), or regression ('regression')
    upper, lower: string, optional
           Use either 'scatter', 'contour', hexbin, or 'hist' (bivariate
           histogram) plots for upper and lower triangles. Upper triangle
           can also be 'none' to eliminate redundancy. Legend uses
           lower triangle style for markers.
    fill_subplots: Boolean, optional
                   if True, subplots are resized to fill their respective axes.
                   This removes unnecessary whitespace, but may be undesirable
                   for some variable combinations.
    legend: Boolean, optional

    """
    x = x[restricted_dims]
    data = x.copy()

    categorical_columns = data.select_dtypes("category").columns.values
    categorical_mappings = {}
    for column in categorical_columns:
        # reorder categorical data so we
        # can capture the categories that are part of the box within a
        # single rectangular patch
        categories_inbox = boxlim.at[0, column]
        categories_all = box_init.at[0, column]
        missing = categories_all - categories_inbox
        categories = list(categories_inbox) + list(missing)

        data[column] = data[column].cat.set_categories(categories)

        # keep the mapping for updating ticklabels
        categorical_mappings[column] = dict(enumerate(data[column].cat.categories))

        # replace column with codes
        data[column] = data[column].cat.codes

    # add outcome of interest to DataFrame
    data["y"] = y

    # ensures cases of interest are plotted on top
    data.sort_values("y", inplace=True)

    # main plot body

    grid = sns.PairGrid(
        data=data, hue="y", vars=x.columns.values, diag_sharey=False
    )  # enables different plots in upper and lower triangles

    def _plot_triangle(which, style):
        func = grid.map_upper if which == "upper" else grid.map_lower

        # upper triangle
        match style:
            case "hexbin":

                def hexbin(x, y, *args, hue=None, **kwargs):
                    a = pd.DataFrame({"x": x, "y": y, "z": data.y})
                    a.plot.hexbin(
                        "x",
                        "y",
                        C="z",
                        reduce_C_function=np.mean,
                        ax=plt.gca(),
                        gridsize=10,
                        colorbar=False,
                        cmap=sns.cubehelix_palette(as_cmap=True),
                        norm=colors.Normalize(vmin=data.y.min(), vmax=data.y.max()),
                    )

                # draw contours twice to get different fill and line alphas, more interpretable
                func(hexbin, hue=None)
            case "contour":
                # draw contours twice to get different fill and line alphas, more interpretable
                func(
                    sns.kdeplot,
                    fill=True,
                    alpha=0.8,
                    bw_adjust=1.2,
                    levels=5,
                    common_norm=False,
                    cut=0,
                )  # cut = 0
                func(
                    sns.kdeplot,
                    fill=False,
                    alpha=1,
                    bw_adjust=1.2,
                    levels=5,
                    common_norm=False,
                    cut=0,
                )
            case "hist":
                func(sns.histplot)
            case "scatter":
                func(sns.scatterplot)
            case "None":
                pass
            case _:
                raise NotImplementedError(
                    f"upper = {upper} not implemented. Use either 'scatter', 'contour', 'hist' (bivariate histogram) or None plots for upper triangle."
                )

    _plot_triangle("upper", upper)
    _plot_triangle("lower", lower)

    # diagonal
    match diag:
        case "regression":

            def my_custom_func(x, *args, hue=None, **kwargs):
                ax = plt.gca()
                sns.scatterplot(x=x, y=data.y, hue=data.y, palette=grid._orig_palette)
                sns.regplot(x=x, y=data.y, ax=ax, scatter=False)

            grid.map_diag(my_custom_func, hue=None)
        case "cdf":
            grid.map_diag(sns.ecdfplot)
        case "kde":
            grid.map_diag(sns.scatterplot, y=data.y)
        case _:
            raise NotImplementedError(
                f"diag = {diag} not implemented. Use either 'kde' (kernel density estimate) or 'cdf' (cumulative density function)."
            )

    # draw box
    pad = 0.1

    cats = set(categorical_columns)
    for row, ylabel in zip(grid.axes, grid.y_vars):
        for ax, xlabel in zip(row, grid.x_vars):
            if ylabel == xlabel:
                continue

            xrange = ax.get_xlim()[1] - ax.get_xlim()[0]
            yrange = ax.get_ylim()[1] - ax.get_ylim()[0]

            ylim = boxlim[ylabel]

            if ylabel in cats:
                height = (len(ylim[0]) - 1) + pad * yrange
                y = -yrange * pad / 2
            else:
                y = ylim[0]
                height = ylim[1] - ylim[0]

            if xlabel in cats:
                xlim = boxlim.at[0, xlabel]
                width = (len(xlim) - 1) + pad * xrange
                x = -xrange * pad / 2
            else:
                xlim = boxlim[xlabel]
                x = xlim[0]
                width = xlim[1] - xlim[0]

            xy = x, y
            box = patches.Rectangle(
                xy, width, height, edgecolor="red", facecolor="none", lw=3, zorder=100
            )
            if ax.has_data():  # keeps box from being drawn in upper triangle if empty
                ax.add_patch(box)
            else:
                ax.set_axis_off()

    # do the yticklabeling for categorical rows
    for row, ylabel in zip(grid.axes, grid.y_vars):
        if ylabel in cats:
            ax = row[0]
            labels = []
            locs = []
            mapping = categorical_mappings[ylabel]
            for i in range(-1, len(mapping) + 1):
                locs.append(i)
                try:
                    label = categorical_mappings[ylabel][i]
                except KeyError:
                    label = ""
                labels.append(label)
            ax.set_yticks(locs)
            ax.set_yticklabels(labels)

    # do the xticklabeling for categorical columns
    for ax, xlabel in zip(grid.axes[-1], grid.x_vars):
        if xlabel in cats:
            labels = []
            locs = []
            mapping = categorical_mappings[xlabel]
            for i in range(-1, len(mapping) + 1):
                locs.append(i)
                try:
                    label = categorical_mappings[xlabel][i]
                except KeyError:
                    label = ""
                labels.append(label)
            ax.set_xticks(locs)
            ax.set_xticklabels(labels, rotation=90)

    # fit subplot to data ranges, with some padding for aesthetics
    if fill_subplots:
        # for row, ylabel in zip(grid.axes, grid.y_vars):
        #     for ax, xlabel in zip(row, grid.x_vars):
        for row, ylabel in zip(grid.axes, grid.y_vars):
            for subplot, xlabel in zip(row, grid.x_vars):
                if xlabel != "":
                    upper = data[xlabel].max()
                    lower = data[xlabel].min()

                    pad_rel = (
                        upper - lower
                    ) * 0.1  # padding relative to range of data points

                    subplot.set_xlim(lower - pad_rel, upper + pad_rel)

                if ylabel != "":
                    upper = data[ylabel].max()
                    lower = data[ylabel].min()

                    pad_rel = (
                        upper - lower
                    ) * 0.1  # padding relative to range of data points

                    subplot.set_ylim(lower - pad_rel, upper + pad_rel)
    if legend:
        grid.add_legend()

    return grid


def _setup_figure(uncs, ax):
    """Helper function for creating the basic layout for the figures that show the box lims.

    Parameters
    ----------
    uncs : list of str
    ax : axes instance

    """
    nr_unc = len(uncs)

    # create the shaded grey background
    rect = mpl.patches.Rectangle(
        (0, -0.5), 1, nr_unc + 1.5, alpha=0.25, facecolor="#C0C0C0", edgecolor="#C0C0C0"
    )
    ax.add_patch(rect)
    ax.set_xlim(left=-0.2, right=1.2)
    ax.set_ylim(top=-0.5, bottom=nr_unc - 0.5)
    ax.yaxis.set_ticks(list(range(nr_unc)))
    ax.xaxis.set_ticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(uncs[::-1])


def plot_box(
    boxlim: pd.DataFrame,
    p_values: dict,
    box_init: pd.DataFrame,
    uncs: list[str],
    stats: dict,
    ax: plt.Axes,
    ticklabel_formatter: str = "{} ({})",
    boxlim_formatter: str = "{: .2g}",
    table_formatter: str = "{:.3g}",
) -> plt.Figure:
    """Helper function for parallel coordinate style visualization of a box.

    Parameters
    ----------
    boxlim : DataFrame
    p_values : dict
    box_init : DataFrame
    uncs : list
    stats: dict
    ticklabel_formatter : str
    boxlim_formatter : str
    table_formatter : str
    ax : Axes instance

    Returns
    -------
    a Figure instance


    """
    norm_box_lim = _normalize(boxlim, box_init, uncs)
    fig = plt.gcf()

    _setup_figure(uncs, ax)
    for j, u in enumerate(uncs):
        # we want to have the most restricted dimension
        # at the top of the figure
        xj = len(uncs) - j - 1

        plot_unc(box_init, xj, j, 0, norm_box_lim, boxlim, u, ax)

        # new part
        dtype = box_init[u].dtype

        props = {"facecolor": "white", "edgecolor": "white", "alpha": 0.25}
        y = xj

        # fixme don't know how to fix this ruff issue
        if dtype == object:  # noqa: E721
            elements = sorted(box_init[u][0])
            max_value = len(elements) - 1
            values = boxlim.loc[0, u]
            x = [elements.index(entry) for entry in values]
            x = [entry / max_value for entry in x]

            for xi, label in zip(x, values):
                ax.text(
                    xi,
                    y - 0.2,
                    label,
                    ha="center",
                    va="center",
                    bbox=props,
                    color="blue",
                    fontweight="normal",
                )

        else:
            props = {"facecolor": "white", "edgecolor": "white", "alpha": 0.25}

            # plot limit text labels
            x = norm_box_lim[j, 0]

            if not np.allclose(x, 0):
                label = boxlim_formatter.format(boxlim.loc[0, u])
                ax.text(
                    x,
                    y - 0.2,
                    label,
                    ha="center",
                    va="center",
                    bbox=props,
                    color="blue",
                    fontweight="normal",
                )

            x = norm_box_lim[j][1]
            if not np.allclose(x, 1):
                label = boxlim_formatter.format(boxlim.loc[1, u])
                ax.text(
                    x,
                    y - 0.2,
                    label,
                    ha="center",
                    va="center",
                    bbox=props,
                    color="blue",
                    fontweight="normal",
                )

            # plot uncertainty space text labels
            x = 0
            label = boxlim_formatter.format(box_init.loc[0, u])
            ax.text(
                x - 0.01,
                y,
                label,
                ha="right",
                va="center",
                bbox=props,
                color="black",
                fontweight="normal",
            )

            x = 1
            label = boxlim_formatter.format(box_init.loc[1, u])
            ax.text(
                x + 0.01,
                y,
                label,
                ha="left",
                va="center",
                bbox=props,
                color="black",
                fontweight="normal",
            )

        # set y labels
        qp_formatted = {}
        for key, values in p_values.items():
            values = [vi for vi in values if vi != -1]  # noqa: PLW2901

            if len(values) == 1:
                value = f"{values[0]:.2g}"
            else:
                value = "{:.2g}, {:.2g}".format(*values)
            qp_formatted[key] = value

        labels = [ticklabel_formatter.format(u, qp_formatted[u]) for u in uncs]

        labels = labels[::-1]
        ax.set_yticklabels(labels)

        # remove x tick labels
        ax.set_xticklabels([])

    cell_text = []
    for v in stats.values():
        cell_text.append([table_formatter.format(v)])

    # add table to the left
    ax.table(
        cellText=cell_text,
        colWidths=[0.1] * len(cell_text),
        rowLabels=list(stats.keys()),
        colLabels=None,
        loc="right",
        bbox=[1.2, 0.9, 0.1, 0.1],
    )
    plt.subplots_adjust(left=0.1, right=0.75)

    return fig


def plot_ppt(peeling_trajectory):
    """Show the peeling and pasting trajectory in a figure."""
    ax = host_subplot(111)
    ax.set_xlabel("peeling and pasting trajectory")

    par = ax.twinx()
    par.set_ylabel("nr. restricted dimensions")

    ax.plot(peeling_trajectory["mean"], label="mean")
    ax.plot(peeling_trajectory["mass"], label="mass")
    ax.plot(peeling_trajectory["coverage"], label="coverage")
    ax.plot(peeling_trajectory["density"], label="density")
    par.plot(peeling_trajectory["res_dim"], label="restricted dims")
    ax.grid(True, which="both")
    ax.set_ylim(bottom=0, top=1)

    fig = plt.gcf()

    make_legend(
        ["mean", "mass", "coverage", "density", "restricted_dim"], ax, ncol=5, alpha=1
    )
    return fig


def plot_tradeoff(
    peeling_trajectory, cmap=mpl.cm.viridis, annotated=False
):  # @UndefinedVariable
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
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="equal")

    boundaries = np.arange(-0.5, max(peeling_trajectory["res_dim"]) + 1.5, step=1)
    ncolors = cmap.N
    norm = mpl.colors.BoundaryNorm(boundaries, ncolors)

    p = ax.scatter(
        peeling_trajectory["coverage"],
        peeling_trajectory["density"],
        c=peeling_trajectory["res_dim"],
        norm=norm,
        cmap=cmap,
    )
    ax.set_ylabel("density")
    ax.set_xlabel("coverage")
    ax.set_ylim(bottom=0, top=1.2)
    ax.set_xlim(left=0, right=1.2)

    if annotated:
        for _, row in peeling_trajectory.iterrows():
            ax.annotate(row["id"], (row["coverage"], row["density"]))

    ticklocs = np.arange(0, max(peeling_trajectory["res_dim"]) + 1, step=1)
    cb = fig.colorbar(p, spacing="uniform", ticks=ticklocs, drawedges=True)
    cb.set_label("nr. of restricted dimensions")

    return fig


def plot_unc(box_init, xi, i, j, norm_box_lim, box_lim, u, ax, color=None):
    """Plot a given uncertainty.

    Parameters
    ----------
    xi : int
         the row at which to plot
    i : int
        the index of the uncertainty being plotted
    j : int
        the index of the box being plotted
    u : string
        the uncertainty being plotted:
    ax : axes instance
         the ax on which to plot
    color : optional, valid mpl color

    """
    if color is None:
        color = sns.color_palette()[0]

    dtype = box_init[u].dtype

    y = xi - j * 0.1

    if dtype is object:
        elements = sorted(box_init[u][0])
        max_value = len(elements) - 1
        box_lim = box_lim[u][0]
        x = [elements.index(entry) for entry in box_lim]
        x = [entry / max_value for entry in x]
        y = [y] * len(x)

        ax.scatter(x, y, edgecolor=color, facecolor=color)

    else:
        ax.plot(norm_box_lim[i], (y, y), c=color)


def plot_boxes(x, boxes, together):
    """Helper function for plotting multiple boxlims.

    Parameters
    ----------
    x : pd.DataFrame
    boxes : list of pd.DataFrame
    together : bool

    """
    box_init = _make_box(x)
    box_lims, uncs = _get_sorted_box_lims(boxes, box_init)

    # normalize the box lims
    # we don't need to show the last box, for this is the
    # box_init, which is visualized by a grey area in this
    # plot.
    norm_box_lims = [_normalize(box_lim, box_init, uncs) for box_lim in boxes]

    if together:
        fig, ax = plt.subplots()
        _setup_figure(uncs, ax)

        for i, u in enumerate(uncs):
            colors = itertools.cycle(COLOR_LIST)
            # we want to have the most restricted dimension
            # at the top of the figure

            xi = len(uncs) - i - 1

            for j, norm_box_lim in enumerate(norm_box_lims):
                color = next(colors)
                plot_unc(box_init, xi, i, j, norm_box_lim, box_lims[j], u, ax, color)

        plt.tight_layout()
        return fig
    else:
        figs = []
        colors = itertools.cycle(COLOR_LIST)

        for j, norm_box_lim in enumerate(norm_box_lims):
            fig, ax = plt.subplots()
            _setup_figure(uncs, ax)
            ax.set_title(f"box {j}")
            color = next(colors)

            figs.append(fig)
            for i, u in enumerate(uncs):
                xi = len(uncs) - i - 1
                plot_unc(box_init, xi, i, 0, norm_box_lim, box_lims[j], u, ax, color)

            plt.tight_layout()
        return figs


class OutputFormatterMixin(abc.ABC):
    """Formatter mixin class."""

    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def boxes(self):
        """Property for getting a list of box limits."""

    @property
    @abc.abstractmethod
    def stats(self):
        """Property for getting a list of dicts containing the statistics for each box."""

    def boxes_to_dataframe(self):
        """Convert boxes to pandas dataframe."""
        boxes = self.boxes

        # determine the restricted dimensions
        # print only the restricted dimension
        box_lims, uncs = _get_sorted_box_lims(boxes, _make_box(self.x))
        nr_boxes = len(boxes)
        dtype = float
        index = [f"box {i + 1}" for i in range(nr_boxes)]
        for value in box_lims[0].dtypes:
            # fixme don't know how to fix this ruff issue
            if value == object:  # noqa E721
                dtype = object
                break

        columns = pd.MultiIndex.from_product([index, ["min", "max"]])
        df_boxes = pd.DataFrame(
            np.zeros((len(uncs), nr_boxes * 2)),
            index=uncs,
            dtype=dtype,
            columns=columns,
        )

        # TODO should be possible to make more efficient
        for i, box in enumerate(box_lims):
            for unc in uncs:
                values = box.loc[:, unc]
                values = values.rename({0: "min", 1: "max"})
                df_boxes.loc[unc, index[i]] = values.values
        return df_boxes

    def stats_to_dataframe(self):
        """Convert stats to pandas dataframe."""
        stats = self.stats

        index = pd.Index([f"box {i + 1}" for i in range(len(stats))])

        return pd.DataFrame(stats, index=index)

    def show_boxes(self, together=False):
        """Display boxes.

        Parameters
        ----------
        together : bool, otional

        """
        plot_boxes(self.x, self.boxes, together=together)
