'''
Scenario discovery utilities used by both :mod:`cart` and :mod:`prim`
'''
import abc
import enum
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import host_subplot  # @UnresolvedImports
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns

from .plotting_util import COLOR_LIST, make_legend

# Created on May 24, 2015
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ["RuleInductionType"]


class RuleInductionType(enum.Enum):
    REGRESSION = 'regression'
    '''constant indicating regression mode'''

    BINARY = 'binary'
    '''constant indicating binary classification mode. This is the most
    common used mode in scenario discovery'''

    CLASSIFICATION = 'classification'
    '''constant indicating classification mode'''


def _get_sorted_box_lims(boxes, box_init):
    '''Sort the uncertainties for each box in boxes based on a
    normalization given box_init. Unrestricted dimensions are dropped.
    The sorting is based on the normalization of the first box in boxes.

    Parameters
    ----------
    boxes : list of numpy structured arrays
    box_init : numpy structured array

    Returns
    -------
    tuple
        with the sorted boxes, and the list of restricted uncertainties

    '''

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
    box_lims = [box for box in boxes]

    return box_lims, uncs.tolist()


def _make_box(x):
    '''
    Make a box that encompasses all the data

    Parameters
    ----------
    x : DataFrame

    Returns
    -------
    DataFrame


    '''

    # x.select_dtypes(np.number)

    def limits(x):
        if (pd.api.types.is_numeric_dtype(x.dtype)):  # @UndefinedVariable
            return pd.Series([x.min(), x.max()])
        else:
            return pd.Series([set(x), set(x)])

    return x.apply(limits)


def _normalize(box_lim, box_init, uncertainties):
    '''Normalize the given box lim to the unit interval derived
    from box init for the specified uncertainties.

    Categorical uncertainties are normalized based on fractionated. So
    value specifies the fraction of categories in the box_lim.

    Parameters
    ----------
    box_lim : DataFrame
    box_init :  DataFrame
    uncertainties : list of strings
                    valid names of columns that exist in both structured
                    arrays.

    Returns
    -------
    ndarray
        a numpy array of the shape (2, len(uncertainties) with the
        normalized box limits.


    '''

    # normalize the range for the first box
    norm_box_lim = np.zeros((len(uncertainties), box_lim.shape[0]))

    for i, u in enumerate(uncertainties):
        dtype = box_lim[u].dtype
        if dtype == np.dtype(object):
            nu = len(box_lim.loc[0, u]) / len(box_init.loc[0, u])
            nl = 0
        else:
            lower, upper = box_lim.loc[:, u]
            dif = (box_init.loc[1, u] - box_init.loc[0, u])
            a = 1 / dif
            b = -1 * box_init.loc[0, u] / dif
            nl = a * lower + b
            nu = a * upper + b
        norm_box_lim[i, :] = nl, nu
    return norm_box_lim


def _determine_restricted_dims(box_limits, box_init):
    '''returns a list of dimensions that is restricted

    Parameters
    ----------
    box_limits : pd.DataFrame
    box_init : pd.DataFrame

    Returns
    -------
    list of str

    '''
    cols = box_init.columns.values
    restricted_dims = cols[np.all(
        box_init.values == box_limits.values, axis=0) == False]
#     restricted_dims = [column for column in box_init.columns if not
#            np.all(box_init[column].values == box_limits[column].values)]
    return restricted_dims


def _determine_nr_restricted_dims(box_lims, box_init):
    '''

    determine the number of restriced dimensions of a box given
    compared to the inital box that contains all the data

    Parameters
    ----------
    box_lims : structured numpy array
               a specific box limit
    box_init : structured numpy array
               the initial box containing all data points


    Returns
    -------
    int

    '''

    return _determine_restricted_dims(box_lims, box_init).shape[0]


def _compare(a, b):
    '''compare two boxes, for each dimension return True if the
    same and false otherwise'''
    dtypesDesc = a.dtype.descr
    logical = np.ones((len(dtypesDesc,)), dtype=np.bool)
    for i, entry in enumerate(dtypesDesc):
        name = entry[0]
        logical[i] = logical[i] &\
            (a[name][0] == b[name][0]) &\
            (a[name][1] == b[name][1])
    return logical


def _in_box(x, boxlim):
    '''

    returns the a boolean index indicated which data points are inside
    and which are outside of the given box_lims

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

    '''

    x_numbered = x.select_dtypes(np.number)
    boxlim_numbered = boxlim.select_dtypes(np.number)
    logical = (boxlim_numbered.loc[0, :].values <= x_numbered.values) &\
        (x_numbered.values <= boxlim_numbered.loc[1, :].values)
    logical = logical.all(axis=1)

    # TODO:: how to speed this up
    for column, values in x.select_dtypes(exclude=np.number).iteritems():
        entries = boxlim.loc[0, column]
        not_present = set(values.cat.categories.values) - entries

        if not_present:
            # what other options do we have here....
            l = pd.isnull(x[column].cat.remove_categories(list(entries)))
            logical = l & logical
    return logical


def _setup(results, classify, incl_unc=[]):
    """helper function for setting up CART or PRIM

    Parameters
    ----------
    results : tuple of DataFrame and dict with numpy arrays
              the return from :meth:`perform_experiments`.
    classify : string, function or callable
               either a string denoting the outcome of interest to
               use or a function.
    incl_unc : list of strings

    Notes
    -----
    CART, PRIM, and feature scoring only work for a 1D numpy array
    for the dependent variable

    Raises
    ------
    TypeError
        if classify is not a string or a callable.

    """
    x, outcomes = results

    if incl_unc:
        drop_names = set(x.columns.values.tolist()) - set(incl_unc)
        x = x.drop(drop_names, axis=1)
    if isinstance(classify, str):
        y = outcomes[classify]
        mode = RuleInductionType.REGRESSION
    elif callable(classify):
        y = classify(outcomes)
        mode = RuleInductionType.BINARY
    else:
        raise TypeError("unknown type for classify")

    assert y.ndim == 1

    return x, y, mode


def _calculate_quasip(x, y, box, Hbox, Tbox):
    '''

    Parameters
    ----------
    x : DataFrame
    y : np.array
    box : DataFrame
    Hbox : int
    Tbox : int

    '''
    logical = _in_box(x, box)
    yi = y[logical]

    # total nr. of cases in box with one restriction removed
    Tj = yi.shape[0]

    # total nr. of cases of interest in box with one restriction
    # removed
    Hj = np.sum(yi)

    p = Hj / Tj

    Hbox = int(Hbox)
    Tbox = int(Tbox)

    # force one sided
    qp = sp.stats.binom_test(Hbox, Tbox, p, alternative='greater')  # @UndefinedVariable

    return qp


def plot_pair_wise_scatter(x, y, boxlim, box_init, restricted_dims):
    ''' helper function for pair wise scatter plotting

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

    '''

    x = x[restricted_dims]
    data = x.copy()

    # TODO:: have option to change
    # diag to CDF, gives you effectively the
    # regional sensitivity analysis results
    categorical_columns = data.select_dtypes('category').columns.values
    categorical_mappings = {}
    for column in categorical_columns:

        # reorder categorical data so we
        # can capture them in a single column
        categories_inbox = boxlim.at[0, column]
        categories_all = box_init.at[0, column]
        missing = categories_all - categories_inbox
        categories = list(categories_inbox) + list(missing)
#         print(column, categories)
        data[column] = data[column].cat.set_categories(categories)

        # keep the mapping for updating ticklabels
        categorical_mappings[column] = dict(
            enumerate(data[column].cat.categories))

        # replace column with codes
        data[column] = data[column].cat.codes

    data['y'] = y  # for testing
    grid = sns.pairplot(data=data, hue='y', vars=x.columns.values)

    cats = set(categorical_columns)
    for row, ylabel in zip(grid.axes, grid.y_vars):
        ylim = boxlim[ylabel]

        if ylabel in cats:
            y = -0.2
            height = len(ylim[0]) - 0.6  # 2 * 0.2
        else:
            y = ylim[0]
            height = ylim[1] - ylim[0]

        for ax, xlabel in zip(row, grid.x_vars):
            if ylabel == xlabel:
                continue

            if xlabel in cats:
                xlim = boxlim.at[0, xlabel]
                x = -0.2
                width = len(xlim) - 0.6  # 2 * 0.2
            else:
                xlim = boxlim[xlabel]
                x = xlim[0]
                width = xlim[1] - xlim[0]

            xy = x, y
            box = patches.Rectangle(xy, width, height, edgecolor='red',
                                    facecolor='none', lw=3)
            ax.add_patch(box)

    # do the yticklabeling for categorical rows
    for row, ylabel in zip(grid.axes, grid.y_vars):
        if ylabel in cats:
            ax = row[0]
            labels = []
            for entry in ax.get_yticklabels():
                _, value = entry.get_position()
                try:
                    label = categorical_mappings[ylabel][value]
                except KeyError:
                    label = ''
                labels.append(label)
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
                    label = ''
                labels.append(label)
            ax.set_xticks(locs)
            ax.set_xticklabels(labels, rotation=90)
    return grid


def _setup_figure(uncs):
    '''

    helper function for creating the basic layout for the figures that
    show the box lims.

    '''
    nr_unc = len(uncs)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # create the shaded grey background
    rect = mpl.patches.Rectangle((0, -0.5), 1, nr_unc + 1.5,
                                 alpha=0.25,
                                 facecolor="#C0C0C0",
                                 edgecolor="#C0C0C0")
    ax.add_patch(rect)
    ax.set_xlim(left=-0.2, right=1.2)
    ax.set_ylim(top=-0.5, bottom=nr_unc - 0.5)
    ax.yaxis.set_ticks([y for y in range(nr_unc)])
    ax.xaxis.set_ticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(uncs[::-1])
    return fig, ax


def plot_box(boxlim, qp_values, box_init, uncs,
             coverage, density,
             ticklabel_formatter="{} ({})",
             boxlim_formatter="{: .2g}",
             table_formatter="{:.3g}"):
    '''Helper function for parallel coordinate style visualization
    of a box

    Parameters
    ----------
    boxlim : DataFrame
    qp_values : dict
    box_init : DataFrame
    uncs : list
    coverage : float
    density : float
    ticklabel_formatter : str
    boxlim_formatter : str
    table_formatter : str

    Returns
    -------
    a Figure instance


    '''
    norm_box_lim = _normalize(boxlim, box_init, uncs)

    fig, ax = _setup_figure(uncs)
    for j, u in enumerate(uncs):
        # we want to have the most restricted dimension
        # at the top of the figure
        xj = len(uncs) - j - 1

        plot_unc(box_init, xj, j, 0, norm_box_lim,
                 boxlim, u, ax)

        # new part
        dtype = box_init[u].dtype

        props = {'facecolor': 'white',
                 'edgecolor': 'white',
                 'alpha': 0.25}
        y = xj

        if dtype == object:
            elements = sorted(list(box_init[u][0]))
            max_value = (len(elements) - 1)
            values = boxlim.loc[0, u]
            x = [elements.index(entry) for entry in
                 values]
            x = [entry / max_value for entry in x]

            for xi, label in zip(x, values):
                ax.text(xi, y - 0.2, label, ha='center', va='center',
                        bbox=props, color='blue', fontweight='normal')

        else:
            props = {'facecolor': 'white',
                     'edgecolor': 'white',
                     'alpha': 0.25}

            # plot limit text labels
            x = norm_box_lim[j, 0]

            if not np.allclose(x, 0):
                label = boxlim_formatter.format(boxlim.loc[0, u])
                ax.text(x, y - 0.2, label, ha='center', va='center',
                        bbox=props, color='blue', fontweight='normal')

            x = norm_box_lim[j][1]
            if not np.allclose(x, 1):
                label = boxlim_formatter.format(boxlim.loc[1, u])
                ax.text(x, y - 0.2, label, ha='center', va='center',
                        bbox=props, color='blue', fontweight='normal')

            # plot uncertainty space text labels
            x = 0
            label = boxlim_formatter.format(box_init.loc[0, u])
            ax.text(x - 0.01, y, label, ha='right', va='center',
                    bbox=props, color='black', fontweight='normal')

            x = 1
            label = boxlim_formatter.format(box_init.loc[1, u])
            ax.text(x + 0.01, y, label, ha='left', va='center',
                    bbox=props, color='black', fontweight='normal')

        # set y labels
        qp_formatted = {}
        for key, values in qp_values.items():
            values = [vi for vi in values if vi != -1]

            if len(values) == 1:
                value = '{:.2g}'.format(values[0])
            else:
                value = '{:.2g}, {:.2g}'.format(*values)
            qp_formatted[key] = value

        labels = [ticklabel_formatter.format(u, qp_formatted[u]) for u in
                  uncs]

        labels = labels[::-1]
        ax.set_yticklabels(labels)

        # remove x tick labels
        ax.set_xticklabels([])

    coverage = table_formatter.format(coverage)
    density = table_formatter.format(density)

    # add table to the left
    ax.table(cellText=[[coverage], [density]],
             colWidths=[0.1] * 2,
             rowLabels=['coverage', 'density'],
             colLabels=None,
             loc='right',
             bbox=[1.2, 0.9, 0.1, 0.1],)
    plt.subplots_adjust(left=0.1, right=0.75)

    return fig


def plot_ppt(peeling_trajectory):
    '''show the peeling and pasting trajectory in a figure'''

    ax = host_subplot(111)
    ax.set_xlabel("peeling and pasting trajectory")

    par = ax.twinx()
    par.set_ylabel("nr. restricted dimensions")

    ax.plot(peeling_trajectory['mean'], label="mean")
    ax.plot(peeling_trajectory['mass'], label="mass")
    ax.plot(peeling_trajectory['coverage'], label="coverage")
    ax.plot(peeling_trajectory['density'], label="density")
    par.plot(peeling_trajectory['res_dim'], label="restricted dims")
    ax.grid(True, which='both')
    ax.set_ylim(bottom=0, top=1)

    fig = plt.gcf()

    make_legend(['mean', 'mass', 'coverage', 'density',
                 'restricted_dim'],
                ax, ncol=5, alpha=1)
    return fig


def plot_tradeoff(peeling_trajectory, cmap=mpl.cm.viridis):  # @UndefinedVariable
    '''Visualize the trade off between coverage and density. Color
    is used to denote the number of restricted dimensions.

    Parameters
    ----------
    cmap : valid matplotlib colormap

    Returns
    -------
    a Figure instance

    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    boundaries = np.arange(-0.5,
                           max(peeling_trajectory['res_dim']) + 1.5,
                           step=1)
    ncolors = cmap.N
    norm = mpl.colors.BoundaryNorm(boundaries, ncolors)

    p = ax.scatter(peeling_trajectory['coverage'],
                   peeling_trajectory['density'],
                   c=peeling_trajectory['res_dim'],
                   norm=norm,
                   cmap=cmap)
    ax.set_ylabel('density')
    ax.set_xlabel('coverage')
    ax.set_ylim(bottom=0, top=1.2)
    ax.set_xlim(left=0, right=1.2)

    ticklocs = np.arange(0,
                         max(peeling_trajectory['res_dim']) + 1,
                         step=1)
    cb = fig.colorbar(p, spacing='uniform', ticks=ticklocs,
                      drawedges=True)
    cb.set_label("nr. of restricted dimensions")

    return fig


def plot_unc(box_init, xi, i, j, norm_box_lim, box_lim, u, ax,
             color=sns.color_palette()[0]):
    '''

    Parameters:
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

    '''

    dtype = box_init[u].dtype

    y = xi - j * 0.1

    if dtype == object:
        elements = sorted(list(box_init[u][0]))
        max_value = (len(elements) - 1)
        box_lim = box_lim[u][0]
        x = [elements.index(entry) for entry in
             box_lim]
        x = [entry / max_value for entry in x]
        y = [y] * len(x)

        ax.scatter(x, y, edgecolor=color,
                   facecolor=color)

    else:
        ax.plot(norm_box_lim[i], (y, y),
                c=color)


def plot_boxes(x, boxes, together):
    '''Helper function for plotting multiple boxlims

    Parameters
    ----------
    x : pd.DataFrame
    boxes : list of pd.DataFrame
    together : bool

    '''

    box_init = _make_box(x)
    box_lims, uncs = _get_sorted_box_lims(boxes, box_init)

    # normalize the box lims
    # we don't need to show the last box, for this is the
    # box_init, which is visualized by a grey area in this
    # plot.
    norm_box_lims = [_normalize(box_lim, box_init, uncs) for
                     box_lim in boxes]

    if together:
        fig, ax = _setup_figure(uncs)

        for i, u in enumerate(uncs):
            colors = itertools.cycle(COLOR_LIST)
            # we want to have the most restricted dimension
            # at the top of the figure

            xi = len(uncs) - i - 1

            for j, norm_box_lim in enumerate(norm_box_lims):
                color = next(colors)
                plot_unc(box_init, xi, i, j, norm_box_lim,
                         box_lims[j], u, ax, color)

        plt.tight_layout()
        return fig
    else:
        figs = []
        colors = itertools.cycle(COLOR_LIST)

        for j, norm_box_lim in enumerate(norm_box_lims):
            fig, ax = _setup_figure(uncs)
            ax.set_title('box {}'.format(j))
            color = next(colors)

            figs.append(fig)
            for i, u in enumerate(uncs):
                xi = len(uncs) - i - 1
                plot_unc(box_init, xi, i, 0, norm_box_lim,
                         box_lims[j], u, ax, color)

            plt.tight_layout()
        return figs


class OutputFormatterMixin(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def boxes(self):
        '''Property for getting a list of box limits'''

        raise NotImplementedError

    @abc.abstractproperty
    def stats(self):
        '''property for getting a list of dicts containing the statistics
        for each box'''

        raise NotImplementedError

    def boxes_to_dataframe(self):
        '''convert boxes to pandas dataframe'''

        boxes = self.boxes

        # determine the restricted dimensions
        # print only the restricted dimension
        box_lims, uncs = _get_sorted_box_lims(boxes, _make_box(self.x))
        nr_boxes = len(boxes)
        dtype = float
        index = ["box {}".format(i + 1) for i in range(nr_boxes)]
        for value in box_lims[0].dtypes:
            if value == object:
                dtype = object
                break

        columns = pd.MultiIndex.from_product([index,
                                              ['min', 'max', ]])
        df_boxes = pd.DataFrame(np.zeros((len(uncs), nr_boxes * 2)),
                                index=uncs,
                                dtype=dtype,
                                columns=columns)

        # TODO should be possible to make more efficient
        for i, box in enumerate(box_lims):
            for unc in uncs:
                values = box.loc[:, unc]
                values = values.rename({0: 'min', 1: 'max'})
                df_boxes.loc[unc][index[i]] = values.values
        return df_boxes

    def stats_to_dataframe(self):
        '''convert stats to pandas dataframe'''

        stats = self.stats

        index = pd.Index(['box {}'.format(i + 1) for i in range(len(stats))])

        return pd.DataFrame(stats, index=index)

    def show_boxes(self, together=False):
        '''display boxes

        Parameters
        ----------
        together : bool, otional

        '''
        plot_boxes(self.x, self.boxes, together=together)
