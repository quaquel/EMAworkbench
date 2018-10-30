'''
Scenario discovery utilities used by both :mod:`cart` and :mod:`prim`
'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import abc
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns

from .plotting_util import COLOR_LIST

# Created on May 24, 2015
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

REGRESSION = 'regression'
'''constant indicating regression mode'''

BINARY = 'binary'
'''constant indicating binary classification mode. This is the most
common used mode in scenario discovery'''

CLASSIFICATION = 'classification'
'''constant indicating classification mode'''


def _get_sorted_box_lims(boxes, box_init):
    '''Sort the uncertainties for each box in boxes based on a normalization
    given box_init. Unrestricted dimensions are dropped. The sorting is based
    on the normalization of the first box in boxes. 

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
    box_size = nbl[:, 1]-nbl[:, 0]

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

    def limits(x):
        if (x.dtype == int) or (x.dtype == float):
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
            dif = (box_init.loc[1, u]-box_init.loc[0, u])
            a = 1/dif
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
    restricted_dims = cols[np.all( box_init.values==box_limits.values, axis=0)==False]
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


def _setup_figure(uncs):
    '''

    helper function for creating the basic layout for the figures that
    show the box lims.

    '''
    nr_unc = len(uncs)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # create the shaded grey background
    rect = mpl.patches.Rectangle((0, -0.5), 1, nr_unc+1.5,
                                 alpha=0.25,
                                 facecolor="#C0C0C0",
                                 edgecolor="#C0C0C0")
    ax.add_patch(rect)
    ax.set_xlim(left=-0.2, right=1.2)
    ax.set_ylim(top=-0.5, bottom=nr_unc-0.5)
    ax.yaxis.set_ticks([y for y in range(nr_unc)])
    ax.xaxis.set_ticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(uncs[::-1])
    return fig, ax


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
    logical = (boxlim_numbered.loc[0, :].values<= x_numbered.values) &\
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
        drop_names = set(x.columns.values.tolist())-set(incl_unc)
        x = x.drop(drop_names, axis=1)
    if isinstance(classify, str):
        y = outcomes[classify]
        mode = REGRESSION
    elif callable(classify):
        y = classify(outcomes)
        mode = BINARY
    else:
        raise TypeError("unknown type for classify")
    
    assert y.ndim==1
    
    return x, y, mode


def _calculate_quasip(x, y, box, Hbox, Tbox):
    '''
    
    Parameters
    ----------
    x : recarray
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

    p = Hj/Tj

    Hbox = int(Hbox)
    Tbox = int(Tbox)

    # force one sided
    qp = sp.stats.binom_test(Hbox, Tbox, p, alternative='greater')  # @UndefinedVariable

    return qp


def plot_box(boxlim, qp_values, box_init, uncs,
             coverage, density,
             ticklabel_formatter="{} ({})",
             boxlim_formatter="{: .2g}",
             table_formatter="{:.3g})"):
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

        OutputFormatterMixin._plot_unc(box_init, xj, j, 0, norm_box_lim,
                                              boxlim, u, ax)

        # new part
        dtype = box_init[u].dtype

        props = {'facecolor': 'white',
                 'edgecolor': 'white',
                 'alpha': 0.25}
        y = xj

        if dtype == object:
            elements = sorted(list(box_init[u][0]))
            max_value = (len(elements)-1)
            values = boxlim.loc[0, u]
            x = [elements.index(entry) for entry in
                 values]
            x = [entry/max_value for entry in x]

            for xi, label in zip(x, values):
                ax.text(xi, y-0.2, label, ha='center', va='center',
                        bbox=props, color='blue', fontweight='normal')

        else:
            props = {'facecolor': 'white',
                     'edgecolor': 'white',
                     'alpha': 0.25}

            # plot limit text labels
            x = norm_box_lim[j, 0]

            if not np.allclose(x, 0):
                label = boxlim_formatter.format(boxlim.loc[0, u])
                ax.text(x, y-0.2, label, ha='center', va='center',
                        bbox=props, color='blue', fontweight='normal')

            x = norm_box_lim[j][1]
            if not np.allclose(x, 1):
                label = boxlim_formatter.format(boxlim.loc[1, u])
                ax.text(x, y-0.2, label, ha='center', va='center',
                        bbox=props, color='blue', fontweight='normal')

            # plot uncertainty space text labels
            x = 0
            label = boxlim_formatter.format(box_init.loc[0, u])
            ax.text(x-0.01, y, label, ha='right', va='center',
                    bbox=props, color='black', fontweight='normal')

            x = 1
            label = boxlim_formatter.format(box_init.loc[1, u])
            ax.text(x+0.01, y, label, ha='left', va='center',
                    bbox=props, color='black', fontweight='normal')

        # set y labels
        qp_formatted = {}
        for key, values in qp_values.items():
            values = [vi for vi in values if vi!=-1]
            
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

        # add table to the left
        ax.table(cellText=[[coverage], [density]],
                 colWidths=[0.1]*2,
                 rowLabels=['coverage', 'density'],
                 colLabels=None,
                 loc='right',
                 bbox=[1.2, 0.9, 0.1, 0.1])

    return fig


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
        index = ["box {}".format(i+1) for i in range(nr_boxes)]
        for value in box_lims[0].dtypes:
            if value == object:
                dtype = object
                break

        columns = pd.MultiIndex.from_product([index,
                                              ['min', 'max', ]])
        df_boxes = pd.DataFrame(np.zeros((len(uncs), nr_boxes*2)),
                                index=uncs,
                                dtype=dtype,
                                columns=columns)
        
        # TODO should be possible to make more efficient
        for i, box in enumerate(box_lims):
            for unc in uncs:
                values = box.loc[:, unc]
                values = values.rename({0:'min', 1:'max'})
                df_boxes.loc[unc][index[i]] = values
        return df_boxes

    def stats_to_dataframe(self):
        '''convert stats to pandas dataframe'''

        stats = self.stats

        index = pd.Index(['box {}'.format(i+1) for i in range(len(stats))])

        return pd.DataFrame(stats, index=index)

    def display_boxes(self, together=False):
        '''display boxes

        Parameters
        ----------
        together : bool, otional

        '''
        # TODO:: use plot_box function
        
        box_init = _make_box(self.x)
        box_lims, uncs = _get_sorted_box_lims(self.boxes, box_init)

        # normalize the box lims
        # we don't need to show the last box, for this is the
        # box_init, which is visualized by a grey area in this
        # plot.
        norm_box_lims = [_normalize(box_lim, box_init, uncs) for
                         box_lim in box_lims[0:-1]]
        
        if together:
            fig, ax = _setup_figure(uncs)

            for i, u in enumerate(uncs):
                colors = itertools.cycle(COLOR_LIST)
                # we want to have the most restricted dimension
                # at the top of the figure
                
                xi = len(uncs) - i - 1

                for j, norm_box_lim in enumerate(norm_box_lims):
                    color = next(colors)
                    self._plot_unc(box_init, xi, i, j, norm_box_lim,
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
                    self._plot_unc(box_init, xi, i, 0, norm_box_lim,
                                   box_lims[j], u, ax, color)

                plt.tight_layout()
            return figs

    @staticmethod
    def _plot_unc(box_init, xi, i, j, norm_box_lim, box_lim, u, ax,
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

        y = xi-j*0.1

        if dtype == object:
            elements = sorted(list(box_init[u][0]))
            max_value = (len(elements)-1)
            box_lim = box_lim[u][0]
            x = [elements.index(entry) for entry in
                 box_lim]
            x = [entry/max_value for entry in x]
            y = [y] * len(x)

            ax.scatter(x, y,  edgecolor=color,
                       facecolor=color)

        else:
            ax.plot(norm_box_lim[i], (y, y),
                    c=color)