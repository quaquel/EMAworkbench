'''

Plotting utility functions

'''
import copy
import enum

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats.kde as kde
from scipy.stats import gaussian_kde, scoreatpercentile

import seaborn as sns

from ..util import EMAError, get_module_logger

# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ["Density", "COLOR_LIST", "LegendEnum", "PlotType"]

_logger = get_module_logger(__name__)

COLOR_LIST = sns.color_palette()
'''Default color list'''
sns.set_palette(COLOR_LIST)

TIME = "TIME"
'''Default key for time'''

# ==============================================================================
# actual plotting functions
# ==============================================================================


class Density(enum.Enum):
    '''Enum for different types of density plots'''

    KDE = 'kde'
    '''constant for plotting density as a kernel density estimate'''

    HIST = 'hist'
    '''constant for plotting density as a histogram'''

    BOXPLOT = 'boxplot'
    '''constant for plotting density as a boxplot'''

    VIOLIN = 'violin'
    '''constant for plotting density as a violin plot, which combines a
    Gaussian density estimate with a boxplot'''


    BOXENPLOT = 'boxenplot'
    '''constant for plotting density as a boxenplot'''

class LegendEnum(enum.Enum):
    '''Enum for different styles of legends
    '''

    # used for legend
    LINE = 'line'
    PATCH = 'patch'
    SCATTER = 'scatter'


class PlotType(enum.Enum):
    ENVELOPE = 'envelope'
    '''constant for plotting envelopes'''

    LINES = 'lines'
    '''constant for plotting lines'''

    ENV_LIN = "env_lin"
    '''constant for plotting envelopes with lines'''


def plot_envelope(ax, j, time, value, fill=False):
    '''

    Helper function, responsible for plotting an envelope.

    Parameters
    ----------
    ax : axes instance
    j : int
    time : ndarray
    value : ndarray
    fill : bool


    '''

    # plot minima and maxima
    minimum = np.min(value, axis=0)
    maximum = np.max(value, axis=0)

    color = get_color(j)

    if fill:
        #        ax.plot(time, minimum, color=color, alpha=0.3)
        #        ax.plot(time, maximum, color=color, alpha=0.3)
        ax.fill_between(time,
                        minimum,
                        maximum,
                        facecolor=color,
                        alpha=0.3,
                        )
    else:
        ax.plot(time, minimum, c=color)
        ax.plot(time, maximum, c=color)


def plot_histogram(ax, values, log):
    '''

    Helper function, responsible for plotting a histogram

    Parameters
    ----------
    ax : axes instance
    values : ndarray
    log : bool


    '''
    if isinstance(values, list):
        color = [get_color(i) for i in range(len(values))]
    else:
        color = get_color(0)
    a = ax.hist(values,
                bins=11,
                orientation='horizontal',
                histtype='bar',
                density=True,
                color=color,
                log=log)
    if not log:
        ax.set_xticks([0, ax.get_xbound()[1]])
    return a


def plot_kde(ax, values, log):
    '''

    Helper function, responsible for plotting a KDE.

    Parameters
    ----------
    ax : axes instance
    values : ndarray
    log : bool


    '''

    for j, value in enumerate(values):
        color = get_color(j)
        kde_x, kde_y = determine_kde(value)
        ax.plot(kde_x, kde_y, c=color, ms=1, markevery=20)

        if log:
            ax.set_xscale('log')
        else:
            ax.set_xticks([int(0),
                           ax.get_xaxis().
                           get_view_interval()[1]])
            labels = ["{0:.2g}".format(0), "{0:.2g}".format(ax.get_xlim()[1])]
            ax.set_xticklabels(labels)


def plot_boxplots(ax, values, log, group_labels=None):
    '''
    helper function for plotting a boxplot

    Parameters
    ----------
    ax : axes instance
    values : ndarray
    log : bool
    group_labels : list of str, optional


    '''

    if log:
        _logger.warning("log option ignored for boxplot")

    ax.boxplot(values)
    if group_labels:
        ax.set_xticklabels(group_labels, rotation='vertical')


def plot_violinplot(ax, values, log, group_labels=None):
    '''
    helper function for plotting violin plots on axes

    Parameters
    ----------
    ax : axes instance
    values : ndarray
    log : bool
    group_labels : list of str, optional

    '''

    if log:
        _logger.warning("log option ignored for violin plot")

    if not group_labels:
        group_labels = ['']
        
    data = pd.DataFrame.from_records({k:v for k, v in zip(group_labels, values)})
    data = pd.melt(data)
        
    sns.violinplot(x='variable', y='value', data=data, order=group_labels,
                  ax=ax)

#     pos = range(len(value))
#     dist = max(pos) - min(pos)
#     _ = min(0.15 * max(dist, 1.0), 0.5)
#     for data, p in zip(value, pos):
#         if len(data) > 0:
#             kde = gaussian_kde(data)  # calculates the kernel density
#             x = np.linspace(np.min(data), np.max(data),
#                             250.)  # support for violin
#             v = kde.evaluate(x)  # violin profile (density curve)
# 
#             scl = 1 / (v.max() / 0.4)
#             v = v * scl  # scaling the violin to the available space
#             ax.fill_betweenx(
#                 x, p - v, p + v, facecolor=get_color(p), alpha=0.6, lw=1.5)
# 
#             for percentile in [25, 75]:
#                 quant = scoreatpercentile(data.ravel(), percentile)
#                 q_x = kde.evaluate(quant) * scl
#                 q_x = [p - q_x, p + q_x]
#                 ax.plot(q_x, [quant, quant], linestyle=":", c='k')
#             med = np.median(data)
#             m_x = kde.evaluate(med) * scl
#             m_x = [p - m_x, p + m_x]
#             ax.plot(m_x, [med, med], linestyle="--", c='k', lw=1.5)
# 
#     if group_labels:
#         labels = group_labels[:]
#         labels.insert(0, '')
#         ax.set_xticklabels(labels, rotation='vertical')


def plot_boxenplot(ax, values, log, group_labels=None):
    '''
    helper function for plotting boxenplot plots on axes

    Parameters
    ----------
    ax : axes instance
    values : ndarray
    log : bool
    group_labels : list of str, optional

    '''

    if log:
        _logger.warning("log option ignored for violin plot")
    if not group_labels:
        group_labels = ['']
        
    data = pd.DataFrame.from_records({k:v for k, v in zip(group_labels, values)})
    data = pd.melt(data)
        
    sns.boxenplot(x='variable', y='value', data=data, order=group_labels,
                  ax=ax)


def group_density(ax_d, density, outcomes, outcome_to_plot, group_labels,
                  log=False, index=-1):
    '''
    helper function for plotting densities in case of grouped data


    Parameters
    ----------
    ax_d : axes instance
    density : {HIST, BOXPLOT, VIOLIN, KDE}
    outcomes :  dict
    outcome_to_plot : str
    group_labels : list of str
    log : bool, optional
    index : int, optional

    Raises
    ------
    EMAError
        if density is unkown

    '''
    values = [outcomes[key][outcome_to_plot][:, index] for key in
              group_labels]

    if density == Density.HIST:
        plot_histogram(ax_d, values, log)
    elif density == Density.BOXPLOT:
        plot_boxplots(ax_d, values, log, group_labels=group_labels)
    elif density == Density.VIOLIN:
        plot_violinplot(ax_d, values, log, group_labels=group_labels)
    elif density == Density.KDE:
        plot_kde(ax_d, values, log)
    elif density == Density.BOXENPLOT:
        plot_boxenplot(ax_d, values, log, group_labels=group_labels)
    else:
        raise EMAError("unknown density type: {}".format(density))
    
    ax_d.set_xlabel('')
    ax_d.set_ylabel('')

def simple_density(density, value, ax_d, ax, log):
    '''

    Helper function, responsible for producing a density plot

    Parameters
    ----------
    density : {HIST, BOXPLOT, VIOLIN, KDE}
    value : ndarray
    ax_d : axes instance
    ax : axes instance
    log : bool

    '''

    if density == Density.KDE:
        plot_kde(ax_d, [value[:, -1]], log)
    elif density == Density.HIST:
        plot_histogram(ax_d, value[:, -1], log)
    elif density == Density.BOXPLOT:
        plot_boxplots(ax_d, value[:, -1], log)
    elif density == Density.VIOLIN:
        plot_violinplot(ax_d, [value[:, -1]], log)
    elif density == Density.BOXENPLOT:
        plot_violinplot(ax_d, [value[:, -1]], log)    
    else:
        raise EMAError("unknown density plot type")

    ax_d.get_yaxis().set_view_interval(
        ax.get_yaxis().get_view_interval()[0],
        ax.get_yaxis().get_view_interval()[1])
    ax_d.set_ylim(bottom=ax.get_yaxis().get_view_interval()[0],
                  top=ax.get_yaxis().get_view_interval()[1])

    ax_d.set_xlabel('')
    ax_d.set_ylabel('')

def simple_kde(outcomes, outcomes_to_show, colormap, log, minima, maxima):
    '''

    Helper function for generating a density heatmap over time

    Parameters
    ----------
    outcomes : dict
    outcomes_to_show : list of str
    colormap : str
    log : bool
    minima : dict
    maxima : dict

    '''
    size_kde = 100
    fig, axes = plt.subplots(len(outcomes_to_show), squeeze=False)
    axes = axes[:, 0]

    axes_dict = {}

    # do the plotting
    for outcome_to_plot, ax in zip(outcomes_to_show, axes):
        axes_dict[outcome_to_plot] = ax

        outcome = outcomes[outcome_to_plot]

        kde_over_time = np.zeros(shape=(size_kde, outcome.shape[1]))
        ymin = minima[outcome_to_plot]
        ymax = maxima[outcome_to_plot]

        # make kde over time
        for j in range(outcome.shape[1]):
            kde_x = determine_kde(outcome[:, j], size_kde, ymin, ymax)[0]
            kde_x = kde_x / np.max(kde_x)

            if log:
                kde_x = np.log(kde_x + 1)
            kde_over_time[:, j] = kde_x

        sns.heatmap(kde_over_time[::-1, :], ax=ax, cmap=colormap, cbar=True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("time")
        ax.set_ylabel(outcome_to_plot)

    return fig, axes_dict


def make_legend(categories, ax, ncol=3, legend_type=LegendEnum.LINE,
                alpha=1):
    '''
    Helper function responsible for making the legend

    Parameters
    ----------
    categories : str or tuple
                 the categories in the legend
    ax : axes instance
         the axes with which the legend is associated
    ncol : int
           the number of columns to use
    legend_type : {LINES, SCATTER, PATCH}
                  whether the legend is linked to lines, patches, or scatter
                  plots
    alpha : float
            the alpha of the artists

    '''

    some_identifiers = []
    labels = []
    for i, category in enumerate(categories):
        color = get_color(i)

        if legend_type == LegendEnum.LINE:
            artist = plt.Line2D([0, 1], [0, 1], color=color,
                                alpha=alpha)  # TODO
        elif legend_type == LegendEnum.SCATTER:
            #             marker_obj = mpl.markers.MarkerStyle('o')
            #             path = marker_obj.get_path().transformed(
            #                              marker_obj.get_transform())
            #             artist  = mpl.collections.PathCollection((path,),
            #                                         sizes = [20],
            #                                         facecolors = COLOR_LIST[i],
            #                                         edgecolors = 'k',
            #                                         offsets = (0,0)
            #                                         )
            # TODO work arround, should be a proper proxyartist for scatter
            # legends
            artist = mpl.lines.Line2D([0], [0], linestyle="none",
                                      c=color, marker='o')

        elif legend_type == LegendEnum.PATCH:
            artist = plt.Rectangle((0, 0), 1, 1, edgecolor=color,
                                   facecolor=color, alpha=alpha)

        some_identifiers.append(artist)

        if isinstance(category, tuple):
            label = '%.2f - %.2f' % category
        else:
            label = category

        labels.append(str(label))

    ax.legend(some_identifiers, labels, ncol=ncol,
              loc=3, borderaxespad=0.1,
              mode='expand', bbox_to_anchor=(0., 1.1, 1., .102))


def determine_kde(data,
                  size_kde=1000,
                  ymin=None,
                  ymax=None):
    '''

    Helper function responsible for performing a KDE

    Parameters
    ----------
    data : ndarray
    size_kde : int, optional
    ymin : float, optional
    ymax : float, optional

    Returns
    -------
    ndarray
        x values for kde
    ndarray
        y values for kde

    ..note:: x and y values are based on rotation as used in density
             plots for end states.


    '''
    if not ymin:
        ymin = np.min(data)
    if not ymax:
        ymax = np.max(data)

    kde_y = np.linspace(ymin, ymax, size_kde)

    try:
        kde_x = kde.gaussian_kde(data)
        kde_x = kde_x.evaluate(kde_y)
#         grid = GridSearchCV(KernelDensity(kernel='gaussian'),
#                             {'bandwidth': np.linspace(ymin, ymax, 20)},
#                             cv=20)
#         grid.fit(data[:, np.newaxis])
#         best_kde = grid.best_estimator_
#         kde_x = np.exp(best_kde.score_samples(kde_y[:, np.newaxis]))
    except Exception as e:
        _logger.warning(e)
        kde_x = np.zeros(kde_y.shape)

    return kde_x, kde_y


def filter_scalar_outcomes(outcomes):
    '''
    Helper function that removes non time series outcomes from all the
    outcomes.

    Parameters
    ----------
    outcomes : dict

    Returns
    -------
    dict
        the filtered outcomes


    '''
    temp = {}
    for key, value in outcomes.items():
        if value.ndim < 2:
            _logger.info(("{} not shown because it is "
                          "not time series data").format(key))
        else:
            temp[key] = value
    return temp


def determine_time_dimension(outcomes):
    '''
    helper function for determining or creating time dimension


    Parameters
    ----------
    outcomes : dict

    Returns
    -------
    ndarray


    '''

    time = None
    try:
        time = outcomes['TIME']
        time = time[0, :]
        outcomes.pop('TIME')
    except KeyError:
        values = iter(outcomes.values())
        for value in values:
            if value.ndim == 2:
                time = np.arange(0, value.shape[1])
                break

    if time is None:
        _logger.info("no time dimension found in results")
    return time, outcomes


def group_results(experiments, outcomes, group_by, grouping_specifiers,
                  grouping_labels):
    '''
    Helper function that takes the experiments and results and returns a list
    based on groupings. Each element in the dictionary contains the experiments
    and results for a particular group, the key is the grouping specifier.

    Parameters
    ----------
    experiments : DataFrame
    outcomes : dict
    group_by : str
               The column in the experiments array to which the grouping
               specifiers apply. If the name is'index' it is assumed that the
               grouping specifiers are valid indices for numpy.ndarray.
    grouping_specifiers : iterable
                    An iterable of grouping specifiers. A grouping
                    specifier is a unique identifier in case of grouping by
                    categorical uncertainties. It is a tuple in case of
                    grouping by a parameter uncertainty. In this cose, the code
                    treats the tuples as half open intervals, apart from the
                    last entry, which is treated as closed on both sides.
                    In case of 'index', the iterable should be a dictionary
                    with the name for each group as key and the value being a
                    valid index for numpy.ndarray.

    Returns
    -------
    dict
        A dictionary with the experiments and results for each group, the
        grouping specifier is used as key

    ..note:: In case of grouping by parameter uncertainty, the list of
             grouping specifiers is sorted. The traversal assumes half open
             intervals, where the upper limit of each interval is open, except
             for the last interval which is closed.

    '''
    groups = {}
    if group_by != 'index':
        column_to_group_by = experiments.loc[:, group_by]

    for label, specifier in zip(grouping_labels, grouping_specifiers):
        if isinstance(specifier, tuple):
            # the grouping is a continuous uncertainty
            lower_limit, upper_limit = specifier

            # check whether it is the last grouping specifier
            if grouping_specifiers.index(specifier) ==\
                    len(grouping_specifiers) - 1:
                # last case

                logical = (column_to_group_by >= lower_limit) &\
                    (column_to_group_by <= upper_limit)
            else:
                logical = (column_to_group_by >= lower_limit) &\
                    (column_to_group_by < upper_limit)
        elif group_by == 'index':
            # the grouping is based on indices
            logical = specifier
        else:
            # the grouping is an integer or categorical uncertainty
            logical = column_to_group_by == specifier

        group_outcomes = {}
        for key, value in outcomes.items():
            value = value[logical]
            group_outcomes[key] = value
        groups[label] = (experiments.loc[logical, :], group_outcomes)

    return groups


def make_continuous_grouping_specifiers(array, nr_of_groups=5):
    '''
    Helper function for discretesizing a continuous array. By default, the
    array is split into 5 equally wide intervals.

    Parameters
    ----------
    array : ndarray
            a 1-d array that is to be turned into discrete intervals.
    nr_of_groups : int, optional

    Returns
    -------
    list of tuples
        list of tuples with the lower and upper bound of the intervals.


    .. note:: this code only produces intervals. :func:`group_results` uses
              these intervals in half-open fashion, apart from the last
              interval: [a, b), [b,c), [c,d]. That is, both the end point
              and the start point of the range of the continuous array are
              included.

    '''

    minimum = np.min(array)
    maximum = np.max(array)
    step = (maximum - minimum) / nr_of_groups
    a = [(minimum + step * x, minimum + step * (x + 1))
         for x in range(nr_of_groups)]
    assert a[0][0] == minimum
    assert a[-1][1] == maximum
    return a


def prepare_pairs_data(experiments, outcomes,
                       outcomes_to_show=None,
                       group_by=None,
                       grouping_specifiers=None,
                       point_in_time=-1,
                       filter_scalar=True):
    '''

    Parameters
    ----------
    results : tuple
    outcomes_to_show : list of str, optional
    group_by : str, optional
    grouping_specifiers : iterable, optional
    point_in_time : int, optional
    filter_scalar : bool, optional

    '''
    if isinstance(outcomes_to_show, str):
        raise EMAError(
            "for pair wise plotting, more than one outcome needs to be provided")

    outcomes, outcomes_to_show, time, grouping_labels = prepare_data(
        experiments, outcomes, outcomes_to_show, group_by, grouping_specifiers, filter_scalar)

    def filter_outcomes(outcomes, point_in_time):
        new_outcomes = {}
        for key, value in outcomes.items():
            if len(value.shape) == 2:
                new_outcomes[key] = value[:, point_in_time]
            else:
                new_outcomes[key] = value
        return new_outcomes

    if point_in_time:
        if point_in_time != -1:
            point_in_time = np.where(time == point_in_time)

        if group_by:
            new_outcomes = {}
            for key, value in outcomes.items():
                new_outcomes[key] = filter_outcomes(value, point_in_time)
            outcomes = new_outcomes
        else:
            outcomes = filter_outcomes(outcomes, point_in_time)
    return outcomes, outcomes_to_show, grouping_labels


def prepare_data(experiments, outcomes, outcomes_to_show=None,
                 group_by=None, grouping_specifiers=None,
                 filter_scalar=True):
    '''Helper function for preparing datasets prior to plotting

    Parameters
    ----------
    experiments : DataFrame
    outcomes : dict
    outcomes_to_show : list of str, optional
    group_by : str, optional
    grouping_specifiers : iterable, optional
    filter_scalar : bool, optional

    '''
    experiments = experiments.copy()
    outcomes = copy.copy(outcomes)

    time, outcomes = determine_time_dimension(outcomes)
    temp_outcomes = {}

    # remove outcomes that are not to be shown
    if outcomes_to_show:
        if isinstance(outcomes_to_show, str):
            outcomes_to_show = [outcomes_to_show]

        for entry in outcomes_to_show:
            temp_outcomes[entry] = outcomes[entry]

    # filter the outcomes to exclude scalar values
    if filter_scalar:
        outcomes = filter_scalar_outcomes(outcomes)
    if not outcomes_to_show:
        outcomes_to_show = outcomes.keys()

    # group the data if desired
    if group_by:
        if not grouping_specifiers:
            # no grouping specifier, so infer from the data
            if group_by == 'index':
                raise EMAError(("no grouping specifiers provided while "
                                "trying to group on index"))
            else:
                column_to_group_by = experiments[group_by]
                if (column_to_group_by.dtype == np.object) or\
                        (column_to_group_by.dtype == 'category'):
                    grouping_specifiers = set(column_to_group_by)
                else:
                    grouping_specifiers = make_continuous_grouping_specifiers(
                        column_to_group_by, grouping_specifiers)
            grouping_labels = grouping_specifiers = sorted(grouping_specifiers)
        else:
            if isinstance(grouping_specifiers, str):
                grouping_specifiers = [grouping_specifiers]
                grouping_labels = grouping_specifiers
            elif isinstance(grouping_specifiers, dict):
                grouping_labels = sorted(grouping_specifiers.keys())
                grouping_specifiers = [grouping_specifiers[key] for key in
                                       grouping_labels]
            else:
                grouping_labels = grouping_specifiers

        outcomes = group_results(experiments, outcomes, group_by,
                                 grouping_specifiers, grouping_labels)

        new_outcomes = {}
        for key, value in outcomes.items():
            new_outcomes[key] = value[1]
        outcomes = new_outcomes
    else:
        grouping_labels = []

    return outcomes, outcomes_to_show, time, grouping_labels


def do_titles(ax, titles, outcome):
    '''
    Helper function for setting the title on an ax

    Parameters
    ----------
    ax : axes instance
    titles : dict
             a dict which maps outcome names to titles
    outcome : str
              the outcome plotted in the ax.

    '''

    if isinstance(titles, dict):
        if not titles:
            ax.set_title(outcome)
        else:
            try:
                ax.set_title(titles[outcome])
            except KeyError:
                _logger.warning(
                    "key error in do_titles, no title provided for `%s`" %
                    (outcome))
                ax.set_title(outcome)


def do_ylabels(ax, ylabels, outcome):
    '''
    Helper function for setting the y labels on an ax

    Parameters
    ----------
    ax : axes instance
    titles : dict
             a dict which maps outcome names to y labels
    outcome : str
              the outcome plotted in the ax.

    '''

    if isinstance(ylabels, dict):
        if not ylabels:
            ax.set_ylabel(outcome)
        else:
            try:
                ax.set_ylabel(ylabels[outcome])
            except KeyError:
                _logger.warning(
                    "key error in do_ylabels, no ylabel provided for `%s`" %
                    (outcome))
                ax.set_ylabel(outcome)


def make_grid(outcomes_to_show, density=False):
    '''
    Helper function for making the grid that specifies the size and location
    of the various axes.

    Parameters
    ----------
    outcomes_to_show : list of str
                       the list of outcomes to show
    density: boolean : bool, optional

    '''

    # make the plotting grid
    if density:
        grid = gridspec.GridSpec(len(outcomes_to_show), 2,
                                 width_ratios=[4, 1])
    else:
        grid = gridspec.GridSpec(len(outcomes_to_show), 1)
    grid.update(wspace=0.1,
                hspace=0.4)

    figure = plt.figure()
    return figure, grid


def get_color(index):
    '''helper function for cycling over color list if the number of items
    is higher than the legnth of the color list
    '''
    corrected_index = index % len(COLOR_LIST)
    return COLOR_LIST[corrected_index]
