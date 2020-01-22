"""


this module provides functions for generating some basic figures. The code can
be used as is, or serve as an example for writing your own code.

"""
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import ConnectionPatch

# from . import plotting_util
from .plotting_util import (prepare_data, simple_kde, group_density,
                            make_grid, make_legend, plot_envelope,
                            simple_density, do_titles, do_ylabels, TIME,
                            PlotType, get_color, Density, LegendEnum)
from ..util import EMAError, get_module_logger

# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['lines',
           'envelopes',
           'kde_over_time',
           'multiple_densities']
_logger = get_module_logger(__name__)
TIME_LABEL = 'Time'


def envelopes(experiments,
              outcomes,
              outcomes_to_show=None,
              group_by=None,
              grouping_specifiers=None,
              density=None,
              fill=False,
              legend=True,
              titles={},
              ylabels={},
              log=False):
    ''' Make envelop plots.

    An envelope shows over time the minimum and maximum  value for a set
    of runs over time. It is thus to be used in case of time series
    data. The function will try to find a result labeled "TIME". If this
    is present, these values will be used on the X-axis. In case of
    Vensim models, TIME is present by default.

    Parameters
    ----------
    experiments : DataFrame
    outcomes : dict
    outcomes_to_show, str, list of str, optional
    group_by : str, optional
               name of the column in the experimentsto group results by.
               Alternatively, `index` can be used to use indexing
               arrays as the basis for grouping.
    grouping_specifiers : iterable or dict, optional
                          set of categories to be used as a basis for
                          grouping by. Grouping_specifiers is only
                          meaningful if group_by is provided as well.
                          In case of grouping by index, the grouping
                          specifiers should be in a  dictionary where
                          the key denotes the name of the group.
    density : {None, HIST, KDE, VIOLIN, BOXPLOT}, optional
    fill : bool, optional
    legend : bool, optional
    titles : dict, optional
             a way for controlling whether each of the axes should have a
             title. There are three possibilities. If set to None, no title
             will be shown for any of the axes. If set to an empty dict,
             the default, the title is identical to the name of the outcome of
             interest. If you want to override these default names, provide a
             dict with the outcome of interest as key and the desired title as
             value. This dict need only contain the outcomes for which you
             want to use a different title.
    ylabels : dict, optional
              way for controlling the ylabels. Works identical to titles.
    log : bool, optional
          log scale density plot

    Returns
    -------
    Figure : Figure instance
    axes : dict
           dict with outcome as key, and axes as value. Density axes' are
           indexed by the outcome followed by _density.

    Note
    ----
    the current implementation is limited to seven different categories in case
    of group_by, categories, and/or discretesize. This limit is due to the colors
    specified in COLOR_LIST.

    Examples
    --------

    >>> import util as util
    >>> data = util.load_results(r'1000 flu cases.cPickle')
    >>> envelopes(data, group_by='policy')

    will show an envelope for three three different policies, for all the
    outcomes of interest. while

    >>> envelopes(data, group_by='policy', categories=['static policy',
                  'adaptive policy'])

    will only show results for the two specified policies, ignoring any results
    associated with \'no policy\'.

    '''
    _logger.debug("generating envelopes")
    prepared_data = prepare_data(experiments, outcomes,
                                 outcomes_to_show, group_by,
                                 grouping_specifiers,
                                 filter_scalar=True)
    outcomes, outcomes_to_show, time, grouping_labels = prepared_data

    figure, grid = make_grid(outcomes_to_show, density)

    # do the plotting
    axes_dict = {}
    for i, outcome_to_plot in enumerate(outcomes_to_show):
        ax = figure.add_subplot(grid[i, 0])
        axes_dict[outcome_to_plot] = ax

        ax_d = None
        if density:
            ax_d = figure.add_subplot(grid[i, 1], sharey=ax)
            axes_dict[outcome_to_plot + "_density"] = ax_d

        if group_by:
            group_by_envelopes(outcomes, outcome_to_plot, time, density,
                               ax, ax_d, fill, grouping_labels, log)
        else:
            single_envelope(outcomes, outcome_to_plot, time, density,
                            ax, ax_d, fill, log)

        if ax_d:
            for tl in ax_d.get_yticklabels():
                tl.set_visible(False)

        ax.set_xlabel(TIME_LABEL)
        do_ylabels(ax, ylabels, outcome_to_plot)
        do_titles(ax, titles, outcome_to_plot)

    if legend and group_by:
        gs1 = grid[0, 0]

        for ax in figure.axes:
            gs2 = ax._subplotspec
            if all((gs1._gridspec == gs2._gridspec,
                    gs1.num1 == gs2.num1,
                    gs1.num2 == gs2.num2)):
                break
        if fill:
            make_legend(grouping_labels, ax, alpha=0.3,
                        legend_type=LegendEnum.PATCH)
        else:
            make_legend(grouping_labels, ax, legend_type=LegendEnum.LINE)

    return figure, axes_dict


def group_by_envelopes(outcomes, outcome_to_plot, time, density, ax,
                       ax_d, fill, group_labels, log):
    ''' Helper function responsible for generating an envelope plot
    based on a grouping.

    Parameters
    ----------
    outcomes : dict
               a dictionary containing the various outcomes to plot
    outcome_to_plot : str
                      the specific outcome to plot
    time : str
           the name of the time dimension
    density :  {None, HIST, KDE, VIOLIN, BOXPLOT}
    ax : Axes instance
         the ax on which to plot
    ax_d : Axes instance
           the ax on which to plot the density
    fill : bool
    group_by_labels : list of str
                      order in which groups should be plotted
    log : bool

    '''

    for j, key in enumerate(group_labels):
        value = outcomes[key]
        value = value[outcome_to_plot]
        try:
            plot_envelope(ax, j, time, value, fill)
        except ValueError:
            _logger.exception("value error when plotting for %s" % (key))
            raise

    if density:
        group_density(ax_d, density, outcomes, outcome_to_plot, group_labels,
                      log)

        ax_d.get_yaxis().set_view_interval(
            ax.get_yaxis().get_view_interval()[0],
            ax.get_yaxis().get_view_interval()[1])


def single_envelope(outcomes,
                    outcome_to_plot,
                    time,
                    density,
                    ax,
                    ax_d,
                    fill,
                    log):
    '''

    Helper function for generating a single envelope plot.

    Parameters
    ----------
    outcomes : dict
               a dictonary containing the various outcomes to plot
    outcome_to_plot : str
                      the specific outcome to plot
    time : str
           the name of the time dimension
    density :  {None, HIST, KDE, VIOLIN, BOXPLOT}
    ax : Axes instance
         the ax on which to plot
    ax_d : Axes instance
           the ax on which to plot the density
    fill : bool
    group_by_labels : list of str
                      order in which groups should be plotted
    log : bool

    '''
    value = outcomes[outcome_to_plot]

    plot_envelope(ax, 0, time, value, fill)
    if density:
        simple_density(density, value, ax_d, ax, log)


def lines(experiments,
          outcomes,
          outcomes_to_show=[],
          group_by=None,
          grouping_specifiers=None,
          density='',
          legend=True,
          titles={},
          ylabels={},
          experiments_to_show=None,
          show_envelope=False,
          log=False):
    '''This function takes the results from :meth:`perform_experiments` and
    visualizes these as line plots. It is thus to be used in case of time
    series data. The function will try to find a result labeled "TIME". If this
    is present, these values will be used on the X-axis. In case of Vensim
    models, TIME is present by default.

    Parameters
    ----------
    experiments : DataFrame
    outcomes : dict
    outcomes_to_show : list of str, optional
                       list of outcome of interest you want to plot. If empty,
                       all outcomes are plotted. **Note**:  just names.
    group_by : str, optional
               name of the column in the cases array to group results by.
               Alternatively, `index` can be used to use indexing arrays as the
               basis for grouping.
    grouping_specifiers : iterable or dict, optional
                          set of categories to be used as a basis for grouping
                          by. Grouping_specifiers is only meaningful if
                          group_by is provided as well. In case of grouping by
                          index, the grouping specifiers should be in a
                          dictionary where the key denotes the name of the
                          group.
    density : {None, HIST, KDE, VIOLIN, BOXPLOT}, optional
    legend : bool, optional
    titles : dict, optional
             a way for controlling whether each of the axes should have a
             title. There are three possibilities. If set to None, no title
             will be shown for any of the axes. If set to an empty dict,
             the default, the title is identical to the name of the outcome of
             interest. If you want to override these default names, provide a
             dict with the outcome of interest as key and the desired title as
             value. This dict need only contain the outcomes for which you
             want to use a different title.
    ylabels : dict, optional
              way for controlling the ylabels. Works identical to titles.
    experiments_to_show : ndarray, optional
                          indices of experiments to show lines for,
                          defaults to None.
    show_envelope : bool, optional
                    show envelope of outcomes. This envelope is the based on
                    the minimum at each column and the maximum at each column.
    log : bool, optional
          log scale density plot

    Returns
    -------
    fig : Figure instance
    axes : dict
           dict with outcome as key, and axes as value. Density axes' are
           indexed by the outcome followed by _density.

    Note
    ----
    the current implementation is limited to seven different categories in case
    of group_by, categories, and/or discretesize. This limit is due to the colors
    specified in COLOR_LIST.

    '''

    _logger.debug("generating line graph")

    # make sure we have the data

    if show_envelope:
        return plot_lines_with_envelopes(
            experiments,
            outcomes,
            outcomes_to_show=outcomes_to_show,
            group_by=group_by,
            legend=legend,
            density=density,
            grouping_specifiers=grouping_specifiers,
            experiments_to_show=experiments_to_show,
            titles=titles,
            ylabels=ylabels,
            log=log)

    if experiments_to_show is not None:
        experiments = experiments.loc[experiments_to_show, :]
        outcomes = {k: v[experiments_to_show] for k, v in outcomes.items()}

    data = prepare_data(experiments, outcomes, outcomes_to_show,
                        group_by, grouping_specifiers)
    outcomes, outcomes_to_show, time, grouping_labels = data

    figure, grid = make_grid(outcomes_to_show, density)
    axes_dict = {}

    # do the plotting
    for i, outcome_to_plot in enumerate(outcomes_to_show):
        ax = figure.add_subplot(grid[i, 0])
        axes_dict[outcome_to_plot] = ax

        ax_d = None
        if density:
            ax_d = figure.add_subplot(grid[i, 1], sharey=ax)
            axes_dict[outcome_to_plot + "_density"] = ax_d

            for tl in ax_d.get_yticklabels():
                tl.set_visible(False)

        if group_by:
            group_by_lines(outcomes, outcome_to_plot, time, density,
                           ax, ax_d, grouping_labels, log)
        else:
            simple_lines(outcomes, outcome_to_plot, time, density,
                         ax, ax_d, log)
        ax.set_xlabel(TIME_LABEL)
        do_ylabels(ax, ylabels, outcome_to_plot)
        do_titles(ax, titles, outcome_to_plot)

    if legend and group_by:
        gs1 = grid[0, 0]

        for ax in figure.axes:
            gs2 = ax._subplotspec
            if all((gs1._gridspec == gs2._gridspec,
                    gs1.num1 == gs2.num1,
                    gs1.num2 == gs2.num2)):
                break

        make_legend(grouping_labels, ax)

    return figure, axes_dict


def plot_lines_with_envelopes(experiments,
                              outcomes,
                              outcomes_to_show=[],
                              group_by=None,
                              grouping_specifiers=None,
                              density='',
                              legend=True,
                              titles={},
                              ylabels={},
                              experiments_to_show=None,
                              log=False):
    '''

    Helper function for generating a plot which contains both an envelope and
    lines.

    Parameters
    ----------
    experiments : DataFrame
    outcomes : dict
    outcomes_to_show : list of str, optional
                       list of outcome of interest you want to plot. If empty,
                       all outcomes are plotted. **Note**:  just names.
    group_by : str, optional
               name of the column in the cases array to group results by.
               Alternatively, `index` can be used to use indexing arrays as the
               basis for grouping.
    grouping_specifiers : iterable or dict, optional
                          set of categories to be used as a basis for grouping
                          by. Grouping_specifiers is only meaningful if
                          group_by is provided as well. In case of grouping by
                          index, the grouping specifiers should be in a
                          dictionary where the key denotes the name of the
                          group.
    density : {None, HIST, KDE, VIOLIN, BOXPLOT}, optional
    legend : bool, optional
    titles : dict, optional
             a way for controlling whether each of the axes should have a
             title. There are three possibilities. If set to None, no title
             will be shown for any of the axes. If set to an empty dict,
             the default, the title is identical to the name of the outcome of
             interest. If you want to override these default names, provide a
             dict with the outcome of interest as key and the desired title as
             value. This dict need only contain the outcomes for which you
             want to use a different title.
    ylabels : dict, optional
              way for controlling the ylabels. Works identical to titles.
    experiments_to_show : ndarray, optional
                          indices of experiments to show lines for,
                          defaults to None.
    log : bool, optional

    Returns
    -------
    Figure
        a figure instance
    dict
        dict with outcome as key, and axes as value. Density axes' are
        indexed by the outcome followed by _density
    '''
    full_outcomes = prepare_data(experiments, outcomes,
                                 outcomes_to_show, group_by,
                                 grouping_specifiers)[0]

    experiments = experiments.loc[experiments_to_show, :]
    temp = {}
    for key, value in outcomes.items():
        temp[key] = value[experiments_to_show]

    data = prepare_data(experiments, temp, outcomes_to_show,
                        group_by, grouping_specifiers)
    outcomes, outcomes_to_show, time, grouping_labels = data

    figure, grid = make_grid(outcomes_to_show, density)
    axes_dict = {}

    # do the plotting
    for i, outcome_to_plot in enumerate(outcomes_to_show):
        ax = figure.add_subplot(grid[i, 0])
        axes_dict[outcome_to_plot] = ax

        ax_d = None
        if density:
            ax_d = figure.add_subplot(grid[i, 1], sharey=ax)
            axes_dict[outcome_to_plot + "_density"] = ax_d

            for tl in ax_d.get_yticklabels():
                tl.set_visible(False)

        if group_by:
            for j, key in enumerate(grouping_labels):
                full_value = full_outcomes[key][outcome_to_plot]
                plot_envelope(ax, j, time, full_value, fill=True)
            for j, key in enumerate(grouping_labels):
                value = outcomes[key][outcome_to_plot]
                full_value = full_outcomes[key][outcome_to_plot]
                ax.plot(time.T[:, np.newaxis], value.T,
                        c=get_color(j))

            if density:
                group_density(ax_d, density, full_outcomes,
                              outcome_to_plot, grouping_labels, log)

                ax_d.get_yaxis().set_view_interval(
                    ax.get_yaxis().get_view_interval()[0],
                    ax.get_yaxis().get_view_interval()[1])

        else:
            value = full_outcomes[outcome_to_plot]
            plot_envelope(ax, 0, time, value, fill=True)
            if density:
                simple_density(density, value, ax_d, ax, log)

            value = outcomes[outcome_to_plot]
            ax.plot(time.T, value.T)

        ax.set_xlim(left=time[0], right=time[-1])
        ax.set_xlabel(TIME_LABEL)
        do_ylabels(ax, ylabels, outcome_to_plot)
        do_titles(ax, titles, outcome_to_plot)

    if legend and group_by:
        gs1 = grid[0, 0]

        for ax in figure.axes:
            gs2 = ax._subplotspec
            if all((gs1._gridspec == gs2._gridspec,
                    gs1.num1 == gs2.num1,
                    gs1.num2 == gs2.num2)):
                break
        make_legend(grouping_labels, ax)

    return figure, axes_dict


def group_by_lines(outcomes, outcome_to_plot, time, density,
                   ax, ax_d, group_by_labels, log):
    '''

    Helper function responsible for generating a grouped lines plot.

    Parameters
    ----------
    results : tupule
              return from :meth:`perform_experiments`.
    outcome_to_plot : str
    time : str
    density : {None, HIST, KDE, VIOLIN, BOXPLOT}
    ax : Axes instance
    ax_d : Axes instance
    group_by_labels : list of str
    log : bool

    '''

    for j, key in enumerate(group_by_labels):
        value = outcomes[key]
        value = value[outcome_to_plot]

        color = get_color(j)
        ax.plot(time.T[:, np.newaxis], value.T, c=color, ms=1, markevery=5)

    if density:
        group_density(ax_d, density, outcomes, outcome_to_plot,
                      group_by_labels, log)

        ax_d.get_yaxis().set_view_interval(
            ax.get_yaxis().get_view_interval()[0],
            ax.get_yaxis().get_view_interval()[1])


def simple_lines(outcomes, outcome_to_plot, time, density,
                 ax, ax_d, log):
    '''

    Helper function responsible for generating a simple lines plot.

    Parameters
    ----------
    outcomes : dict
    outcomes_to_plot : str
    time : str
    density : {None, HIST, KDE, VIOLIN, BOXPLOT}
    ax : Axes instance
    ax_d : Axes instance
    log : bool

    '''
    value = outcomes[outcome_to_plot]
    ax.plot(time.T, value.T)
    if density:
        simple_density(density, value, ax_d, ax, log)


def kde_over_time(experiments,
                  outcomes,
                  outcomes_to_show=[],
                  group_by=None,
                  grouping_specifiers=None,
                  colormap='viridis',
                  log=True):
    '''

    Plot a KDE over time. The KDE is is visualized through a heatmap

    Parameters
    ----------
    experiments : DataFrame
    outcomes : dict
    outcomes_to_show : list of str, optional
                       list of outcome of interest you want to plot. If
                       empty, all outcomes are plotted.
                       **Note**:  just names.
    group_by : str, optional
               name of the column in the cases array to group results
               by. Alternatively, `index` can be used to use indexing
               arrays as the basis for grouping.
    grouping_specifiers : iterable or dict, optional
                          set of categories to be used as a basis for
                          grouping by. Grouping_specifiers is only
                          meaningful if group_by is provided as well.
                          In case of grouping by index, the grouping
                          specifiers should be in a dictionary where
                          the key denotes the name of the group.
    colormap : str, optional
               valid matplotlib color map name
    log : bool, optional

    Returns
    -------
    list of Figure instances
        a figure instance for each group for each outcome
    dict
        dict with outcome as key, and axes as value. Density axes' are
        indexed by the outcome followed by _density

    '''

    # determine the minima and maxima over all runs
    minima = {}
    maxima = {}
    for key, value in outcomes.items():
        minima[key] = np.min(value)
        maxima[key] = np.max(value)

    prepared_data = prepare_data(experiments, outcomes,
                                 outcomes_to_show, group_by,
                                 grouping_specifiers,
                                 filter_scalar=True)
    outcomes, outcomes_to_show, time, grouping_specifiers = prepared_data
    del time

    if group_by:
        figures = []
        axes_dicts = {}
        for key, value in outcomes.items():
            fig, axes_dict = simple_kde(value, outcomes_to_show,
                                        colormap, log, minima, maxima)
            fig.suptitle(key)
            figures.append(fig)
            axes_dicts[key] = axes_dict

        return figures, axes_dicts
    else:
        return simple_kde(outcomes, outcomes_to_show, colormap, log,
                          minima, maxima)


def multiple_densities(experiments,
                       outcomes,
                       points_in_time=[],
                       outcomes_to_show=[],
                       group_by=None,
                       grouping_specifiers=None,
                       density=Density.KDE,
                       legend=True,
                       titles={},
                       ylabels={},
                       experiments_to_show=None,
                       plot_type=PlotType.ENVELOPE,
                       log=False,
                       **kwargs):
    ''' Make an envelope plot with multiple density plots over the run time

    Parameters
    ----------
    experiments : DataFrame
    outcomes : dict
    points_in_time : list
                     a list of points in time for which you want to see the
                     density. At the moment  up to 6 points in time are
                     supported
    outcomes_to_show : list of str, optional
                       list of outcome of interest you want to plot. If empty,
                       all outcomes are plotted. **Note**:  just names.
    group_by : str, optional
               name of the column in the cases array to group results by.
               Alternatively, `index` can be used to use indexing arrays as the
               basis for grouping.
    grouping_specifiers : iterable or dict, optional
                          set of categories to be used as a basis for grouping
                          by. Grouping_specifiers is only meaningful if
                          group_by is provided as well. In case of grouping by
                          index, the grouping specifiers should be in a
                          dictionary where the key denotes the name of the
                          group.
    density : {Density.KDE, Density.HIST, Density.VIOLIN, Density.BOXPLOT},
              optional
    legend : bool, optional
    titles : dict, optional
             a way for controlling whether each of the axes should have a
             title. There are three possibilities. If set to None, no title
             will be shown for any of the axes. If set to an empty dict,
             the default, the title is identical to the name of the outcome of
             interest. If you want to override these default names, provide a
             dict with the outcome of interest as key and the desired title as
             value. This dict need only contain the outcomes for which you
             want to use a different title.
    ylabels : dict, optional
              way for controlling the ylabels. Works identical to titles.
    experiments_to_show : ndarray, optional
                          indices of experiments to show lines for,
                          defaults to None.
    plot_type : {PlotType.ENVELOPE, PlotType.ENV_LIN, PlotType.LINES}, optional
    log : bool, optional

    Returns
    -------
    fig : Figure instance
    axes : dict
           dict with outcome as key, and axes as value. Density axes' are
           indexed by the outcome followed by _density.

    Note
    ----
    the current implementation is limited to seven different categories
    in case of group_by, categories, and/or discretesize. This limit is
    due to the colors specified in COLOR_LIST.

    Note
    ----
    the connection patches are for some reason not drawn if log scaling is
    used for the density plots. This appears to be an issue in matplotlib
    itself.

    '''
    if not outcomes_to_show:
        outcomes_to_show = [k for k, v in outcomes.items() if v.ndim == 2]
        outcomes_to_show.remove(TIME)
    elif isinstance(outcomes_to_show, str):
        outcomes_to_show = [outcomes_to_show]

    data = prepare_data(experiments, outcomes,
                        outcomes_to_show, group_by,
                        grouping_specifiers)
    outcomes, _, time, grouping_labels = data

    axes_dicts = {}
    figures = []
    for outcome_to_show in outcomes_to_show:
        axes_dict = {}
        axes_dicts[outcome_to_show] = axes_dict

        # start of plotting
        fig = plt.figure()
        figures.append(fig)

        # making of grid
        if not points_in_time:
            raise EMAError("no points in time specified")
        if len(points_in_time) == 1:
            ax_env = plt.subplot2grid((2, 3), (0, 0), colspan=3)
            ax1 = plt.subplot2grid((2, 3), (1, 1), sharey=ax_env)
            kde_axes = [ax1]
        elif len(points_in_time) == 2:
            ax_env = plt.subplot2grid((2, 2), (0, 0), colspan=2)
            ax1 = plt.subplot2grid((2, 2), (1, 0), sharey=ax_env)
            ax2 = plt.subplot2grid((2, 2), (1, 1), sharex=ax1, sharey=ax_env)
            kde_axes = [ax1, ax2]
        elif len(points_in_time) == 3:
            ax_env = plt.subplot2grid((2, 3), (0, 0), colspan=3)
            ax1 = plt.subplot2grid((2, 3), (1, 0), sharey=ax_env)
            ax2 = plt.subplot2grid((2, 3), (1, 1), sharex=ax1, sharey=ax_env)
            ax3 = plt.subplot2grid((2, 3), (1, 2), sharex=ax1, sharey=ax_env)
            kde_axes = [ax1, ax2, ax3]
        elif len(points_in_time) == 4:
            ax_env = plt.subplot2grid((2, 4), (0, 1), colspan=2)
            ax1 = plt.subplot2grid((2, 4), (1, 0), sharey=ax_env)
            ax2 = plt.subplot2grid((2, 4), (1, 1), sharex=ax1, sharey=ax_env)
            ax3 = plt.subplot2grid((2, 4), (1, 2), sharex=ax1, sharey=ax_env)
            ax4 = plt.subplot2grid((2, 4), (1, 3), sharex=ax1, sharey=ax_env)
            kde_axes = [ax1, ax2, ax3, ax4]
        elif len(points_in_time) == 5:
            ax_env = plt.subplot2grid((2, 5), (0, 1), colspan=3)
            ax1 = plt.subplot2grid((2, 5), (1, 0), sharey=ax_env)
            ax2 = plt.subplot2grid((2, 5), (1, 1), sharex=ax1, sharey=ax_env)
            ax3 = plt.subplot2grid((2, 5), (1, 2), sharex=ax1, sharey=ax_env)
            ax4 = plt.subplot2grid((2, 5), (1, 3), sharex=ax1, sharey=ax_env)
            ax5 = plt.subplot2grid((2, 5), (1, 4), sharex=ax1, sharey=ax_env)
            kde_axes = [ax1, ax2, ax3, ax4, ax5]
        elif len(points_in_time) == 6:
            ax_env = plt.subplot2grid((2, 6), (0, 1), colspan=4)
            ax1 = plt.subplot2grid((2, 6), (1, 0), sharey=ax_env)
            ax2 = plt.subplot2grid((2, 6), (1, 1), sharex=ax1, sharey=ax_env)
            ax3 = plt.subplot2grid((2, 6), (1, 2), sharex=ax1, sharey=ax_env)
            ax4 = plt.subplot2grid((2, 6), (1, 3), sharex=ax1, sharey=ax_env)
            ax5 = plt.subplot2grid((2, 6), (1, 4), sharex=ax1, sharey=ax_env)
            ax6 = plt.subplot2grid((2, 6), (1, 5), sharex=ax1, sharey=ax_env)
            kde_axes = [ax1, ax2, ax3, ax4, ax5, ax6, ]
        else:
            raise EMAError("too many points in time provided")

        axes_dict["main plot"] = ax_env
        for n, entry in enumerate(kde_axes):
            axes_dict["density_%s" % n] = entry

            # turn of ticks for all but the first density
            if n > 0:
                for tl in entry.get_yticklabels():
                    tl.set_visible(False)

        # bit of a trick to avoid duplicating code. If no subgroups are
        # specified, nest the outcomes one step deeper in the dict so the
        # iteration below can proceed normally.
        if not grouping_labels:
            grouping_labels = [""]
            outcomes[""] = outcomes

        for j, key in enumerate(grouping_labels):
            value = outcomes[key][outcome_to_show]

            if plot_type == PlotType.ENVELOPE:
                plot_envelope(ax_env, j, time, value, **kwargs)
            elif plot_type == PlotType.LINES:
                ax_env.plot(time.T, value.T)
            elif plot_type == PlotType.ENV_LIN:
                plot_envelope(ax_env, j, time, value, **kwargs)
                if experiments_to_show is not None:
                    ax_env.plot(time.T, value[experiments_to_show].T)
                else:
                    ax_env.plot(time.T, value.T)
            ax_env.set_xlim(time[0], time[-1])

            ax_env.set_xlabel(TIME_LABEL)
            do_ylabels(ax_env, ylabels, outcome_to_show)
            do_titles(ax_env, titles, outcome_to_show)

        for ax, time_value in zip(kde_axes, points_in_time):
            index = np.where(time == time_value)[0][0]

            group_density(ax, density, outcomes, outcome_to_show,
                          grouping_labels, index=index, log=log)

        min_y, max_y = ax_env.get_ylim()
        ax_env.autoscale(enable=False, axis='y')

        # draw line to connect each point in time in the main plot
        # to the associated density plot
        for i, ax in enumerate(kde_axes):
            time_value = points_in_time[i]

            ax_env.plot([time_value, time_value],
                        [min_y, max_y], c='k', ls='--')
            con = ConnectionPatch(xyA=(time_value, min_y),
                                  xyB=(ax.get_xlim()[0],
                                       max_y), coordsA="data",
                                  coordsB="data", axesA=ax_env, axesB=ax)
            ax_env.add_artist(con)

        if legend and group_by:
            lt = LegendEnum.PATCH
            alpha = 0.3
            if plot_type == PlotType.LINES:
                lt = LegendEnum.LINE
                alpha = 1
            make_legend(grouping_labels, ax_env, legend_type=lt, alpha=alpha)
    return figures, axes_dicts
