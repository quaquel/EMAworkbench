"""
.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


this module provides functions for generating some basic figures. The code can
be used as is, or serve as an example for writing your own code. These plots
rely on `matplotlib <http://matplotlib.sourceforge.net/>`_, 
`numpy <http://numpy.scipy.org/>`_, and 
`scipy.stats.kde <http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html#scipy.stats.gaussian_kde>`_

"""
from __future__ import division
import copy
from types import StringType

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from expWorkbench.ema_logging import debug, warning
from expWorkbench.ema_exceptions import EMAError
from plotting_util import prepare_data, COLOR_LIST, simple_kde,\
                         determine_kde, make_grid, make_legend, plot_envelope,\
                         plot_kde, plot_histogram, simple_density, do_titles,\
                         do_ylabels, TIME, plot_boxplots
import plotting_util


__all__ = ['lines', 'envelopes', 'kde_over_time', 'ENVELOPE', 'LINES', 
           'ENV_LIN', 'KDE', 'HIST', 'BOXPLOT', 'multiple_densities']

ENVELOPE = 'envelope'
LINES = 'lines'
ENV_LIN = "env_lin"
KDE = 'kde'
HIST = 'hist'
BOXPLOT = 'box plot'

TIME_LABEL = "time"

def envelopes(results, 
              outcomes_to_show = [],
              group_by = None,
              grouping_specifiers = None,
              density='',
              fill=False,
              legend=True,
              titles={},
              ylabels={},
              **kwargs):
    '''
    
    Make envelop plots. An envelope shows over time the minimum and maximum 
    value for a set of runs over time. It is thus to be used in case of time 
    series data. The function will try to find a result labeled "TIME". If this
    is present, these values will be used on the X-axis. In case of Vensim 
    models, TIME is present by default.  
    
    :param results: return from :meth:`perform_experiments`.
    :param outcomes_to_show: list of outcome of interest you want to plot. If 
                             empty, all outcomes are plotted. **Note**:  just 
                             names.
    :param group_by: name of the column in the cases array to group results by.
                     Alternatively, `index` can be used to use indexing arrays 
                     as the basis for grouping.
    :param grouping_specifiers: set of categories to be used as a basis for 
                                grouping by. Grouping_specifiers is only 
                                meaningful if group_by is provided as well. In
                                case of grouping by index, the grouping 
                                specifiers should be in a dictionary where the
                                key denotes the name of the group. 
    :param density: boolean, if true, the density of the endstates will be 
                    plotted.
    :param fill: boolean, if true, fill the envelope. 
    :param legend: boolean, if true, and there is a column specified for 
                   grouping, show a legend.
    :param titles: a way for controlling whether each of the axes should have
                   a title. There are three possibilities. If set to None, no
                   title will be shown for any of the axes. If set to an empty 
                   dict, the default, the title is identical to the name of the
                   outcome of interest. If you want to override these default 
                   names, provide a dict with the outcome of interest as key 
                   and the desired title as value. This dict need only contain
                   the outcomes for which you want to use a different title. 
    :param ylabels: a way for controlling the ylabels. Works identical to 
                    titles.
    :rtype: a `figure <http://matplotlib.sourceforge.net/api/figure_api.html>`_ instance
            and a dict with the individual axes.
            
    Additional key word arguments will be passed along to the density function,
    if density is `True`.
    
    ======== ===================================
    property description
    ======== ===================================
    log      log the resulting histogram or GKDE
    ======== ===================================
    
    .. rubric:: an example of use
    
    >>> import expWorkbench.util as util
    >>> data = util.load_results(r'1000 flu cases.cPickle')
    >>> envelopes(data, column='policy')
    
    will show an envelope for three three different policies, for all the 
    outcomes of interest.
    
    .. plot:: ../docs/source/pyplots/basicEnvelope.py
   
     while

    >>> envelopes(data, column='policy', categories=['static policy', 'adaptive policy'])
    
    will only show results for the two specified policies, ignoring any results 
    associated with \'no policy\'.

    .. plot:: ../docs/source/pyplots/basicEnvelope2.py
    
    .. note:: the current implementation is limited to seven different 
              categories in case of column, categories, and/or discretesize.
              This limit is due to the colors specified in COLOR_LIST.
    '''
    debug("generating envelopes")
   
    prepared_data = prepare_data(results, outcomes_to_show, group_by,
                                 grouping_specifiers)
    outcomes, outcomes_to_show, time, grouping_labels = prepared_data
    
    figure, grid = make_grid(outcomes_to_show, density)
    
    # do the plotting
    axes_dict = {}
    for i, outcome_to_plot in enumerate(outcomes_to_show):
        ax = figure.add_subplot(grid[i,0])
        axes_dict[outcome_to_plot] = ax
        
        ax_d= None
        if density:
            ax_d = figure.add_subplot(grid[i,1], sharey=ax)
            axes_dict[outcome_to_plot+"_density"] = ax_d
    
        if group_by:
            group_by_envelopes(outcomes,outcome_to_plot, time, density,
                               ax, ax_d, fill, grouping_labels, **kwargs)
        else:
            single_envelope(outcomes, outcome_to_plot, time, density,
                            ax, ax_d, fill, **kwargs)
            
        if ax_d:
            for tl in ax_d.get_yticklabels():
                tl.set_visible(False)
        
        ax.set_xlabel(TIME_LABEL)
        do_ylabels(ax, ylabels, outcome_to_plot)
        do_titles(ax, titles, outcome_to_plot)
        
    if legend and group_by:
        if fill:
            make_legend(grouping_labels, figure, alpha=0.3, 
                        legend_type=plotting_util.PATCH)
        else:
            make_legend(grouping_labels, figure, legend_type=plotting_util.LINE)
    
    if plotting_util.TIGHT:
        grid.tight_layout(figure)
    
    return figure, axes_dict

def group_by_envelopes(outcomes,
                       outcome_to_plot,
                       time,
                       density,
                       ax,
                       ax_d,
                       fill,
                       group_labels,
                       **kwargs):
    '''
    
    Helper function, responsible for generating an envelope plot based on
    a grouping. 
    
    :param outcomes: a dictonary containing the various outcomes to plot
    :param outcome_to_plot: the specific outcome to plot
    :param time: the name of the time dimension
    :param density: string, either hist, kde, or empty/None.    
    :param ax: the ax on which to plot
    :param ax_d: the ax on which to plot the density
    :param fill: boolean, if true, fill the envelope. 
    :param group_by_labels: order in which groups should be plotted
    :param kwargs: kwargs to be passed on to the helper function for plotting
                   the density.
    
    '''
    
    for j, key in enumerate(group_labels):
        value = outcomes[key]
        value = value[outcome_to_plot]
        try:
            plot_envelope(ax, j, time, value,fill)
        except ValueError:
            warning("value error when plotting for %s" % (key))
            raise
    
        if density=='kde':
            kde_x, kde_y = determine_kde(value[:,-1])
            plot_kde(ax_d, kde_x, kde_y, j, **kwargs)
    
    if density:
        if density=='hist':
            # rather nasty indexing going on here, outcomes[key] returns
            # a tuple, hence the[1] to get the dictionary with outcomes
            # out of this, we need the right outcome, and the final column
            # of values
            values = [outcomes[key][outcome_to_plot][:,-1] for key in group_labels]
            plot_histogram(ax_d, values, **kwargs)
        if density=='box plot':
            values = [outcomes[key][outcome_to_plot][:,-1] for key in group_labels]
            plot_boxplots(ax_d, values, group_labels, **kwargs)
        
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
                    **kwargs):
    '''
    
    Helper function for generating a single envelope plot.

    :param outcomes: a dictonary containing the various outcomes to plot
    :param outcome_to_plot: the specific outcome to plot
    :param time: the name of the time dimension
    :param density: string, either hist, kde, or empty/None.    
    :param ax: the ax on which to plot
    :param ax_d: the ax on which to plot the density
    :param fill: boolean, if true, fill the envelope. 
    :param kwargs: kwargs to be passed on to the hlepr function for plotting
                   the density.
    
    '''
    value = outcomes[outcome_to_plot]
    
    plot_envelope(ax, 0, time, value,fill)
    if density:
        simple_density(density, value, ax_d, ax, **kwargs)

          
def lines(results, 
          outcomes_to_show = [],
          group_by = None,
          grouping_specifiers = None,
          density='',
          titles={},
          ylabels={},
          legend=True,
          experiments_to_show=None,
          show_envelope=False,
          **kwargs):
    '''
    
    This function takes the results from :meth:`perform_experiments` and 
    visualizes these as line plots. It is thus to be used in case of time 
    series data. The function will try to find a result labeled "TIME". If this
    is present, these values will be used on the X-axis. In case of Vensim 
    models, TIME is present by default.  

    :param results: return from :meth:`perform_experiments`.
    :param outcomes_to_show: list of outcome of interest you want to plot. If 
                             empty, all outcomes are plotted. **Note**:  just 
                             names.
    :param group_by: name of the column in the cases array to group results by.
                     Alternatively, `index` can be used to use indexing arrays 
                     as the basis for grouping.
    :param grouping_specifiers: set of categories to be used as a basis for 
                                grouping by. Grouping_specifiers is only 
                                meaningful if group_by is provided as well. In
                                case of grouping by index, the grouping 
                                specifiers should be in a dictionary where the
                                key denotes the name of the group. 
    :param density: boolean, if true, the density of the endstates will be 
                    plotted.
    :param legend: boolean, if true, and there is a column specified for 
                   grouping, show a legend.
    :param titles: a way for controlling whether each of the axes should have
                   a title. There are three possibilities. If set to None, no
                   title will be shown for any of the axes. If set to an empty 
                   dict, the default, the title is identical to the name of the
                   outcome of interest. If you want to override these default 
                   names, provide a dict with the outcome of interest as key 
                   and the desired title as value. This dict need only contain
                   the outcomes for which you want to use a different title. 
    :param ylabels: a way for controlling the ylabels. Works identical to 
                    titles.
    :param experiments_to_show: numpy array containing the indices of the 
                                experiments to be visualized. Defaults to None,
                                implying that all experiments should be shown.
    :param show_envelope: boolean, indicates whether envelopes should be 
                          plotted in combination with lines. Default is False.
    :rtype: a `figure <http://matplotlib.sourceforge.net/api/figure_api.html>`_ instance
            and a dict with the individual axes.
   
    .. note:: the current implementation is limited to seven different 
          categories in case of column, categories, and/or discretesize.
          This limit is due to the colors specified in COLOR_LIST.
    '''
    
    debug("generating line graph")

    if show_envelope:
        return plot_lines_with_envelopes(results, 
                                outcomes_to_show=outcomes_to_show, 
                                group_by=group_by, legend=legend, density=density,    
                                grouping_specifiers=grouping_specifiers, 
                                experiments_to_show=experiments_to_show, 
                                titles=titles, ylabels=ylabels, **kwargs)
    
    if experiments_to_show != None:
        experiments, outcomes = results
        experiments = experiments[experiments_to_show]
        new_outcomes = {}
        for key, value in outcomes.items():
            new_outcomes[key] = value[experiments_to_show]
        results = experiments, new_outcomes

    data = prepare_data(results, outcomes_to_show, group_by, 
                        grouping_specifiers)
    outcomes, outcomes_to_show, time, grouping_labels = data

    figure, grid = make_grid(outcomes_to_show, density)
    axes_dict = {}

    # do the plotting
    for i, outcome_to_plot in enumerate(outcomes_to_show):
        ax = figure.add_subplot(grid[i,0])
        axes_dict[outcome_to_plot] = ax
        
        ax_d= None
        if density:
            ax_d = figure.add_subplot(grid[i,1], sharey=ax)
            axes_dict[outcome_to_plot+"_density"] = ax_d
            
            for tl in ax_d.get_yticklabels():
                tl.set_visible(False)
    
        if group_by:
            group_by_lines(outcomes,outcome_to_plot, time, density,
                           ax, ax_d, grouping_labels, **kwargs)
        else:
            simple_lines(outcomes, outcome_to_plot, time, density,
                         ax, ax_d, **kwargs)
        ax.set_xlabel(TIME_LABEL)
        do_ylabels(ax, ylabels, outcome_to_plot)
        do_titles(ax, titles, outcome_to_plot)
            
    if legend and group_by:
        make_legend(grouping_labels, figure)
    
    if plotting_util.TIGHT:
        grid.tight_layout(figure)
    
    return figure, axes_dict

def plot_lines_with_envelopes(results, 
                              outcomes_to_show = [],
                              group_by = None,
                              grouping_specifiers = None,
                              density='',
                              titles={},
                              ylabels={},
                              legend=True,
                              experiments_to_show=None,
                              **kwargs):
    '''
    
    Helper function for generating a plot which contains both an envelope and
    lines.  

    :param results: return from :meth:`perform_experiments`.
    :param outcomes_to_show: list of outcome of interest you want to plot. If 
                             empty, all outcomes are plotted. **Note**:  just 
    :param group_by: name of the column in the cases array to group results by.
                     Alternatively, `index` can be used to use indexing arrays 
                     as the basis for grouping.
    :param grouping_specifiers: set of categories to be used as a basis for 
                                grouping by. Grouping_specifiers is only 
                                meaningful if group_by is provided as well. In
                                case of grouping by index, the grouping 
                                specifiers should be in a dictonary where the
                                key denotes the name of the group. 
    :param density: boolean, if true, the density of the endstates will be 
                    plotted.
    :param titles: a way for controlling whether each of the axes should have
               a title. There are three possibilities. If set to None, no
               title will be shown for any of the axes. If set to an empty 
               dict, the default, the title is identical to the name of the
               outcome of interest. If you want to override these default 
               names, provide a dict with the outcome of interest as key 
               and the desired title as value. This dict need only contain
               the outcomes for which you want to use a different title. 
    :param ylabels: a way for controlling the ylablels. Works identical to 
                    titles.
    :param legend: boolean, if true, and there is a column specified for 
                   grouping, show a legend
    :param experiments_to_show: numpy array containing the indices of the 
                                experiments to be visualized. Defaults to None,
                                implying that all experiments should be shown.
    :rtype: a `figure <http://matplotlib.sourceforge.net/api/figure_api.html>`_ instance

    Additional key word arguments will be passed along to the density function.
    
    ======== ===============================
    property description
    ======== ===============================
    log      log scale the histogram or GKDE
    ======== ===============================
    
    '''
   
    # make sure we have the data
    full_results = copy.deepcopy(results)
      
    experiments, outcomes = results
    experiments = experiments[experiments_to_show]
    new_outcomes={}
    for key, value in outcomes.items():
        new_outcomes[key] =value[experiments_to_show]
    results = experiments, new_outcomes

    data = prepare_data(results, outcomes_to_show, group_by, 
                        grouping_specifiers)
    outcomes, outcomes_to_show, time, grouping_labels = data
    
    full_outcomes = prepare_data(full_results, outcomes_to_show, group_by,
                             grouping_specifiers)[0]

    figure, grid = make_grid(outcomes_to_show, density)
    axes_dict = {}

    # do the plotting
    for i, outcome_to_plot in enumerate(outcomes_to_show):
        ax = figure.add_subplot(grid[i,0])
        axes_dict[outcome_to_plot] = ax
        
        ax_d= None
        if density:
            ax_d = figure.add_subplot(grid[i,1], sharey=ax)
            axes_dict[outcome_to_plot+"_density"] = ax_d
            
            for tl in ax_d.get_yticklabels():
                tl.set_visible(False)
    
        if group_by:
#            group_by_labels = sorted(outcomes.keys())
            for j, key in enumerate(grouping_labels):
                full_value = full_outcomes[key][outcome_to_plot]
                plot_envelope(ax, j, time, full_value, fill=True)
            for j, key in enumerate(grouping_labels):
                value = outcomes[key][outcome_to_plot]
                full_value = full_outcomes[key][outcome_to_plot]
                ax.plot(time.T[:, np.newaxis], value.T, COLOR_LIST[j])
                if density=='kde':
#                    simple_density(density, full_value, ax_d, ax, **kwargs)
                    kde_x, kde_y = determine_kde(full_value[:,-1])
                    plot_kde(ax_d, kde_x, kde_y, j, **kwargs)
            
            if density:
                if density=='hist':
                    values = [full_outcomes[key][outcome_to_plot][:,-1]\
                                                    for key in grouping_labels]
                    plot_histogram(ax_d, values, **kwargs)

                if density=='box plot':
                    values = [full_outcomes[key][outcome_to_plot][:,-1]\
                                                    for key in grouping_labels]
                    plot_boxplots(ax_d, values, grouping_labels, **kwargs)
               
                ax_d.get_yaxis().set_view_interval(
                             ax.get_yaxis().get_view_interval()[0],
                             ax.get_yaxis().get_view_interval()[1])
            
        else:
            value = full_outcomes[outcome_to_plot]
            plot_envelope(ax, 0, time, value, fill=True)
            if density:
                simple_density(density, value, ax_d, ax, **kwargs)
            
            value = outcomes[outcome_to_plot]
            ax.plot(time.T, value.T)
        
        ax.set_xlim(xmin=time[0] , xmax=time[-1])
        ax.set_xlabel(TIME_LABEL)
        do_ylabels(ax, ylabels, outcome_to_plot)
        do_titles(ax, titles, outcome_to_plot)

    if legend and group_by:
        make_legend(grouping_labels, figure)
    
    if plotting_util.TIGHT:
        grid.tight_layout(figure)
    
    return figure, axes_dict


def group_by_lines(outcomes, outcome_to_plot, time, density,
                   ax, ax_d, group_by_labels, **kwargs):
    '''
    
    Helper function responsible for generating a grouped lines plot. 
 
    :param outcomes: a dictonary containing the various outcomes to plot
    :param outcome_to_plot: the specific outcome to plot
    :param time: the name of the time dimension
    :param density: string, either hist, kde, or empty/None. 
    :param ax: the ax on which to plot
    :param ax_d: the ax on which to plot the density
    :param group_by_labels: order in which groups should be plotted
    :param kwargs: kwargs to be passed on to the hlepr function for plotting
                   the density.
        
    '''
    
    for j, key in enumerate(group_by_labels):
        value = outcomes[key]
        value = value[outcome_to_plot]

        color = COLOR_LIST[j]
        ax.plot(time.T[:, np.newaxis], value.T, c=color, ms=1, markevery=5)
        if density=='kde':
            kde_x, kde_y = determine_kde(value[:,-1])
            plot_kde(ax_d, kde_x, kde_y, j, **kwargs)
    
    if density:
        if density=='hist':
            values = [outcomes[key][outcome_to_plot][:,-1] for key in group_by_labels]
            plot_histogram(ax_d, values, **kwargs)
        if density=='box plot':
            values = [outcomes[key][outcome_to_plot][:,-1]\
                                            for key in group_by_labels]
            plot_boxplots(ax_d, values, group_by_labels, **kwargs)        
        
        ax_d.get_yaxis().set_view_interval(
                     ax.get_yaxis().get_view_interval()[0],
                     ax.get_yaxis().get_view_interval()[1])

def simple_lines(outcomes, outcome_to_plot, time, density,
                 ax, ax_d, **kwargs):
    '''
    
    Helper function responsible for generating a simple lines plot. 
    
    :param outcomes: a dictonary containing the various outcomes to plot
    :param outcome_to_plot: the specific outcome to plot
    :param time: the name of the time dimension
    :param density: string, either hist, kde, or empty/None. 
    :param ax: the ax on which to plot
    :param ax_d: the ax on which to plot the density
    :param kwargs: kwargs to be passed on to the hlepr function for plotting
                   the density.
    
    '''    
    value = outcomes[outcome_to_plot]
    ax.plot(time.T, value.T)
    if density:
        simple_density(density, value, ax_d, ax, **kwargs)

def kde_over_time(results, 
                  outcomes_to_show = [],
                  group_by = None,
                  grouping_specifiers = None,
                  results_to_show=None,
                  colormap='jet',
#                  color_bar=False,
                  log=True):
    '''
    
    This is the 2d equivalent of 3d envelopes, where the density is visualized
    through a heatmap, rather then in the third dimension. 
    
    :param results: return from :meth:`perform_experiments`.
    :param outcomes_to_show: list of outcome of interest you want to plot. If 
                             empty, all outcomes are plotted. **Note**:  just 
                             names.
    :param group: name of the column in the cases array to group results by.
    :param grouping_specifiers: set of categories to be used as a basis for 
                                grouping by. Categories is only meaningful if 
                                column is provided as well. **Note**: grouping
                                specifiers should be an iterable.
    :param colormap:
    :param log:
    
    TODO:: a colorbar boolean should be added. This controls whether a
           colorbar is shown for each axes.
    
    '''
    
    #determine the minima and maxima over all runs
    minima = {}
    maxima = {}
    for key, value in results[1].items():
        minima[key] = np.min(value)
        maxima[key] = np.max(value)
    
    prepared_data = prepare_data(results, outcomes_to_show, group_by, 
                                 grouping_specifiers)
    outcomes, outcomes_to_show, time, grouping_specifiers = prepared_data
    del time
    
    if group_by:
        figures = []
        axes_dicts = {}
        for key, value in outcomes.items():
            fig, axes_dict = simple_kde(value, outcomes_to_show, colormap, log, minima, 
                                        maxima)
            fig.suptitle(key)
            figures.append(fig)
            axes_dicts[key] = axes_dict
        
        for outcome in outcomes_to_show:
            vmax = -1
            for entry in axes_dicts.values():
                vmax =  max(entry[outcome].images[0].norm.vmax, vmax)
            for entry in axes_dicts.values():
                ax = entry[outcome]
                ax.images[0].set_clim(vmin=0, vmax=vmax)
            del vmax
        
        return figures, axes_dicts
    else:
        return simple_kde(outcomes, outcomes_to_show, colormap, log, minima,
                          maxima)
        
def multiple_densities(results, 
                       outcomes_to_show=[],
                       points_in_time=[],
                       group_by = None,
                       grouping_specifiers = None,
                       density=KDE,
                       titles={},
                       ylabels={},
                       legend=True,
                       experiments_to_show=None,
                       plot_type = ENVELOPE,
                       **kwargs):
    '''
    Make an envelope plot with multiple density plots over the run time
    
    :param results: return from :meth:`perform_experiments`.
    :param outcomes_to_show: list of outcome of interest you want to plot. If 
                             empty, all outcomes are plotted. **Note**:  just 
                             names.
    :param points_in_time: a list of points in time for which you want to see
                           the density. At the moment up to 6 points in time
                           are supported.
    :param group_by: name of the column in the cases array to group results by.
                     Alternatively, `index` can be used to use indexing arrays 
                     as the basis for grouping.
    :param grouping_specifiers: set of categories to be used as a basis for 
                                grouping by. Grouping_specifiers is only 
                                meaningful if group_by is provided as well. In
                                case of grouping by index, the grouping 
                                specifiers should be in a dictonary where the
                                key denotes the name of the group. 
    :param density: field, either KDE or HIST 
    :param titles: a way for controlling whether each of the axes should have
                   a title. There are three possibilities. If set to None, no
                   title will be shown for any of the axes. If set to an empty 
                   dict, the default, the title is identical to the name of the
                   outcome of interest. If you want to override these default 
                   names, provide a dict with the outcome of interest as key 
                   and the desired title as value. This dict need only contain
                   the outcomes for which you want to use a different title. 
    :param ylabels: a way for controlling the ylablels. Works identical to 
                    titles.
    :param legend: boolean, if true, and there is a column specified for 
                   grouping, show a legend.
    :param experiments_to_show: numpy array containing the indices of the 
                                experiments to be visualized. Defaults to None,
                                implying that all experiments should be shown.
    :plot_type: kwarg for controling the type of main plot. Can be one of 
                ENVELOPE, LINES, or ENV_LIN
    :rtype: a `figure <http://matplotlib.sourceforge.net/api/figure_api.html>`_ instance
            
    Additional key word arguments will be passed along to the density function,
    
    ======== ===================================
    property description
    ======== ===================================
    log      log the resulting histogram or GKDE
    ======== ===================================
    
    .. rubric:: an example of use
    
    .. note:: the current implementation is limited to seven different 
              categories in case of column, categories, and/or discretesize.
              This limit is due to the colors specified in COLOR_LIST.
              
    .. note:: the connection patches are for some reason not drawn if log
              scaling is used for the density plots. This appears to be an
              issue in matplotlib itself.
              
              
    '''
    if not outcomes_to_show:
        outcomes_to_show =  results[1].keys()
        outcomes_to_show.remove(TIME)
    elif type(outcomes_to_show)==StringType:
        outcomes_to_show=[outcomes_to_show]
    
    axes_dicts = {}
    figures = []
    for outcome_to_show in outcomes_to_show:
        temp_results = copy.deepcopy(results)
        axes_dict = {}
        axes_dicts[outcome_to_show] = axes_dict
     
        if plot_type != ENV_LIN:
            # standard way of pre processing data
            if experiments_to_show != None:
                experiments, outcomes = temp_results
                experiments = experiments[experiments_to_show]
                new_outcomes = {}
                for key, value in outcomes.items():
                    new_outcomes[key] = value[experiments_to_show]
                temp_results = experiments, new_outcomes
    
        data = prepare_data(temp_results, [outcome_to_show], group_by, 
                            grouping_specifiers)
        outcomes, outcomes_to_show, time, grouping_labels = data
        del outcomes_to_show
    
        #start of plotting
        fig = plt.figure()
        figures.append(fig)
        
        # making of grid
        if not points_in_time:
            raise EMAError("no points in time specified")
        if len(points_in_time) == 1:
            ax_env = plt.subplot2grid((2,3), (0,0), colspan=3)
            ax1 = plt.subplot2grid((2,3), (1,1), )
            kde_axes = [ax1]
        elif len(points_in_time) == 2:
            ax_env = plt.subplot2grid((2,2), (0,0), colspan=2)
            ax1 = plt.subplot2grid((2,2), (1,0), )
            ax2 = plt.subplot2grid((2,2), (1,1), sharex=ax1)
            kde_axes = [ax1, ax2]
        elif len(points_in_time) == 3:
            ax_env = plt.subplot2grid((2,3), (0,0), colspan=3)
            ax1 = plt.subplot2grid((2,3), (1,0), )
            ax2 = plt.subplot2grid((2,3), (1,1), sharex=ax1)
            ax3 = plt.subplot2grid((2,3), (1,2), sharex=ax1)
            kde_axes = [ax1, ax2, ax3]
        elif len(points_in_time) == 4:
            ax_env = plt.subplot2grid((2,4), (0,1), colspan=2)
            ax1 = plt.subplot2grid((2,4), (1,0), )
            ax2 = plt.subplot2grid((2,4), (1,1), sharex=ax1)
            ax3 = plt.subplot2grid((2,4), (1,2), sharex=ax1)
            ax4 = plt.subplot2grid((2,4), (1,3), sharex=ax1)
            kde_axes = [ax1, ax2, ax3, ax4]
        elif len(points_in_time) == 5:
            ax_env = plt.subplot2grid((2,5), (0,1), colspan=3)
            ax1 = plt.subplot2grid((2,5), (1,0), )
            ax2 = plt.subplot2grid((2,5), (1,1), sharex=ax1)
            ax3 = plt.subplot2grid((2,5), (1,2), sharex=ax1)
            ax4 = plt.subplot2grid((2,5), (1,3), sharex=ax1)
            ax5 = plt.subplot2grid((2,5), (1,4), sharex=ax1)
            kde_axes = [ax1, ax2, ax3, ax4, ax5]
        elif len(points_in_time) == 6:
            ax_env = plt.subplot2grid((2,6), (0,1), colspan=4)
            ax1 = plt.subplot2grid((2,6), (1,0), )
            ax2 = plt.subplot2grid((2,6), (1,1), sharex=ax1)
            ax3 = plt.subplot2grid((2,6), (1,2), sharex=ax1)
            ax4 = plt.subplot2grid((2,6), (1,3), sharex=ax1)
            ax5 = plt.subplot2grid((2,6), (1,4), sharex=ax1)
            ax6 = plt.subplot2grid((2,6), (1,5), sharex=ax1)
            
            kde_axes = [ax1, ax2, ax3, ax4, ax5, ax6, ]
        else:
            raise EMAError("too many points in time provided")

        axes_dict["main plot"] = ax_env
        for n, entry in enumerate(kde_axes):
            axes_dict["density_%s" % n] = entry

            #turn of ticks for all but the first density            
            if n > 0:
                for tl in entry.get_yticklabels():
                    tl.set_visible(False)
                    
        
        # bit of a trick to avoid duplicating code. If no subgroups are 
        # specified, nest the outcomes one step deeper in de dict so the
        # iteration below can proceed normally.
        if not grouping_labels:
            grouping_labels=[""]
            outcomes[""]=outcomes
            
        max_x = 0
        for j, key in enumerate(grouping_labels):
            value = outcomes[key][outcome_to_show]
            
            if plot_type == ENVELOPE:
                plot_envelope(ax_env, j, time, value, fill=False)
            elif plot_type == LINES:
                ax_env.plot(time.T, value.T)
            elif plot_type == ENV_LIN:
                plot_envelope(ax_env, j, time, value, fill=True)
                if experiments_to_show!=None:
                    ax_env.plot(time.T, value[experiments_to_show].T)
                else:
                    ax_env.plot(time.T, value.T)
            ax_env.set_xlim(time[0], time[-1])
            
            ax_env.set_xlabel(TIME_LABEL)
            do_ylabels(ax_env, ylabels, outcome_to_show)
            do_titles(ax_env, titles, outcome_to_show)
            
            # this might seem a bit strange but under some conditions can the 
            # autoscaling of the y_axis be # changed due to the plot command 
            # for the crossection line. This overrides the autoscaling 
            # updating.  
            min_y, max_y = ax_env.get_ylim()
            ax_env.autoscale(enable=False, axis='y')
            
            for i, ax in enumerate(kde_axes):
                time_value = points_in_time[i]
                
                if time_value:
                    index = np.where(time==time_value)[0][0]
                    if density==KDE:
                        kde_x, kde_y = determine_kde(value[:,index])
                        plot_kde(ax, kde_x, kde_y, j,**kwargs)      
                        
                        #update max_x
                        max_kde =np.max(kde_x)
                        if  max_kde > max_x and max_kde < 10:
                            max_x = max_kde
                    
                    ax_env.plot([points_in_time[i],points_in_time[i]], 
                                [min_y,max_y], c='k', ls='--')
                    con = ConnectionPatch(xyA=(time_value, 0), 
                                          xyB=(min_y,max_y), coordsA="data", 
                                          coordsB="data", axesA=ax_env, 
                                          axesB=ax)
                    ax_env.add_artist(con)
                            
        if density == HIST:
            for i, ax in enumerate(kde_axes):
                time_value = points_in_time[i]
                index = np.where(time==points_in_time[i])[0][0]
                
                
                values = [outcomes[key][outcome_to_show][:,index] for key in\
                          grouping_labels]
                n, bins, patches = plot_histogram(ax, values, **kwargs)
                del bins, patches
                
                if np.max(n) > max_x and np.max(n)<10:
                    max_x =  np.max(n)        
    
        for ax in kde_axes:
            ax.get_yaxis().set_view_interval(
                         ax_env.get_yaxis().get_view_interval()[0],
                         ax_env.get_yaxis().get_view_interval()[1]) 
#            ax.set_xlim(xmin=0,xmax=math.ceil(max_x))
            ax.set_xlim(xmin=0,xmax=max_x)
        
        if legend and group_by:
            make_legend(grouping_labels, fig)
    return figures, axes_dicts