'''

This module provides R style pairs plotting functionality.

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from . import plotting_util
from .plotting_util import SCATTER, LINE
from .plotting_util import prepare_pairs_data, make_legend
from ..util import debug, info

# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['pairs_scatter', 'pairs_lines', 'pairs_density']


def pairs_lines(results, 
                outcomes_to_show = [],
                group_by = None,
                grouping_specifiers = None,
                ylabels = {},
                legend=True,
                **kwargs):
    '''
    
    Generate a `R style pairs <http://www.stat.psu.edu/~dhunter/R/html/graphics/html/pairs.html>`_ 
    lines multiplot. It shows the behavior of two outcomes over time against
    each other. The origin is denoted with a circle and the end is denoted
    with a '+'. 
    
    Parameters
    ----------
    results : tuple
              return from perform_experiments.
    outcomes_to_show : list of str, optional
                       list of outcome of interest you want to plot.
    group_by : str, optional
               name of the column in the cases array to group results by. 
               Alternatively, `index` can be used to use indexing arrays as the 
               basis for grouping.
    grouping_specifiers : dict, optional
                          dict of categories to be used as a basis for grouping 
                          by. Grouping_specifiers is only meaningful if 
                          group_by is provided as well. In case of grouping by
                          index, the grouping  specifiers should be in a 
                          dictionary where the key denotes the name of the 
                          group. 
    ylabels : dict, optional
              ylabels is a dictionary with the outcome names as keys, the 
              specified values will be used as labels for the y axis. 
    legend : bool, optional
             if true, and group_by is given, show a legend.
    point_in_time : float, optional
                    the point in time at which the scatter is to be made. If 
                    None is provided (default), the end states are used. 
                    point_in_time should be a valid value on time
    
    Returns
    -------
    fig
        the figure instance
    dict
        key is tuple of names of outcomes, value is associated axes
        instance
    
    '''
    
    #unravel return from run_experiments   
    debug("making a pars lines plot")
    
    prepared_data = prepare_pairs_data(results, outcomes_to_show, group_by, 
                                       grouping_specifiers, None)
    outcomes, outcomes_to_show, grouping_labels = prepared_data
    
    grid = gridspec.GridSpec(len(outcomes_to_show), len(outcomes_to_show))                             
    grid.update(wspace = 0.1,
                hspace = 0.1)
    
    #the plotting
    figure = plt.figure()
    axes_dict = {}
  
    combis = [(field1, field2) for field1 in outcomes_to_show\
                               for field2 in outcomes_to_show]
    
    for field1, field2 in combis:
        i = list(outcomes_to_show).index(field1)
        j = list(outcomes_to_show).index(field2)
        ax = figure.add_subplot(grid[i,j])
        
        axes_dict[(field1, field2)] = ax

        if group_by:
            for x, entry in enumerate(grouping_labels):
                data1 = outcomes[entry][field1]
                data2 = outcomes[entry][field2]
                color = plotting_util.COLOR_LIST[x]
                if i==j: 
                    color = 'white'
                simple_pairs_lines(ax, data1, data2, color)
        else:
            data1 = outcomes[field1]
            data2 = outcomes[field2]
            color = 'b'
            if i==j: 
                color = 'white'
            simple_pairs_lines(ax, data1, data2, color)
        do_text_ticks_labels(ax, i, j, field1, field2, ylabels, 
                             outcomes_to_show)

    if group_by and legend:
        gs1 = grid[0,0]
        
        for ax in figure.axes:
            gs2 = ax._subplotspec
            if all((gs1._gridspec == gs2._gridspec,
                    gs1.num1 == gs2.num1,
                    gs1.num2 == gs2.num2)):
                break  
        
        make_legend(grouping_labels, ax, legend_type=LINE)

    return figure, axes_dict
 
 
def simple_pairs_lines(ax, y_data, x_data, color):    
    '''
    
    Helper function for generating a simple pairs lines plot

    Parameters
    ----------
    ax : axes
    data1 : ndarray
    data2 : ndarray
    color : str
    
    '''
             
    ax.plot(x_data.T, y_data.T, c=color)
    ax.scatter(x_data[:, 0], y_data[:, 0],
               edgecolor=color, facecolor=color,
               marker='o')
    ax.scatter(x_data[:, -1], y_data[:, -1],
               edgecolor=color, facecolor=color,
               marker='+')     


def pairs_density(results, 
                  outcomes_to_show = [],
                  group_by = None,
                  grouping_specifiers = None,
                  ylabels = {},
                  point_in_time=-1,
                  log=True,
                  gridsize=50,
                  colormap='coolwarm',
                  filter_scalar=True): 
    '''
    
    Generate a `R style pairs <http://www.stat.psu.edu/~dhunter/R/html/graphics/html/pairs.html>`_ 
    hexbin density multiplot. In case of time-series data, the end states are 
    used.
    
    hexbin makes hexagonal binning plot of x versus y, where x, y are 1-D 
    sequences of the same length, N. If C is None (the default), this is a 
    histogram of the number of occurences of the observations at (x[i],y[i]).
    For further detail see `matplotlib on hexbin <http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.hexbin>`_

    Parameters
    ----------
    results : tuple
              return from perform_experiments.
    outcomes_to_show : list of str, optional
                       list of outcome of interest you want to plot.
    group_by : str, optional
               name of the column in the cases array to group results by. 
               Alternatively, `index` can be used to use indexing arrays as the 
               basis for grouping.
    grouping_specifiers : dict, optional
                          dict of categories to be used as a basis for grouping 
                          by. Grouping_specifiers is only meaningful if 
                          group_by is provided as well. In case of grouping by
                          index, the grouping  specifiers should be in a 
                          dictionary where the key denotes the name of the 
                          group. 
    ylabels : dict, optional
              ylabels is a dictionary with the outcome names as keys, the 
              specified values will be used as labels for the y axis. 
    point_in_time : float, optional
                    the point in time at which the scatter is to be made. If 
                    None is provided (default), the end states are used. 
                    point_in_time should be a valid value on time
    log: bool, optional
        indicating whether density should be log scaled. Defaults to True.
    gridsize : int, optional
               controls the gridsize for the hexagonal bining. (default = 50)
    cmap : str
           color map that is to be used in generating the hexbin. For details 
           on the available maps, see `pylab <http://matplotlib.sourceforge.net/examples/pylab_examples/show_colormaps.html#pylab-examples-show-colormaps>`_.
           (Defaults = coolwarm)
    filter_scalar: bool, optional 
                   remove the non-time-series outcomes. Defaults to True.
    
    Returns
    -------
    fig
        the figure instance
    dict
        key is tuple of names of outcomes, value is associated axes
        instance
    
    '''
    debug("generating pairwise density plot")
    
    prepared_data = prepare_pairs_data(results, outcomes_to_show, group_by, 
                                       grouping_specifiers, point_in_time,
                                       filter_scalar)
    outcomes, outcomes_to_show, grouping_specifiers = prepared_data
   
    if group_by:
        #figure out the extents for each combination
        extents = determine_extents(outcomes, outcomes_to_show)
        
        axes_dicts = {}
        figures = []
        for key, value in outcomes.items():
            figure, axes_dict = simple_pairs_density(value, outcomes_to_show, 
                                       log, colormap, gridsize, ylabels,
                                       extents=extents, title=key)
            axes_dicts[key] = axes_dict
            figures.append(figure)
        
        # harmonize the color scaling across figures
        combis = [(field1, field2) for field1 in outcomes_to_show\
                           for field2 in outcomes_to_show]
        for combi in combis:
            vmax = -1
            for entry in axes_dicts.values():
                vmax =  max(entry[combi].collections[0].norm.vmax, vmax)
            for entry in axes_dicts.values():
                ax = entry[combi]
                ax.collections[0].set_clim(vmin=0, vmax=vmax)
            del vmax
            
        return figures, axes_dicts
    else:
        return simple_pairs_density(outcomes, outcomes_to_show, log, colormap, 
                                    gridsize, ylabels)


def determine_extents(outcomes, outcomes_to_show):
    '''
    Helper function used by pairs_density to make sure that multiple groups
    share the same axes extent.

    Parameters
    ----------
    outcomes : dict
    outcomes_to_show : list of str
    
    Returns
    -------
    dict
        tuple of str as key, and 4-tuple with extent
    
    '''
    
    limits = {}
    for pol_out in outcomes.values():
        for entry in outcomes_to_show:
            out = pol_out[entry]
            minimum = np.amin(out)
            maximum = np.amax(out)
            try:
                cur = limits[entry]
                new = (min(cur[0], minimum),
                     max(cur[1], maximum)) 
                limits[entry] = new
            except KeyError:
                limits[entry] = (minimum, maximum)
    extents = {}
    combis = [(field1, field2) for field1 in outcomes_to_show\
                               for field2 in outcomes_to_show]
    for field1, field2 in combis:
        limits_1 = limits[field1]
        limits_2 = limits[field2]  
        extents[(field1, field2)] = (limits_1[0], limits_1[1],
                                     limits_2[0], limits_2[1])
    return extents


def simple_pairs_density(outcomes,
                         outcomes_to_show,
                         log,
                         colormap,
                         gridsize,
                         ylabels,
                         extents=None,
                         title=None):
    '''
    
    Helper function for generating a simple pairs density plot

    Parameters
    ----------
    outcomes : dict
    outcomes_to_show : list of str
    log : bool
    colormap : str
    gridsize : int
    ylabels: dict
    extents : dict, optional
             used to control scaling of plots across figures, If provided, it 
             should be a dict with a tuple of outcomes as key and the extend to 
             be used as value.
    title : str, optional
                    
    
    '''
    
    grid = gridspec.GridSpec(len(outcomes_to_show), len(outcomes_to_show))
    grid.update(wspace = 0.1,
                hspace = 0.1)
     
    #the plotting
    figure = plt.figure()
     
    combis = [(field1, field2) for field1 in outcomes_to_show\
                               for field2 in outcomes_to_show]
    axes_dict = {}
    for field1, field2 in combis:
        i = list(outcomes_to_show).index(field1)
        j = list(outcomes_to_show).index(field2)
        
        ax = figure.add_subplot(grid[i,j])
        axes_dict[(field1, field2)] = ax

        y_data = outcomes[field1]
        x_data = outcomes[field2]

        bins=None
        if log:
            bins='log'
        
        extent = None
        if extents:
            extent = extents[(field2, field1)]
       
        #text and labels
        if i == j:
            #only plot the name in the middle
            ax.hexbin(x_data,y_data, bins=bins, gridsize=gridsize, 
                      cmap=cm.__dict__[colormap], alpha=0, edgecolor='white', 
                      linewidths=1, extent=extent)
        else:
            ax.hexbin(x_data,y_data, bins=bins, gridsize=gridsize, 
                      cmap=cm.__dict__[colormap], edgecolor='black', 
                      linewidths=1, extent=extent, mincnt=1)
        do_text_ticks_labels(ax, i, j, field1, field2, ylabels, 
                             outcomes_to_show)

    return figure, axes_dict

  
def pairs_scatter(results, 
                  outcomes_to_show = [],
                  group_by = None,
                  grouping_specifiers = None,
                  ylabels = {},
                  legend=True,
                  point_in_time=-1,
                  filter_scalar=False,
                  **kwargs):
    '''
    
    Generate a `R style pairs <http://www.stat.psu.edu/~dhunter/R/html/graphics/html/pairs.html>`_ 
    scatter multiplot. In case of time-series data, the end states are used.
    
    Parameters
    ----------
    results : tuple
              return from perform_experiments.
    outcomes_to_show : list of str, optional
                       list of outcome of interest you want to plot.
    group_by : str, optional
               name of the column in the cases array to group results by. 
               Alternatively, `index` can be used to use indexing arrays as the 
               basis for grouping.
    grouping_specifiers : dict, optional
                          dict of categories to be used as a basis for grouping 
                          by. Grouping_specifiers is only meaningful if 
                          group_by is provided as well. In case of grouping by
                          index, the grouping  specifiers should be in a 
                          dictionary where the key denotes the name of the 
                          group. 
    ylabels : dict, optional
              ylabels is a dictionary with the outcome names as keys, the 
              specified values will be used as labels for the y axis. 
    legend : bool, optional
             if true, and group_by is given, show a legend.
    point_in_time : float, optional
                    the point in time at which the scatter is to be made. If 
                    None is provided (default), the end states are used. 
                    point_in_time should be a valid value on time
    filter_scalar: bool, optional 
                   remove the non-time-series outcomes. Defaults to True.

    Returns
    -------
    fig : Figure instance
          the figure instance
    axes  : dict
            key is tuple of names of outcomes, value is associated axes
            instance

    .. note:: the current implementation is limited to seven different 
              categories in case of column, categories, and/or discretesize.
              This limit is due to the colors specified in COLOR_LIST. 
    
    '''
    
    debug("generating pairwise scatter plot")
   
    prepared_data = prepare_pairs_data(results, outcomes_to_show, group_by, 
                                       grouping_specifiers, point_in_time,
                                       filter_scalar)
    outcomes, outcomes_to_show, grouping_labels = prepared_data
   
    grid = gridspec.GridSpec(len(outcomes_to_show), len(outcomes_to_show))                             
    grid.update(wspace = 0.1,
                hspace = 0.1)    
    
    
    #the plotting
    figure = plt.figure()
    axes_dict = {}
    
    combis = [(field1, field2) for field1 in outcomes_to_show\
                               for field2 in outcomes_to_show]
    
    for field1, field2 in combis:
        i = list(outcomes_to_show).index(field1)
        j = list(outcomes_to_show).index(field2)
        ax = figure.add_subplot(grid[i,j])
        axes_dict[(field1, field2)] = ax

        if group_by:
            for x, group in enumerate(grouping_labels):
                y_data = outcomes[group][field1]
                x_data = outcomes[group][field2]
                
                facecolor = plotting_util.COLOR_LIST[x]
                edgecolor = 'k'
                if i==j: 
                    facecolor = 'white'
                    edgecolor = 'white'
                ax.scatter(x_data, y_data, 
                           facecolor=facecolor, edgecolor=edgecolor)
        else:
            y_data = outcomes[field1]
            x_data = outcomes[field2]

            facecolor = 'b'
            edgecolor = 'k'
            if i==j: 
                facecolor = 'white'
                edgecolor = 'white'
            ax.scatter(x_data, y_data, 
                       facecolor=facecolor, edgecolor=edgecolor)
        do_text_ticks_labels(ax, i, j, field1, field2, ylabels, 
                             outcomes_to_show)

    if group_by and legend:
        gs1 = grid[0,0]
        
        for ax in figure.axes:
            gs2 = ax._subplotspec
            if all((gs1._gridspec == gs2._gridspec,
                    gs1.num1 == gs2.num1,
                    gs1.num2 == gs2.num2)):
                break  
        
        make_legend(grouping_labels, ax, legend_type=SCATTER)

    return figure, axes_dict


def do_text_ticks_labels(ax, i, j, field1, field2, ylabels, outcomes_to_show):
    '''
    
    Helper function for setting the tick labels on the axes correctly on and 
    off

    Parameters
    ----------
    ax : axes
    i : int
    j : int
    field1 : str
    field2 : str
    ylabels : dict, optional
    outcomes_to_show : str
    
    
    '''
    
    #text and labels
    if i == j:
        #only plot the name in the middle
        if ylabels:
            text = ylabels[field1]
        else:
            text = field1
        ax.text(0.5, 0.5, text,
                horizontalalignment='center',
                verticalalignment='center',
                transform = ax.transAxes)  
    
    # are we at the end of the row?
    if i != len(outcomes_to_show)-1:
        #xaxis off
        ax.set_xticklabels([])
    else:
        if ylabels:
            try:
                ax.set_xlabel(ylabels.get(field2))
            except KeyError:
                info("no label specified for "+field2)
        else:
            ax.set_xlabel(field2) 
    
    # are we at the end of the column?
    if j != 0:
        #yaxis off
        ax.set_yticklabels([])
    else:
        if ylabels:
            try:
                ax.set_ylabel(ylabels.get(field1))
            except KeyError:
                info("no label specified for "+field1) 
        else:
            ax.set_ylabel(field1)   
