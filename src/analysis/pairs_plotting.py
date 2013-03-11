'''
.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

This module provides R style pairs plotting functionality.

'''
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import matplotlib.cm as cm

from expWorkbench.ema_logging import debug, info

from plotting_util import prepare_pairs_data, make_legend, COLOR_LIST

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
    
    :param results: return from perform_experiments.
    :param outcomes_to_show: list of outcome of interest you want to plot. If 
                             empty, all outcomes are plotted.
    :param group_by: name of the column in the cases array to group results by.
                     Alternatively, `index` can be used to use indexing arrays 
                     as the basis for grouping.
    :param grouping_specifiers: set of categories to be used as a basis for 
                                grouping by. Grouping_specifiers is only 
                                meaningful if group_by is provided as well. In
                                case of grouping by index, the grouping 
                                specifiers should be in a dictionary where the
                                key denotes the name of the group. 
    :param ylabels: ylabels is a dictionary with the outcome names as keys, the 
                    specified values will be used as labels for the y axis. 
    :param legend: boolean, if true, and there is a column specified for 
                   grouping, show a legend.
    :param point_in_time: the point in time at which the scatter is to be made.
                          If None is provided, the end states are used. 
                          point_in_time should be a valid value on time
    :rtype: a `figure <http://matplotlib.sourceforge.net/api/figure_api.html>`_ instance
            and a dict with the individual axes.
    
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
    
    if group_by and legend:
        make_legend(grouping_labels, figure)
     
    combis = [(field1, field2) for field1 in outcomes_to_show\
                               for field2 in outcomes_to_show]
    
    for field1, field2 in combis:
        i = outcomes_to_show.index(field1)
        j = outcomes_to_show.index(field2)
        ax = figure.add_subplot(grid[i,j])
        axes_dict[(field1, field2)] = ax

        if group_by:
            for x, entry in enumerate(grouping_labels):
                data1 = outcomes[entry][field1]
                data2 = outcomes[entry][field2]
                color = COLOR_LIST[x]
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
            

    return figure, axes_dict
 
def simple_pairs_lines(ax, y_data, x_data, color):    
    '''
    
    Helper function for generating a simple pairs lines plot
    
    :param ax:
    :param data1:
    :param data2:
    :param color:
    
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
                  colormap='jet',
                  filter_scalar=True): 
    '''
    
    Generate a `R style pairs <http://www.stat.psu.edu/~dhunter/R/html/graphics/html/pairs.html>`_ 
    hexbin density multiplot. In case of time-series data, the end states are 
    used.
    
    hexbin makes hexagonal binning plot of x versus y, where x, y are 1-D 
    sequences of the same length, N. If C is None (the default), this is a 
    histogram of the number of occurences of the observations at (x[i],y[i]).
    For further detail see `matplotlib on hexbin <http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.hexbin>`_
    
    :param results: return from perform_experiments.
    :param outcomes_to_show: list of outcome of interest you want to plot. If 
                             empty, all outcomes are plotted.
    :param group_by: name of the column in the cases array to group results by.
                     Alternatively, `index` can be used to use indexing arrays 
                     as the basis for grouping.
    :param grouping_specifiers: set of categories to be used as a basis for 
                                grouping by. Grouping_specifiers is only 
                                meaningful if group_by is provided as well. In
                                case of grouping by index, the grouping 
                                specifiers should be in a dictionary where the
                                key denotes the name of the group. 
    :param ylabels: ylabels is a dictionary with the outcome names as keys, the 
                    specified values will be used as labels for the y axis. 
    :param point_in_time: the point in time at which the scatter is to be made.
                          If None is provided, the end states are used. 
                          point_in_time should be a valid value on time
    :param log: boolean, indicating whether density should be log scaled. 
                Defaults to True.
    :param gridsize: controls the gridsize for the hexagonal binning
    :param cmap: color map that is to be used in generating the hexbin. For 
                 details on the available maps, 
                 see `pylab <http://matplotlib.sourceforge.net/examples/pylab_examples/show_colormaps.html#pylab-examples-show-colormaps>`_.
                 (Defaults = jet)
    :param filter_scalar: boolean, remove the non-time-series outcomes.  
                          Defaults to True.
    :rtype: a `figure <http://matplotlib.sourceforge.net/api/figure_api.html>`_ instance
            and a dict with the individual axes.
    
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
    
    :param outcomes:
    :param outcomes_to_show:
    
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
    
    :param outcomes:
    :param outcomes_to_show:
    :param log:
    :param colormap:
    :param gridsize:
    :param ylabels:
    :param extents: used to control scaling of plots across figures, 
                    If provided, it should be a dict with a tuple of outcomes
                    as key and the extend to be used as value.
                    
    
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
        i = outcomes_to_show.index(field1)
        j = outcomes_to_show.index(field2)
        
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
                  filter_scalar=True,
                  **kwargs):
    '''
    
    Generate a `R style pairs <http://www.stat.psu.edu/~dhunter/R/html/graphics/html/pairs.html>`_ 
    scatter multiplot. In case of time-series data, the end states are used.
    
    :param results: return from perform_experiments.
    :param outcomes_to_show: list of outcome of interest you want to plot. If 
                             empty, all outcomes are plotted.
    :param group_by: name of the column in the cases array to group results by.
                     Alternatively, `index` can be used to use indexing arrays 
                     as the basis for grouping.
    :param grouping_specifiers: set of categories to be used as a basis for 
                                grouping by. Grouping_specifiers is only 
                                meaningful if group_by is provided as well. In
                                case of grouping by index, the grouping 
                                specifiers should be in a dictionary where the
                                key denotes the name of the group. 
    :param ylabels: ylabels is a dictionary with the outcome names as keys, the 
                    specified values will be used as labels for the y axis. 
    :param legend: boolean, if true, and there is a column specified for 
                   grouping, show a legend.
    :param point_in_time: the point in time at which the scatter is to be made.
                          If None is provided, the end states are used. 
                          point_in_time should be a valid value on time
    :param filter_scalar: boolean, remove the non-time-series outcomes.  
                          Defaults to True.
    :rtype: a `figure <http://matplotlib.sourceforge.net/api/figure_api.html>`_ instance
            and a dict with the individual axes.
    

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
    
    if group_by and legend:
        make_legend(grouping_labels, figure, legend_type='scatter')
     
    combis = [(field1, field2) for field1 in outcomes_to_show\
                               for field2 in outcomes_to_show]
    
    for field1, field2 in combis:
        i = outcomes_to_show.index(field1)
        j = outcomes_to_show.index(field2)
        ax = figure.add_subplot(grid[i,j])
        axes_dict[(field1, field2)] = ax

        if group_by:
            for x, group in enumerate(grouping_labels):
                y_data = outcomes[group][field1]
                x_data = outcomes[group][field2]
                
                facecolor = COLOR_LIST[x]
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

    return figure, axes_dict

def do_text_ticks_labels(ax, i, j, field1, field2, ylabels, outcomes_to_show):
    '''
    
    Helper function for setting the tick labels on the axes correctly on and of
    
    :param ax:
    :param i:
    :param j:
    :param field1:
    :param field2:
    :param ylabels:
    :param outcomes_to_show:
    
    
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
