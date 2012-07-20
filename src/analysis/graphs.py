"""



this module provides functions for generating some basic figures. The code can
be used as is, or serve as an example for writing your own code. These plots
rely on `matplotlib <http://matplotlib.sourceforge.net/>`_, 
`numpy <http://numpy.scipy.org/>`_, and `scipy.stats.kde <http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html#scipy.stats.gaussian_kde>`_

"""
from __future__ import division
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
from matplotlib.cm import jet #@UnresolvedImport

import scipy.stats.kde as kde 
import numpy as np
from string import lower

from expWorkbench.EMAlogging import debug, info

__all__ = ['lines', 'envelopes', 'multiplot_scatter', 
           'multiplot_lines', 'multiplot_density']


tight = False

#see http://matplotlib.sourceforge.net/users/customizing.html for details
mpl.rcParams['axes.formatter.limits'] = (-5, 5)
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'

TIME = "TIME"

COLOR_LIST = ['b',
              'g',
              'r',
              'm',
              'c',
              'y',
              'k',
              'Aqua',
              'DarkBlue',
              'GreenYellow',
              'OrangeRed']

#==============================================================================
# helper functions
#==============================================================================
def __make_legend(categories, figure):
    some_lines = []
    labels = []
    for i, category in enumerate(categories):
        line = plt.Line2D([0,1], [0,1], color=COLOR_LIST[i])
        some_lines.append(line)
        
        if type(category) == tuple:
            label =  '%.2f - %.2f' % category 
        else:
            label = category
        
        labels.append(str(label))

    ncol = int(np.ceil(len(categories)/2))
    
    figure.legend(some_lines, labels, ncol=ncol,
                      loc='upper center', borderaxespad=0.1)

def __density(j, axDensity, ax, data, hist=False, log=False):
    '''
    
    Helper function responsible for plotting the density part.    
    
    :param j: index of set of runs being plotted. 
    :param axDensity: the axes for plotting the density on.
    :param ax: the axes in which the actual data is being plotted.
    :param data: the data to be used for plotting the KDE or the histogram.
    :param hist: boolean, if true, use histograms, otherwise use gaussian Kernel Densitiy Estimate.
    :param log: boolean, if true log the results from histogram or kde
    
    .. note: all parameters apart from hist are provided by lines or envelopes.
    
    '''
    
    if hist:
        axDensity.hist(data, 
                     bins=21, 
                     orientation='horizontal',
                     histtype='bar', 
                     normed = True,
                     log=log)
        for tl in axDensity.get_yticklabels():
            tl.set_visible(False)
        axDensity.set_xticks([0, axDensity.get_xbound()[1]])
    else:
        #make a KDE instead of a histogram 

        axDensity.get_yaxis().set_view_interval(
                                 ax.get_yaxis().get_view_interval()[0],
                                 ax.get_yaxis().get_view_interval()[1])
        ymin = np.min(data)
        ymax = np.max(data)
        
        line = np.linspace(ymin, ymax, 1000)[::-1]
        b = kde.gaussian_kde(data)
        b = b.evaluate(line)
        
        if log:
            b = np.log((b+1))
        
        axDensity.plot(b, line, color=COLOR_LIST[j])
#            axHisty.yaxis.set_major_locator(NullLocator())
        
        for tl in axDensity.get_yticklabels():
            tl.set_visible(False)
#            axHisty.set_yticks([])
        axDensity.set_xticks([int(0), 
                            axDensity.get_xaxis().
                            get_view_interval()[1]])
        

def __discretesize(array):
    '''
    Helper function for discretesizing a continous array. By default, the array 
    is split into 5 equally wide intervals.
    
    :param array: a 1-d array that is to be turned into discrete intervals.
    :return: list of tuples with the lower and upper bound of the intervals.
    
    '''
    
    minima = np.min(array)
    maxima = np.max(array)
    step = (maxima-minima)/5
    a = [(minima+step*x, minima+step*(x+1)) for x in range(5)]
    
    # we add a tiny bit to the upside of the interval to make sure that all
    # the cases are shown
    a[-1] = (a[-1][0], a[-1][1]+0.00000001)

    return a

#==============================================================================
# actual plotting functions
#==============================================================================
def envelopes(results, 
              outcomes = [],
              column = None,
              categories = None,
              ylabels = {},
              fill=False,
              density=True,
              legend=True,
              discretesize=__discretesize,
              **kwargs):
    '''
    
    Make envelop plots. An envelope shows over time the minimum and maximum 
    value for a set of runs over time. It is thus to be used in case of time 
    series data. The function will try to find a result labeled "TIME". If this
    is present, these values will be used on the X-axis. In case of Vensim 
    models, TIME is present by default.  
    
    :param results: return from :meth:`perform_experiments`.
    :param outcomes: list of outcome of interest you want to plot. If empty, 
                     all outcomes are plotted. **Note**:  just names.
    :param column: name of the column in the cases array to group results by.
    :param categories: set of categories to be used as a basis for grouping by. 
                       Categories is only meaningful if column is provided as 
                       well.
    :param ylabels: ylabels is a dictionary with the outcome names as keys, the 
                    specified values will be used as labels for the y axis. 
    :param fill: boolean, if true, fill the envelope. 
    :param density: boolean, if true, the density of the endstates will be 
                    plotted.
    :param legend: boolean, if true, and there is a column specified for 
                   grouping, show a legend.
    :param discretesize: function to be used to turn a continuous column into 
                         intervals in order to use for grouping by.
    :rtype: a `figure <http://matplotlib.sourceforge.net/api/figure_api.html>`_ instance
            
    Additional key word arguments will be passed along to the density function,
    if density is `True`.
    
    ======== ===================================
    property description
    ======== ===================================
    hist     use a histogram instead of a GKDE
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

    #unravel return from run_experiments   
    experiments, results = results
    
    #establish time axis
    keys = results.keys()
    try:
        time =  results.get('TIME')[0, :]
        keys.pop(keys.index('TIME'))
    except KeyError:
        time =  np.arange(0, results.values()[0].shape[1])
        
    #establish outcomes to plot
    if not outcomes:
        outcomes = keys
    FIELDS = outcomes

    #establish basis for clustering, if specified
    if not categories:
        if column:
            categories = experiments[column]
            if categories.dtype == np.float64 or categories.dtype == np.float32:
                categories = discretesize(categories)
            else:
                categories=set(categories)

    #specify the grid to be used for plotting
    if density:
        grid = gridspec.GridSpec(len(FIELDS), 2,
                                 width_ratios = [4, 1])
    else:
        grid = gridspec.GridSpec(len(FIELDS), 1)
    grid.update(wspace = 0.05,
                hspace = 0.4)

    #the plotting
    figure = plt.figure()
    
    #legend
    if (categories != None) and legend:
        __make_legend(categories, figure)
        
    
    for i, field in enumerate(FIELDS):
        associatedData = results.get(field)
               
        #make the axes
        ax = figure.add_subplot(grid[i, 0])
        if density:
            axDensity = figure.add_subplot(grid[i, 1], 
                                 sharey=ax,
                                 adjustable='box-forced')
        if ylabels:
            try:
                ax.set_ylabel(ylabels.get(field))
            except KeyError:
                info("no label specified for "+field)
        
        if categories:
            for j, category in enumerate(categories):
                if type(category)==tuple:
                    a = (experiments[column] >= category[0]) &\
                         (experiments[column]  < category[1])
                else:
                    a = experiments[column]==category
                
                value = associatedData[a, :]
           
                #plot minima and maxima
                minimum = np.min(value, axis=0)
                maximum = np.max(value, axis=0)
                
                ax.plot(time, minimum, c=COLOR_LIST[j])
                ax.plot(time, maximum, c=COLOR_LIST[j])
                
                if fill:
                    ax.fill_between(time, 
                                    minimum,
                                    maximum,
                                    facecolor=COLOR_LIST[j], 
                                    alpha=0.3)
                
                data = value[:, -1]
                if density:
                    __density(j, axDensity, ax, data, **kwargs)
        else:
            value = associatedData
       
            #plot minima and maxima
            minimum = np.min(value, axis=0)
            maximum = np.max(value, axis=0)
            
            ax.plot(time, minimum, c=COLOR_LIST[0])
            ax.plot(time, maximum, c=COLOR_LIST[0])
            
            if fill:
                ax.fill_between(time, 
                                minimum,
                                maximum,
                                facecolor=COLOR_LIST[0], 
                                alpha=0.3)
            
            data = value[:, -1]
            if density:
                __density(0, axDensity, ax, data, **kwargs)
    
        ax.set_title(lower(field))
#        ax.set_xlabel('Time (Years)')
    
    if tight:    
        grid.tight_layout(figure)
    return figure

def lines(results, 
          outcomes = [],
          column = None,
          categories = None,
          ylabels = {},
          density=False,
          legend=True,
          discretesize=__discretesize,
          **kwargs):
    '''
    
    This function takes the results from :meth:`perform_experiments` and 
    visualizes these as line plots. It is thus to be used in case of time 
    series data. The function will try to find a result labeled "TIME". If this
    is present, these values will be used on the X-axis. In case of Vensim 
    models, TIME is present by default.  
    
    :param results: return from :meth:`perform_experiments`.
    :param outcomes: list of outcome of interest you want to plot. If empty, 
                     all outcomes are plotted. **Note**:  just names.
    :param column: name of the column in the cases array to group results by.
    :param categories: set of categories to be used as a basis for grouping by. 
                       Categories is only meaningful if column is provided as 
                       well.
    :param ylabels: ylabels is a dictionary with the outcome names as keys, the 
                    specified values will be used as labels for the y axis. 
    :param density: boolean, if true, the density of the endstates will be 
                    plotted.
    :param legend: boolean, if true, and there is a column specified for 
                   grouping, show a legend.
    :param discretesize: function to be used to turn a continuous column into 
                         intervals in order to use for grouping by.
    :rtype: a `figure <http://matplotlib.sourceforge.net/api/figure_api.html>`_ instance

    
    Additional key word arguments will be passed along to the density function,
    if density is `True`.
    
    ======== ===================================
    property description
    ======== ===================================
    hist     use a histogram instead of a GKDE
    log      log the resulting histogram or GKDE
    ======== ===================================
   
    .. rubric:: an example of use
    
    >>> import expWorkbench.util as util
    >>> data = util.load_results(r'1000 flu cases.cPickle')
    >>> lines(data, density=True, hist=True)
    
    will show lines for all the outcomes of interest, and also shows 
    histograms for the endstate densities.
    
    .. plot:: ../docs/source/pyplots/basicLines.py
    
   
     while
    
    >>> lines(data, column='fatality ratio region 1', density=False)
    
    will group the result by the \'fatality ratio region 1\', this uncertainty
    is grouped into five intervals generated by the default discretesize 
    function.
    
    .. plot:: ../docs/source/pyplots/basicLines2.py
   
    
    the legend at the top shows the intervals used. Through the categorize 
    keyword argument, or through providing a different discretesize function, 
    alternative intervals can be specified.
   
   .. note:: the current implementation is limited to seven different 
          categories in case of column, categories, and/or discretesize.
          This limit is due to the colors specified in COLOR_LIST.
   
    '''
    
    debug("generating line graph")

    experiments, results = results
    
    #establish time axis
    keys = results.keys()
    try:
        time =  results['TIME']
        time = time[0, :]
        keys.pop(keys.index('TIME'))
    except KeyError:
        time =  np.arange(0, results.values()[0].shape[1])
        
    #establish outcomes to plot
    if not outcomes:
        outcomes = keys

    #establish basis for clustering, if specified
    if not categories:
        if column:
            categories = experiments[column]
            if categories.dtype == np.float64 or categories.dtype == np.float32:
                categories = discretesize(categories)
            else:
                categories=set(categories)

    #specify the grid to be used for plotting
    if density:
        grid = gridspec.GridSpec(len(outcomes), 2,
                                 width_ratios = [4, 1])
    else:
        grid = gridspec.GridSpec(len(outcomes), 1)
    grid.update(wspace = 0.05,
                hspace = 0.4)

    #the plotting
    figure = plt.figure()
    
    #legend
    if (categories != None) & legend:
        __make_legend(categories, figure)
   
    i = -1
    for i, field in enumerate(outcomes):
        value = results.get(field)
        debug("making graph for "+ field)
        
        ax = figure.add_subplot(grid[i, 0])
        if density:
            axDensity = figure.add_subplot(grid[i, 1], 
                                 sharey=ax,
                                 adjustable='box-forced')
        if ylabels:
            try:
                ax.set_ylabel(ylabels.get(field))
            except KeyError:
                info("no label specified for "+field)
        
        if categories:
            for index, category in enumerate(categories):
                if type(category)==tuple:
                    a = (experiments[column] >= category[0]) &\
                         (experiments[column]  < category[1])
                else:
                    a = experiments[column]==category
                
                y = value[a, :].T
                ax.plot(time.T[:, np.newaxis], y, COLOR_LIST[index])
                data = value[a, -1]
                
                if density:
                    __density(index, axDensity, ax, data, **kwargs)
        else:
            ax.plot(time.T, value.T)
            data = value[:, -1]
                
            if density:
                __density(0, axDensity, ax, data, **kwargs)
                
        ax.set_title(lower(field))
#        ax.set_xlabel('Time')
    
    if tight:
        grid.tight_layout(figure)
    return figure

def multiplot_lines(results, 
                    outcomes = [], 
                    column = None,
                    categories = None,
                    ylabels = {},
                    legend=False,
                    discretesize=__discretesize,):
    '''
    
    Generate a `R style pairs <http://www.stat.psu.edu/~dhunter/R/html/graphics/html/pairs.html>`_ 
    lines multiplot. It shows the behavior of two outcomes over time against
    each other. The origin is denoted with a cicle and the end is denoted
    with a '+'. 
    
    :param input: return from perform_experiments.
    :param outcomes: list of outcome of interest you want to plot. If empty, 
                     all outcomes are plotted note:  just names.
    :param column: name of the column in the cases array to group results by
    :param categories: set of categories to be used as a basis for grouping by. 
                       Categories is only meaningful if column is provided as 
                       well. 
    :param ylabels: ylabels is a dictionary with the outcome names as keys, the 
                    specified values will be used as labels for the y axis. 
    :param legend: boolean, if true, and there is a column specified for 
                   grouping, show a legend.
    :param discretesize: function to be used to turn a continuous column into 
                         intervals in order to use for grouping by.
    :rtype: a `figure <http://matplotlib.sourceforge.net/api/figure_api.html>`_ 
            instance.
        
    .. rubric:: an example of use
    
    .. plot:: ../docs/source/pyplots/basicMultiplotLines.py
    
    '''
    
    #unravel return from run_experiments   
    experiments, results = results
    
    #establish time axis
    try:
        results.pop('TIME')[0, :]
    except KeyError:
        pass
        
    #establish outcomes to plot
    if not outcomes:
        outcomes = results.keys()
    FIELDS = outcomes

    #establish basis for clustering, if specified
    if not categories:
        if column:
            categories = experiments[column]
            if categories.dtype == np.float64 or categories.dtype == np.float32:
                categories = discretesize(categories)
            else:
                categories=set(categories)
    
    grid = gridspec.GridSpec(len(FIELDS), len(FIELDS))                             
    
    #the plotting
    figure = plt.figure()
    
    if (categories != None) & legend:
        __make_legend(categories, figure)
     
    combis = [(field1, field2) for field1 in FIELDS for field2 in FIELDS]
    for field1, field2 in combis:
        i = FIELDS.index(field1)
        j = FIELDS.index(field2)
        ax = figure.add_subplot(grid[i,j])


        if categories:
            for x, category in enumerate(categories):
                if type(category)==tuple:
                    a = (experiments[column] >= category[0]) &\
                        (experiments[column]  < category[1])
                else:
                    a = experiments[column]==category

                data1 = results[field1][a]
                data2 = results[field2][a]
                color = COLOR_LIST[x]
                if i==j: 
                    color = 'white'
                ax.plot(data2.T, data1.T, c=color)
                ax.scatter(data2[:, 0], data1[:, 0],
                           edgecolor=color, facecolor=color,
                           marker='o')
                ax.scatter(data2[:, -1], data1[:, -1],
                           edgecolor=color, facecolor=color,
                           marker='+')
  
        else:
            data1 = results[field1]
            data2 = results[field2]
            color = 'b'
            if i==j: 
                color = 'white'
            ax.plot(data2.T, data1.T, c=color)
            ax.scatter(data2[:, 0], data1[:, 0],
                       edgecolor=color, facecolor=color,
                       marker='o')
            ax.scatter(data2[:, -1], data1[:, -1],
                       edgecolor=color, facecolor=color,
                       marker='+')
        
        #text and labels
        if i == j:
            #only plot the name in the middle
            ax.text(0.5, 0.5, field1,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform = ax.transAxes)  
        if i != len(FIELDS)-1:
            #xaxis off
            ax.set_xticklabels([])
        elif ylabels:
            try:
                ax.set_xlabel(ylabels.get(field1))
            except KeyError:
                info("no label specified for "+field1)        
        if j != 0:
            #yaxis off
            ax.set_yticklabels([])
        elif ylabels:
            try:
                ax.set_ylabel(ylabels.get(field2))
            except KeyError:
                info("no label specified for "+field2)
                   

def multiplot_density(results, 
                      outcomes = [],
                      ylabels={},
                      log=True,
                      gridsize=50,
                      cmap= jet): 
    '''
    
    Generate a `R style pairs <http://www.stat.psu.edu/~dhunter/R/html/graphics/html/pairs.html>`_ 
    hexbin density multiplot. In case of time-series data, the end states are 
    used.
    
    hexbin makes hexagonal binning plot of x versus y, where x, y are 1-D 
    sequences of the same length, N. If C is None (the default), this is a 
    histogram of the number of occurences of the observations at (x[i],y[i]).
    For further detail see `matplotlib on hexbin <http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.hexbin>`_
    
    :param input: return from perform_experiments.
    :param outcomes: list of outcome of interest you want to plot. If empty, 
                     all outcomes are plotted note:  just names.
    :param ylabels: ylabels is a dictionary with the outcome names as keys, the 
                    specified values will be used as labels for the y axis. 
    :param log: If True, the log of the value for each bin is taken prior
                to determining the color. (Default=True)
    :param gridsize: The number of hexagons in the x-direction. The 
                     corresponding number of hexagons in the y-direction is 
                     chosen such that the hexagons are approximately regular. 
                     Alternatively, gridsize can be a tuple with two elements 
                     specifying the number of hexagons in the x-direction and 
                     the y-direction. (Default is 50)
    :param cmap: color map that is to be used in generating the hexbin. For 
                 details on the available maps, 
                 see `pylab <http://matplotlib.sourceforge.net/examples/pylab_examples/show_colormaps.html#pylab-examples-show-colormaps>`_.
                 (Defaults = jet)
    :rtype: a `figure <http://matplotlib.sourceforge.net/api/figure_api.html>`_ 
            instance.
    
    .. rubric:: an example of use
    
    .. plot:: ../docs/source/pyplots/basicMultiplotDensity.py
    
    '''
    
    #unravel return from run_experiments   
    experiments, results = results
    
    #establish time axis
    try:
        results.pop('TIME')[0, :]
    except KeyError:
        pass
        
    #establish outcomes to plot
    if not outcomes:
        outcomes = results.keys()
    FIELDS = outcomes
    
    grid = gridspec.GridSpec(len(FIELDS), len(FIELDS))                             
    
    #the plotting
    figure = plt.figure()
     
    combis = [(field1, field2) for field1 in FIELDS for field2 in FIELDS]
    for field1, field2 in combis:
        i = FIELDS.index(field1)
        j = FIELDS.index(field2)
        ax = figure.add_subplot(grid[i,j])

        try:
            data1 = results[field1][:, -1]
            data2 = results[field2][:, -1]
        except:
            #no time axis
            data1 = results[field1]
            data2 = results[field2]
        
        #text and labels
        if i == j:
            #only plot the name in the middle
            ax.text(0.5, 0.5, field1,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform = ax.transAxes)  
        else:
            bins=None
            if log:
                bins='log'
            ax.hexbin(data2, data1, bins=bins, gridsize=gridsize, cmap=cmap)
        if i != len(FIELDS)-1:
            #xaxis off
            ax.set_xticklabels([])
        elif ylabels:
            try:
                ax.set_xlabel(ylabels.get(field1))
            except KeyError:
                info("no label specified for "+field1)        
        if j != 0:
            #yaxis off
            ax.set_yticklabels([])
        elif ylabels:
            try:
                ax.set_ylabel(ylabels.get(field2))
            except KeyError:
                info("no label specified for "+field2)
  
def multiplot_scatter(results, 
                      outcomes = [], 
                      column = None,
                      categories = None,
                      ylabels = {},
                      legend=False,
                      discretesize=__discretesize,):
    '''
    
    Generate a `R style pairs <http://www.stat.psu.edu/~dhunter/R/html/graphics/html/pairs.html>`_ 
    scatter multiplot. In case of time-series data, the end states are used.
    
    :param input: return from perform_experiments.
    :param outcomes: list of outcome of interest you want to plot. If empty, 
                     all outcomes are plotted note:  just names.
    :param column: name of the column in the cases array to group results by
    :param categories: set of categories to be used as a basis for grouping by. 
                       Categories is only meaningful if column is provided as 
                       well. 
    :param ylabels: ylabels is a dictionary with the outcome names as keys, the 
                    specified values will be used as labels for the y axis. 
    :param legend: boolean, if true, and there is a column specified for 
                   grouping, show a legend.
    :param discretesize: function to be used to turn a continuous column into 
                         intervals in order to use for grouping by.
    :rtype: a `figure <http://matplotlib.sourceforge.net/api/figure_api.html>`_ 
            instance.
    
    .. rubric:: an example of use
    
    >>> import expWorkbench.util as util
    >>> data = util.load_results(r'1000 flu cases.cPickle')
    >>> multiplot_scatter(data, column='policy', legend=True)
    
    generates 
    
    .. plot:: ../docs/source/pyplots/basicMultiplotScatter.py
    
   .. note:: the current implementation is limited to seven different 
          categories in case of column, categories, and/or discretesize.
          This limit is due to the colors specified in COLOR_LIST.    
    
    '''
    
    #unravel return from run_experiments   
    experiments, results = results
    
    #establish time axis
    try:
        results.pop('TIME')[0, :]
    except KeyError:
        pass
        
    #establish outcomes to plot
    if not outcomes:
        outcomes = results.keys()
    FIELDS = outcomes

    #establish basis for clustering, if specified
    if not categories:
        if column:
            categories = experiments[column]
            if categories.dtype == np.float64 or categories.dtype == np.float32:
                categories = discretesize(categories)
            else:
                categories=set(categories)
    
    grid = gridspec.GridSpec(len(FIELDS), len(FIELDS))                             
    
    #the plotting
    figure = plt.figure()
    
    if (categories != None) & legend:
        __make_legend(categories, figure)
     
    combis = [(field1, field2) for field1 in FIELDS for field2 in FIELDS]
    for field1, field2 in combis:
        i = FIELDS.index(field1)
        j = FIELDS.index(field2)
        ax = figure.add_subplot(grid[i,j])

        if categories:
            for x, category in enumerate(categories):
                if type(category)==tuple:
                    a = (experiments[column] >= category[0]) &\
                        (experiments[column]  < category[1])
                else:
                    a = experiments[column]==category
                
                try:
                    data1 = results[field1][a, -1]
                    data2 = results[field2][a, -1]
                except:
                    #no time axis
                    data1 = results[field1][a]
                    data2 = results[field2][a]
                
                facecolor = COLOR_LIST[x]
                edgecolor = 'k'
                if i==j: 
                    facecolor = 'white'
                    edgecolor = 'white'
                ax.scatter(data2, data1, 
                           facecolor=facecolor, edgecolor=edgecolor)
        else:
            try:
                data1 = results[field1][:, -1]
                data2 = results[field2][:, -1]
            except:
                #no time axis
                data1 = results[field1]
                data2 = results[field2]
            facecolor = 'b'
            edgecolor = 'k'
            if i==j: 
                facecolor = 'white'
                edgecolor = 'white'
            ax.scatter(data2, data1, 
                   facecolor=facecolor, edgecolor=edgecolor)
        
        #text and labels
        if i == j:
            #only plot the name in the middle
            ax.text(0.5, 0.5, field1,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform = ax.transAxes)  
        if i != len(FIELDS)-1:
            #xaxis off
            ax.set_xticklabels([])
        elif ylabels:
            try:
                ax.set_xlabel(ylabels.get(field1))
            except KeyError:
                info("no label specified for "+field1)        
        if j != 0:
            #yaxis off
            ax.set_yticklabels([])
        elif ylabels:
            try:
                ax.set_ylabel(ylabels.get(field2))
            except KeyError:
                info("no label specified for "+field2)   

#==============================================================================
# Test methods
#==============================================================================
def test_lines():
    from expWorkbench.EMAlogging import log_to_stderr, INFO
    from expWorkbench.model import SimpleModelEnsemble
    from examples.FLUvensimExample import FluModel
    
    log_to_stderr(level= INFO)
        
    model = FluModel(r'..\..\models\flu', "fluCase")
    ensemble = SimpleModelEnsemble()
    ensemble.set_model_structure(model)
    ensemble.parallel = True
    
    policies = [{'name': 'no policy',
                 'file': r'\FLUvensimV1basecase.vpm'},
                {'name': 'static policy',
                 'file': r'\FLUvensimV1static.vpm'},
                {'name': 'adaptive policy',
                 'file': r'\FLUvensimV1dynamic.vpm'}
                ]
    ensemble.add_policies(policies)
    
    results = ensemble.perform_experiments(10)
    
    lines(results, column='policy', density=True, log=True )
#    lines(results, column='fatality ratio region 1', density=False)
    plt.show()

def test_envelopes():
    from expWorkbench.EMAlogging import log_to_stderr, INFO
    from expWorkbench.model import SimpleModelEnsemble
    from examples.FLUvensimExample import FluModel
    
    log_to_stderr(level= INFO)
        
    model = FluModel(r'..\..\models\flu', "fluCase")
    ensemble = SimpleModelEnsemble()
    ensemble.parallel = True
    ensemble.set_model_structure(model)
    
    policies = [{'name': 'no policy',
                 'file': r'\FLUvensimV1basecase.vpm'},
                {'name': 'static policy',
                 'file': r'\FLUvensimV1static.vpm'},
                {'name': 'adaptive policy',
                 'file': r'\FLUvensimV1dynamic.vpm'}
                ]
    ensemble.add_policies(policies)
    
    results = ensemble.perform_experiments(100)
    
    envelopes(results, column="normal contact rate region 1", hist=False, fill=False)
#    envelopes(results, hist= False)
    
    plt.show()


def test_multiplot_lines():
    from expWorkbench.EMAlogging import log_to_stderr, INFO
    from expWorkbench.model import SimpleModelEnsemble
    from examples.FLUvensimExample import FluModel
    
    log_to_stderr(level= INFO)
        
    model = FluModel(r'..\..\models\flu', "fluCase")
    ensemble = SimpleModelEnsemble()
    ensemble.parallel = True
    ensemble.set_model_structure(model)
    
    policies = [{'name': 'no policy',
                 'file': r'\FLUvensimV1basecase.vpm'},
                {'name': 'static policy',
                 'file': r'\FLUvensimV1static.vpm'},
                {'name': 'adaptive policy',
                 'file': r'\FLUvensimV1dynamic.vpm'}
                ]
    ensemble.add_policies(policies)
    
    results = ensemble.perform_experiments(20)
    
    multiplot_lines(results, column='policy', legend=True)
    plt.show()

def test_multiplot_density():
    from expWorkbench.EMAlogging import log_to_stderr, INFO
    from expWorkbench.model import SimpleModelEnsemble
    from examples.FLUvensimExample import FluModel
    
    log_to_stderr(level= INFO)
        
    model = FluModel(r'..\..\models\flu', "fluCase")
    ensemble = SimpleModelEnsemble()
    ensemble.parallel = True
    ensemble.set_model_structure(model)

    policies = [{'name': 'no policy',
                 'file': r'\FLUvensimV1basecase.vpm'}]
    ensemble.add_policies(policies)
    
    results = ensemble.perform_experiments(100)
    
    multiplot_density(results)
    plt.show()

def test_multiplot_scatter():
    from expWorkbench.EMAlogging import log_to_stderr, INFO
    from expWorkbench.model import SimpleModelEnsemble
    from examples.FLUvensimExample import FluModel
    
    log_to_stderr(level= INFO)
        
    model = FluModel(r'..\..\models\flu', "fluCase")
    ensemble = SimpleModelEnsemble()
    ensemble.parallel = True
    ensemble.set_model_structure(model)
    
    policies = [{'name': 'no policy',
                 'file': r'\FLUvensimV1basecase.vpm'},
                {'name': 'static policy',
                 'file': r'\FLUvensimV1static.vpm'},
                {'name': 'adaptive policy',
                 'file': r'\FLUvensimV1dynamic.vpm'}
                ]
    ensemble.add_policies(policies)
    
    results = ensemble.perform_experiments(100)
    
    multiplot_scatter(results, column='policy', legend=True)
    plt.show()

if __name__ == '__main__':
    test_lines()
#    test_envelopes()
#    test_multiplot_scatter()
#    test_multiplot_lines()
#    test_multiplot_density()
 
