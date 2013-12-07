'''
.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>



'''
from __future__ import division
from types import StringType, DictType, ListType
import copy

import numpy as np
# import scipy.stats.kde as kde

from sklearn.neighbors import KernelDensity


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec 
import matplotlib.cm as cm

from expWorkbench.ema_exceptions import EMAError
from expWorkbench.ema_logging import info, warning

'''

Default color list

'''
COLOR_LIST = ['b',
              'g',
              'r',
              'c',
              'm',
              'y',
              'k',
              'b',
              'g',
              'r',
              'c',
              'm',
              'y',
              'k'
                ]


'''

Parameter controlling whether tight layout from matplotlib should be used

'''
TIGHT = False

'''

Default key for time

'''
TIME = "TIME"

ENVELOPE = 'envelope'
LINES = 'lines'
ENV_LIN = "env_lin"
KDE = 'kde'
HIST = 'hist'
BOXPLOT = 'box plot'

# used for legend
LINE = 'line'
PATCH = 'patch'
SCATTER = 'scatter'

#see http://matplotlib.sourceforge.net/users/customizing.html for details
#mpl.rcParams['savefig.dpi'] = 600
#mpl.rcParams['axes.formatter.limits'] = (-5, 5)
#mpl.rcParams['font.family'] = 'serif'
#mpl.rcParams['font.serif'] = 'Times New Roman'
#mpl.rcParams['font.size'] = 12.0

#==============================================================================
# actual plotting functions
#==============================================================================
def plot_envelope(ax, j, time, value, fill):
    '''
    
    Helper function, responsible for plotting an envelope.
    
    :param ax:
    :param j:
    :param time:
    :param time:
    :param value:
    :param fill:
    
    
    '''
    
    #plot minima and maxima
    minimum = np.min(value, axis=0)
    maximum = np.max(value, axis=0)

    color = COLOR_LIST[j]
    
    if fill:
        ax.plot(time, minimum, color=color, alpha=0.3)
        ax.plot(time, maximum, color=color, alpha=0.3)
        ax.fill_between(time, 
                        minimum,
                        maximum,
                        facecolor=color, 
                        alpha=0.3,
                        )
    else:
        ax.plot(time, minimum, c=color)
        ax.plot(time, maximum, c=color)


def plot_histogram(ax, values, log=False):
    '''
    
    Helper function, responsible for plotting a histogram
    
    :param ax:
    :param values:
    :param log:
    
    
    '''
    if type(values)==ListType:
        color = COLOR_LIST[0:len(values)]
    else:
        color='b'
    a = ax.hist(values, 
             bins=11, 
             orientation='horizontal',
             histtype='bar', 
             normed = True,
             color=color,
             log=log)
    if not log:
        ax.set_xticks([0, ax.get_xbound()[1]])
    return a
  
def plot_kde(ax, kde_x, kde_y, j, log=False):
    '''
    
    Helper function, responsible for plotting a KDE.
    
    :param ax: the axes on which to plot the kde
    :param kde_x: the x values of the kde
    :param kde_y: the y values of the kde
    :param j: the index of the group being shown
    :param log: boolean, whether to log scale the data are not
    
    
    '''

    color = COLOR_LIST[j]
    
    ax.plot(kde_x, kde_y, c=color, ms=1, markevery=20)

    if log:
        ax.set_xscale('log')
    else:
        maximum = ax.get_xaxis().get_view_interval()[1]
        
        ax.set_xticks([int(0), 
                      ax.get_xaxis().
                      get_view_interval()[1]])
        labels =["{0:.2g}".format(0), "{0:.2g}".format(ax.get_xlim()[1])]
        ax.set_xticklabels(labels)

def plot_boxplots(ax, values, group_labels=None, log=False):
    ax.boxplot(values)
    if group_labels:
        ax.set_xticklabels(group_labels, rotation='vertical')

def simple_density(density, value, ax_d, ax, loc=-1, **kwargs):
    '''
    
    Helper function, responsible for producing a density plot
    
    :param density: type of density
    :param value: the data for which to calculate the density
    :param ax_d:
    :param ax:
    :param kwargs: 
    
    
    '''
    
    if density==KDE:
        kde_x, kde_y = determine_kde(value[:,loc])
        plot_kde(ax_d, kde_x, kde_y, 0, **kwargs)
    elif density==HIST:
        plot_histogram(ax_d, value[:,-1], log=kwargs.get('log'))
    elif density==BOXPLOT:
        plot_boxplots(ax_d, value[:,-1], log=kwargs.get('log'))
    else:
        raise EMAError("unknown density plot type")
        
    ax_d.get_yaxis().set_view_interval(
                 ax.get_yaxis().get_view_interval()[0],
                 ax.get_yaxis().get_view_interval()[1]) 
    ax_d.set_ylim(ymin=ax.get_yaxis().get_view_interval()[0],
              ymax=ax.get_yaxis().get_view_interval()[1])
    
def simple_kde(outcomes, outcomes_to_show, colormap, log, minima, maxima):
    '''
    
    Helper function for generating a density heatmap over time
    
    :param outcomes:
    :param outcomes_to_show:
    :param colormap:
    :param log:
    :param minima:
    :param maxima:
    
    
    '''


    figure, grid = make_grid(outcomes_to_show)
    axes_dict = {}
    
    # do the plotting
    for i, outcome_to_plot in enumerate(outcomes_to_show):
        ax = figure.add_subplot(grid[i,0])
        axes_dict[outcome_to_plot] = ax
        
        outcome = outcomes[outcome_to_plot]
        
        size_kde = 100
        np.zeros
        kde_over_time = np.zeros(shape=(size_kde, outcome.shape[1]))
        ymin = minima[outcome_to_plot]
        ymax = maxima[outcome_to_plot]
        
        #make kde over time
        for j in range(outcome.shape[1]):
            kde_x = determine_kde(outcome[:, j], size_kde, ymin, ymax)[0]
            if log:
                kde_x = np.log(kde_x+1)
            kde_over_time[:, j] = kde_x
        ax.matshow(kde_over_time, 
                    cmap=cm.__dict__[colormap])
#        a = ax.get_yticklabels()
#        b = len(a)
#        c = ymax-ymin/b
#        d = ymax+c
#        e = np.arange(ymin, d, c)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("time")
        ax.set_ylabel(outcome_to_plot)
        
    return figure, axes_dict


def make_legend(categories,
                figure,
                ncol=3,
                legend_type=LINE,
                alpha=1):
    '''
    
    
    :param categories:
    :param figure:
    :param legend_type:
    
    
    '''
    
    some_identifiers = []
    labels = []
    for i, category in enumerate(categories):
        if legend_type == LINE:    
            artist = plt.Line2D([0,1], [0,1], color=COLOR_LIST[i], 
                                alpha=alpha) #TODO
        elif legend_type == SCATTER:
            marker_obj = mpl.markers.MarkerStyle('o')
            path = marker_obj.get_path().transformed(
                             marker_obj.get_transform())
            artist  = mpl.collections.PathCollection((path,),
                                        sizes = [20],
                                        facecolors = COLOR_LIST[i],
                                        edgecolors = 'k',
                                        offsets = (0,0))
        elif legend_type == PATCH:
            artist = plt.Rectangle((0,0), 1,1, edgecolor=COLOR_LIST[i],
                                   facecolor=COLOR_LIST[i], alpha=alpha)

        some_identifiers.append(artist)
        
        if type(category) == tuple:
            label =  '%.2f - %.2f' % category 
        else:
            label = category
        
        labels.append(str(label))

#    ncol = int(np.ceil(len(categories)/3))
    
    figure.legend(some_identifiers, labels, ncol=ncol,
                      loc='upper center', borderaxespad=0.1)

def determine_kde(data, 
                  size_kde=1000,
                  ymin=None,
                  ymax=None):
    '''
    
    Helper function responsible for performing a KDE    
    
    :param data:
    
    
    '''
    if not ymin:
        ymin = np.min(data)
    if not ymax:
        ymax = np.max(data)
    
    kde_y = np.linspace(ymin, ymax, size_kde)[::-1]
    
    try:
#         kde_x = kde.gaussian_kde(data)
#         kde_x = kde_x.evaluate(kde_y)

        kde_skl = KernelDensity(bandwidth=0.2)
        kde_skl.fit(data)
        # score_samples() returns the log-likelihood of the samples
        log_pdf = kde_skl.score_samples(kde_y)
        kde_x =  np.exp(log_pdf)

    except np.linalg.LinAlgError as e:
        warning(e)
        kde_x = np.zeros(kde_y.shape)
    
    return kde_x, kde_y
        
def filter_scalar_outcomes(outcomes):
    '''
    Helper function that removes non time series outcomes from all the 
    outcomes.
    
    :param outcomes:
    :return: the filtered outcomes
    
    
    '''
    outcomes_to_remove = []
    for key, value in outcomes.items():
        if len(value.shape) <2:
            outcomes_to_remove.append(key)
            info("%s not shown because it is not time series data" %key)
    [outcomes.pop(entry) for entry in outcomes_to_remove]
    return outcomes

def determine_time_dimension(outcomes):
    '''
    
    :param outcomes:
    
    
    '''

    time = None
    try:
        time = outcomes['TIME']
        time = time[0, :]
        outcomes.pop('TIME')
    except KeyError:
        values = iter(outcomes.values())
        for value in values:
            if len(value.shape)==2:
                time =  np.arange(0, value.shape[1])
                break
    if time==None:
        info("no time dimension found in results")
    return time, outcomes    

def group_results(experiments, outcomes, group_by, grouping_specifiers):
    '''
    Helper function that takes the experiments and results and returns a list 
    based on groupings. Each element in the dictionary contains the experiments 
    and results for a particular group, the key is the grouping specifier.
    
    :param experiments:
    :param outcomes:
    :param group_by: The column in the experiments array to which the 
                     grouping specifiers apply. If the name is'index'
                     it is assumed that the grouping specifiers are valid
                     indices for numpy.ndarray .
    :param grouping_specifiers: An iterable of grouping specifiers. A grouping 
                    specifier is a unique identifier in case of grouping by 
                    categorical uncertainties. It is a tuple in case of 
                    grouping by a parameter uncertainty. In this cose, the code
                    treats the tuples as half open intervals, apart from the 
                    last entry, which is treated as closed on both sides.  
                    In case of 'index', the iterable should be a dictionary 
                    with the name for each group as key and the value being a 
                    valid index for numpy.ndarray. 
    :return: A dictionary with the experiments and results for each group, the
             the grouping specifier is used as key
             
    ..note:: In case of grouping by parameter uncertainty, the list of 
             grouping specifiers is sorted. The traversal assumes half open
             intervals, where the upper limit of each interval is open, except 
             for the last interval which is closed.
    
    
    '''
    groups = {}
    
    if group_by != 'index':
        column_to_group_by = experiments[group_by]
        grouping_specifiers = sorted(grouping_specifiers)
    else:
        grouping_specifiers = grouping_specifiers.items()
    
    for grouping_specifier in grouping_specifiers:
        if isinstance(grouping_specifier, tuple):
            if isinstance(grouping_specifier[1], np.ndarray):
                # the grouping is based on indices
                logical = grouping_specifier[1]
                grouping_specifier = grouping_specifier[0]
            
            else:
                # the grouping is a continuous uncertainty
                lower_limit, upper_limit = grouping_specifier
                
                #check whether it is the last grouping specifier
                if grouping_specifiers.index(grouping_specifier) ==\
                    len(grouping_specifiers)-1:
                    #last case
                    
                    logical = (column_to_group_by>=lower_limit) &\
                               (column_to_group_by<=upper_limit)
                else:
                    logical = (column_to_group_by>=lower_limit) &\
                               (column_to_group_by<upper_limit)
        else:
            # the grouping is an integer or categorical uncertainty
            logical = column_to_group_by==grouping_specifier
        
        group_outcomes = {}
        for key, value in outcomes.items():
            value = value[logical]
            group_outcomes[key] = value
        groups[grouping_specifier] = (experiments[logical], group_outcomes)
        
    return groups

def make_continuous_grouping_specifiers(array, nr_of_groups=5):
    '''
    Helper function for discretesizing a continuous array. By default, the 
    array is split into 5 equally wide intervals.
    
    :param array: a 1-d array that is to be turned into discrete intervals.
    :param nr_of_groups:
    :return: list of tuples with the lower and upper bound of the intervals. 
    
    .. note:: this code only produces intervals. :func:`group_results` uses
              these intervals in half-open fashion, apart from the last 
              interval: [a, b), [b,c), [c,d]. That is, both the end point
              and the start point of the range of the continuous array are 
              included.
    
    
    '''
    
    minimum = np.min(array)
    maximum = np.max(array)
    step = (maximum-minimum)/nr_of_groups
    a = [(minimum+step*x, minimum+step*(x+1)) for x in range(nr_of_groups)]
    assert a[0][0] == minimum
    assert a[-1][1] == maximum
    return a

def prepare_pairs_data(results, 
                        outcomes_to_show=None,
                        group_by=None,
                        grouping_specifiers=None,
                        point_in_time=-1,
                        filter_scalar=True):
    '''
    
    
    :param results:
    :param outcomes_to_show:
    :param group_by:
    :param grouping_specifiers:
    :param point_in_time:
    
    
    '''
    if type(outcomes_to_show) == StringType:
        raise EMAError("for pair wise plotting, more than one outcome needs to be provided")
    
    outcomes, outcomes_to_show, time, grouping_labels = prepare_data(results, 
                                                        outcomes_to_show,
                                                        group_by,
                                                        grouping_specifiers,
                                                        filter_scalar)

    def filter_outcomes(outcomes, point_in_time):
        new_outcomes = {}
        for key, value in outcomes.items():
            if len(value.shape)==2:
                new_outcomes[key] = value[:, point_in_time]
            else:
                new_outcomes[key] = value
        return new_outcomes
    
    if point_in_time:
        if point_in_time != -1:
            point_in_time = np.where(time==point_in_time)
        
        if group_by:
            new_outcomes = {}
            for key, value in outcomes.items():
                new_outcomes[key] = filter_outcomes(value, point_in_time)
            outcomes = new_outcomes
        else:
            outcomes = filter_outcomes(outcomes, point_in_time)
    return outcomes, outcomes_to_show, grouping_labels 

def prepare_data(results,
                 outcomes_to_show=None,
                 group_by=None,
                 grouping_specifiers=None,
                 filter_scalar = True):
    '''
    
    
    :param results: the results to visualize
    :param outcomes_to_show:
    :param group_by:
    :param grouping_specifiers:
    :param filter_scalar:
    
    
    '''


    results = copy.deepcopy(results)
    
    #unravel results
    experiments, outcomes = results

    time, outcomes = determine_time_dimension(outcomes)
    
    # remove outcomes that are not to be shown
    if outcomes_to_show:
        if type(outcomes_to_show) == StringType:
            outcomes_to_show  = [outcomes_to_show]
        
        [outcomes.pop(entry) for entry in\
         set(outcomes.keys()) - set(outcomes_to_show)]

    # filter the outcomes to exclude scalar values
    if filter_scalar:
        outcomes = filter_scalar_outcomes(outcomes)
    outcomes_to_show = outcomes.keys()
        
    # group the data if desired
    if group_by:
        if not grouping_specifiers:
            #no grouping specifier, so infer from the data
            if group_by=='index':
                raise EMAError("no grouping specifiers provided while trying to group on index")
            else:
                column_to_group_by = experiments[group_by]
                if column_to_group_by.dtype == np.object:
                    grouping_specifiers = set(column_to_group_by)
                else:
                    grouping_specifiers = make_continuous_grouping_specifiers(column_to_group_by, 
                                                        grouping_specifiers)
            grouping_labels=sorted(grouping_specifiers)
        else:
            if type(grouping_specifiers) == StringType:
                grouping_specifiers = [grouping_specifiers]
                grouping_labels=grouping_specifiers
            elif type(grouping_specifiers) == DictType:
                grouping_labels=sorted(grouping_specifiers.keys())
            else:
                grouping_labels=grouping_specifiers
                
        
        outcomes = group_results(experiments, outcomes, group_by,\
                                 grouping_specifiers)
        
        new_outcomes = {}
        for key, value in outcomes.items():
            new_outcomes[key] = value[1]
        outcomes = new_outcomes
    else:
        grouping_labels=[]

    return outcomes, outcomes_to_show, time, grouping_labels

def do_titles(ax, titles, outcome):
    '''
    Helper function for setting the title on an ax
    
    :param ax: the ax on which to set the title
    :param titles: a dict which maps outcome names to titles
    :param outcome: the outcome plotted in the ax.
    
    
    '''
    
    if type(titles)==DictType:
        if not titles:
            ax.set_title(outcome)
        else:
            try:
                ax.set_title(titles[outcome])
            except KeyError:
                warning("key error in do_titles, no title provided for `%s`" % (outcome))
                ax.set_title(outcome)

def do_ylabels(ax, ylabels, outcome):
    '''
    Helper function for setting the y labels on an ax
    
    :param ax: the ax on which to set the y label
    :param titles: a dict which maps outcome names to y labels
    :param outcome: the outcome plotted in the ax.
    
    
    '''
    
    if type(ylabels)==DictType:
        if not ylabels:
            ax.set_ylabel(outcome)
        else:
            try:
                ax.set_ylabel(ylabels[outcome])
            except KeyError:
                warning("key error in do_ylabels, no ylabel provided for `%s`" % (outcome))
                ax.set_ylabel(outcome)    

def make_grid(outcomes_to_show, density=None):
    '''
    Helper function for making the grid that specifies the size and location
    of the various axes. 
    
    :param outcomes_to_show: the list of outcomes to show
    :param density: boolean, whether to show density or not
    
    
    '''

    
    # make the plotting grid
    if density:
        grid = gridspec.GridSpec(len(outcomes_to_show), 2,
                                 width_ratios = [4, 1])
    else:
        grid = gridspec.GridSpec(len(outcomes_to_show), 1) 
    grid.update(wspace = 0.1,
                hspace = 0.4)
    
    figure = plt.figure()
    return figure, grid

