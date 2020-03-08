'''This module offers a general purpose parallel coordinate plotting Class
using matplotlib.


'''
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from sklearn import preprocessing

from pandas.api.types import CategoricalDtype

# Created on 11 Sep 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['ParallelAxes',
           'get_limits']


def setup_parallel_plot(labels, minima, maxima, formatter=None, fs=14, rot=90):
    '''helper function for setting up the parallel axes plot

    Parameters
    ----------
    labels : list of str
    minima : ndarray
    maxima : ndarray
    formattter : dict with precision format strings for labels, optional
                 defaults to .2f
    fs : int, optional
               fontsize for defaults text items
    rot : float, optional
          rotation of axis labels

    '''
    if formatter is None:
        formatter = {}
    
    sns.set_style('white')
    # labels is a list, minima and maxima pd series
    nr_columns = len(labels)
    fig = plt.figure()
    axes = []
    tick_labels = {}

    # we need one axes less than the shape
    for i, label in enumerate(labels[:-1]):
        i += 1
        ax = fig.add_subplot(1, nr_columns - 1, i, ylim=(-0.1, 1.1))
        axes.append(ax)
        ax.set_xlim([i, i + 1])
        ax.xaxis.set_major_locator(ticker.FixedLocator([i]))
        ax.xaxis.set_ticklabels([labels[i - 1]], rotation=rot, fontsize=fs)
        ax.xaxis.set_tick_params(bottom=False, top=False)

        # let's put our own tick labels
        ax.yaxis.set_ticks([])
        
        # TODO::consider moving to f-strin
        # so 
        # label = f"{{maxima[label]}:{precision}}"
        try:
            precision = formatter[label]
        except KeyError:
            precision = ".2f"
        
        max_label = f"{maxima[label]:{precision}}"
        min_label = f"{minima[label]:{precision}}"
        max_label = ax.text(i, 1.01, max_label, va="bottom",
                            ha="center", fontsize=fs)
        min_label = ax.text(i, -0.01, min_label, va="top",
                            ha="center", fontsize=fs)
        tick_labels[label] = (min_label, max_label)

        ax.spines['left'].set_bounds(0, 1)
        ax.spines['right'].set_bounds(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    # for the last axis, we need 2 ticks (also for the right hand side
    ax.xaxis.set_major_locator(ticker.FixedLocator([i, i + 1]))
    ax.xaxis.set_ticklabels(labels[i - 1:i + 1], fontsize=fs, rotation=rot)

    label = labels[-1]

    try:
        precision = formatter[label]
    except KeyError:
        precision = ".2f"
    
    max_label = f"{maxima[label]:{precision}}"
    min_label = f"{minima[label]:{precision}}"

    max_label = ax.text(i + 1, 1.01, max_label, va="bottom",
                        ha="center", fontsize=fs)
    min_label = ax.text(i + 1, -0.01, min_label, va="top",
                        ha="center", fontsize=fs)
    tick_labels[label] = (min_label, max_label)

    # add the tick labels to the rightmost spine
    for tick in ax.yaxis.get_major_ticks():
        tick.label2On = True

    # stack the subplots together
    plt.subplots_adjust(wspace=0)

    return fig, axes, tick_labels


def get_limits(data):
    '''helper function to get limits of a FataFrame that can serve as input
    to ParallelAxis

    Parameters
    ----------
    data : DataFrame

    Returns
    -------
    DataFrame

    '''
    def limits(x):
        if x.dtype == 'object':
            return pd.Series([set(x), set(x)])
        else:
            return pd.Series([x.min(), x.max()])

    return data.apply(limits)


class ParallelAxes(object):
    '''Base class for creating a parallel axis plot.

    Parameters
    ----------
    limits : DataFrame
             A DataFrame specifying the limits for each dimension in the
             data set. For categorical data, the first cell should contain all
             categories. See get_limits for more details.
    formattter : dict , optional
                 dict with precision format strings for minima and maxima, use
                 column name as key. If column is not present, or no formatter
                 dict is provided, precision formatting defaults to .2f
    fontsize : int, optional
               fontsize for defaults text items
    rot : float, optional
          rotation of axis labels

    '''

    def __init__(self, limits, formatter=None, fontsize=14, rot=90):
        '''

        Parameters
        ----------
        limits : DataFrame
                 categorical data, first cell should contain all categories
        formatter : dict, optional
                    specify precision formatters for minima and maxima,
                    defaults to .2f
                     
        fontsize : int, optional
                   fontsize for defaults text items
        rot : float, optional
              rotation of axis labels

        '''
        self.limits = limits.copy() # copy to avoid side effects
        self.recoding = {}
        self.flipped_axes = set()
        self.axis_labels = list(limits.columns.values)
        self.fontsize = fontsize

        # recode data
        for column, dtype in limits.dtypes.iteritems():
            if dtype == 'object':
                cats = limits[column][0]
                self.recoding[column] = CategoricalDtype(categories=cats,
                                                         ordered=False)
                limits.ix[:, column] = [0, len(cats) - 1]

        self.normalizer = preprocessing.MinMaxScaler()
        self.normalizer.fit(self.limits)

        fig, axes, ticklabels = setup_parallel_plot(
                        self.axis_labels, limits.min(), limits.max(),
                        fs=self.fontsize, rot=rot, formatter=formatter)
        self.fig = fig
        self.axes = axes
        self.ticklabels = ticklabels
        self.datalabels = []

        # TODO:: can't we force the wspace attribute instead having
        # to reset it after tight_layout?
        plt.tight_layout(h_pad=0, w_pad=0)
        plt.subplots_adjust(wspace=0)

    def plot(self, data, color=None, label=None, **kwargs):
        '''plot data on parallel axes

        Parameters
        ----------
        data : DataFrame or Series
        color : valid mpl color, optional
        label : str, optional

        any additional kwargs will be passed to matplotlib's plot
        method.

        Data is normalized using the limits specified when initializing
        ParallelAxis.

        '''
        data = data.copy() # copy to avoid side effects
        
        if isinstance(data, pd.Series):
            data = data.to_frame().T

        if label:
            self.datalabels.append((label, color))
            
        # ensures any data to be plotted is in the same order
        # as the limits
        data = data[self.axis_labels]

        # recode the data
        recoded = data.copy()
        for key, value in self.recoding.items():
            recoded[key] = data[key].astype(value).cat.codes

        # normalize the data
        normalized_data = pd.DataFrame(self.normalizer.transform(recoded),
                                       columns=recoded.columns)

        # plot the data
        self._plot(normalized_data, color=color, **kwargs)

    def legend(self):
        '''add a legend to the figure'''

        artists = []
        labels = []
        for label, color in self.datalabels:
            artist = plt.Line2D([0, 1], [0, 1], color=color)
            artists.append(artist)
            labels.append(label)

        self.fig.legend(
            artists,
            labels,
            ncol=1,
            fontsize=self.fontsize,
            loc=2,
            borderaxespad=0.1,
            bbox_to_anchor=(
                1.025,
                0.925))
        plt.tight_layout(h_pad=0, w_pad=0)
        plt.subplots_adjust(wspace=0)

    def _plot(self, data, **kwargs):
        '''Plot the data onto the paralel axis

        Parameters
        ----------
        data : DataFrame

        '''

        j = -1
        for ax, label_i, label_j in zip(self.axes, self.axis_labels[:-1],
                                        self.axis_labels[1::]):
            plotdata = data.loc[:, [label_i, label_j]]
            j += 1
            lines = ax.plot([j + 1, j + 2], plotdata.values.T, **kwargs)

            if label_i in self.flipped_axes:
                self._update_plot_data(ax, 0, lines=lines)
            if label_j in self.flipped_axes:
                self._update_plot_data(ax, 1, lines=lines)

    def invert_axis(self, axis):
        '''flip direction for specified axis

        Parameters
        ----------
        axis : str or list of str

        '''
        if isinstance(axis, str):
            axis = [axis]
        for entry in axis:
            self._invert_axis(entry)

        # keep track of flipped axes
        for entry in axis:
            if entry not in self.flipped_axes:
                self.flipped_axes.add(entry)
            else:
                self.flipped_axes.remove(entry)

    def _invert_axis(self, axis):
        '''

        Parameters
        ----------

        '''

        ids = self._get_axes_ids(axis)

        if len(ids) == 1:
            id = ids[0]  # @ReservedAssignment
            if id == 0:
                index = 0
            else:
                index = 1

            ax = self.axes[id]
            self._update_plot_data(ax, index)
        else:
            for i, direction in enumerate(ids[::-1]):
                self._update_plot_data(self.axes[direction], i)

        self._update_ticklabels(axis)

    def _update_plot_data(self, ax, index, lines=None):
        '''

        Parameters
        ----------
        index : {0, 1}

        '''
        if lines is None:
            lines = ax.get_lines()

        for line in lines:
            ydata = line.get_data()[1]
            ydata[index] = 1 - ydata[index]
            line.set_ydata(ydata)

    def _update_ticklabels(self, axis):
        '''

        Parameters
        ----------
        axis : str

        '''

        for label in self.ticklabels[axis]:
            x, y = label.get_position()
            if y == -0.01:
                y = 1.01
                label.set_va('bottom')
            else:
                y = -0.01
                label.set_va('top')
            label.set_position((x, y))

# TODO:: more fine-grained control for intermediate ticklabels
#        probably enable this by default for categorical axes
#        while having it disabled for continuous variables
# from http://benalexkeen.com/parallel-coordinates-in-matplotlib/
#         # Set the tick positions and labels on y axis for each plot
#         # Tick positions based on normalised data
#         # Tick labels are based on original data
#         def set_ticks_for_axis(dim, ax, ticks):
#             min_val, max_val, val_range = min_max_range[cols[dim]] # the limits
#             step = val_range / float(ticks-1)
#             tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
#             norm_min = df[cols[dim]].min()
#             norm_range = np.ptp(df[cols[dim]])
#             norm_step = norm_range / float(ticks-1)
#             ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
#             ax.yaxis.set_ticks(ticks)
#             ax.set_yticklabels(tick_labels)

    def _get_axes_ids(self, column):
        '''

        Parameters
        ----------
        column : str

        '''
        index = self.limits.columns.get_loc(column)
        if index == 0 or index >= (len(self.axes)):
            index = min(index, (len(self.axes) - 1))
            return (index,)
        else:
            other_index = index - 1
            return other_index, index
