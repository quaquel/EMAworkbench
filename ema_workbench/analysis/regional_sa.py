'''

Module offers support for performing basic regional sensitivity analysis. The
module can be used to perform regional sensitivity analysis on all 
uncertainties specified in the experiment array, as well as the ability to 
zoom in on any given uncertainty in more detail. 


'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import operator

import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as rf
import seaborn as sns

# Created on Aug 18, 2015
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['plot_cdf',
           'plot_cdfs']

cp = sns.color_palette()

def build_legend(x,y):
    '''helper function for building a legend
    
    Parameters
    ----------
    x : ndarray
    y : ndarray
    
    '''
    proxies = []
    labels = []
    for i in range(np.max(y)+1):
        proxy = plt.Line2D([0,1], [0,1], color=cp[i+1])
        proxies.append(proxy) 
        labels.append('{} (N={})'.format(i,x[y==i].shape[0]))
    proxies.append(plt.Line2D([0,1], [0,1], lw=1,color='darkgrey'))
    labels.append('unconditioned')
    return proxies, labels


def plot_discrete_cdf(ax, unc, x, y, xticklabels_on,
                     ccdf):
    '''plot a discrete cdf on ax for data,
    grouping data by logical index.
    
    Parameters
    ----------
    ax : matplotlib axes
    unc : str
    x  : ndarray
    y : ndarray
    xticklabels_on : bool
    ccdf : bool
    
    '''
    cats = sorted(set(x))
    n_cat = len(cats)
    for i in range(np.max(y)+1):
        data_i = x[y==i]

        freqs = []
        for cat in cats:
            freq = data_i[data_i==cat].shape[0]/data_i.shape[0]
            freqs.append((cat, freq))
            
        freqs.sort(key=operator.itemgetter(1))
        cats = map(operator.itemgetter(0), freqs)
        freqs = map(operator.itemgetter(1), freqs)
        
        cum_freq = 0
        for j, freq in enumerate(freqs):
            cum_freq += freq
            
            freq = cum_freq
            
            if ccdf:
                freq = 1 - cum_freq

                
            x_plot = [j*1, j*1+1]
            y_plot = [freq,]*2
            
            ax.plot(x_plot, y_plot ,c=cp[i+1], label=i==1)
            ax.scatter(x_plot[0], y_plot[0], edgecolors=cp[i+1], facecolors=cp[i+1],
                      linewidths=1, zorder=2)
            ax.scatter(x_plot[1], y_plot[0], edgecolors=cp[i+1], facecolors='white',
                      linewidths=1, zorder=2)
            
            # misnomer
            cum_freq_un = (j+1)/n_cat
            if ccdf:
                cum_freq_un = (len(freqs)-j-1)/n_cat
            
            ax.plot(x_plot, [cum_freq_un,]*2, lw=1, c='darkgrey',
                   zorder=1, label='x==y')
            ax.scatter(x_plot[0], cum_freq_un, edgecolors='darkgrey', 
                       facecolors='darkgrey', linewidths=1,
                       zorder=1)
            ax.scatter(x_plot[1], cum_freq_un, edgecolors='darkgrey',
                       facecolors='white', linewidths=1,
                      zorder=1)
            

    ax.set_xticklabels([])
    if xticklabels_on:
        for i,cat in enumerate(cats):
            ax.text(i*1+0.5,-0.1, cat, ha='center', rotation=45)

    ax.set_ylim(ymin=-0.01, ymax=1.01)

    xmin=-0.02*n_cat
    xmax=n_cat+0.02*n_cat
    ax.set_xticks(np.linspace(xmin, xmax, 4))
    ax.set_xlim(xmin=xmin, xmax=xmax)

    
def plot_continuous_cdf(ax, unc, x, y, xticklabels_on,
                       ccdf):
    '''plot a continuous cdf on ax for data,grouping data by the groups
    specified in y.
    
    Parameters
    ----------
    ax : matplotlib axes 
    unc : str
    x  : ndarray
    y : ndarray
    xticklabels_on : bool
    ccdf : bool
    
    '''
    
    for i in range(np.max(y)+1):
        data_i = x[y==i]
        sorted_data = np.sort(data_i)
        yvals = np.arange(len(sorted_data))/float(len(sorted_data))
        if ccdf:
            yvals = 1 - yvals
        ax.plot(sorted_data,yvals, color=cp[i+1], label='{}'.format(i))
    
    x0 = min(x)
    x1 = max(x)
    
    sorted_data = np.sort(x)
    yvals = np.arange(len(x))/float(len(x))
    if ccdf:
        yvals = 1 - yvals
    
    ax.plot(sorted_data, yvals, c='darkgrey', lw=1)         
    
    ax.set_xlim(xmin=x0, xmax=x1)
    xticklocs = np.linspace(x0, x1, 4)
    ax.set_xticks(xticklocs)
    if xticklabels_on:
        ax.set_xticklabels(['{:.2g}'.format(entry) for entry in xticklocs])
    else:
        ax.set_xticklabels([])

        
def plot_cdf(ax, unc, x, y, discrete=False,
            legend=False, xticklabels_on=False,
            yticklabels_on=False, ccdf=False):
    '''plot cdf for x conditional on y
    
    Parameters
    ----------
    ax : Axes instance
         axes on which to plot the cdf
    unc : str
          the name of the uncertainty
    x : ndarray of shape (1,)
        the data to plot
    y : ndarray(1,)
        the categorization for the data
    discrete : bool, optional
               if true, plot a discrete cdf. Default is false.
    legend : bool, optional
    xticklabels_on : bool, optional
    ccdf : bool, optional
           if true, plot a complementary cdf instead of a normal cdf.
    
    '''

    if discrete:
        plot_discrete_cdf(ax, unc, x, y, xticklabels_on,
                         ccdf)
    else:
        plot_continuous_cdf(ax, unc, x, y, xticklabels_on,
                           ccdf)

    if legend:
        proxies, labels = build_legend(x,y)
        ax.legend(proxies, labels, loc='best')

    yticklocs = np.linspace(0,1,4)
    ax.set_yticks(yticklocs)
    
    if yticklabels_on:
        ax.set_yticklabels(['$0$', '$\\frac{1}{3}$', '$\\frac{2}{3}$', '$1$'])
    else:
        ax.set_yticklabels([])
        
    if xticklabels_on:
        ax.set_xlabel(str(unc))
    else:
        x0, x1 = ax.get_xlim()
        ax.text(x0+0.01*x1, 1, str(unc), va='top', ha='left')
        

def plot_cdfs(x, y, ccdf=False):
    '''plot cumulative density functions for each column in x, based on the 
    classification specified in y.
    
    Parameters
    ----------
    x : recarray
        the experiments to use in the cdfs
    y : ndaray 
        the categorization for the data
    ccdf : bool, optional
           if true, plot a complementary cdf instead of a normal cdf.
    
    '''
    x = rf.drop_fields(x, "scenario_id", asrecarray=True)
    uncs = rf.get_names(x.dtype)
    cp = sns.color_palette()
    
    n_col = 4
    n_row = len(uncs)//n_col +1
    size = 3 
    aspect = 1
    figsize = n_col * size * aspect, n_row * size
    fig, axes = plt.subplots(n_row, n_col,
                             figsize=figsize,
                             squeeze=False)

    for i, unc in enumerate(uncs):
        discrete = False
        
        i_col = i % n_col
        i_row = i // n_col
        ax = axes[i_row, i_col]
        
        data = x[unc]
        if x.dtype[unc] == np.dtype('O'):
            discrete = True
        plot_cdf(ax, unc, data, y, discrete, ccdf=ccdf)
    
    # last row might contain empty axis, 
    # let's make them disappear
    i_row = len(uncs) // n_col
    i_col = len(uncs) % n_col
    for i_col in range(i_col, n_col):
        ax = axes[i_row, i_col]
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_yticks([])
        
        sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)
    
    proxies, labels = build_legend(x, y)
    
    fig.legend(proxies, labels, "upper center")

    return fig