"""

Module offers support for performing basic regional sensitivity analysis. The
module can be used to perform regional sensitivity analysis on all
uncertainties specified in the experiment array, as well as the ability to
zoom in on any given uncertainty in more detail.

"""
import math
import operator

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Created on Aug 18, 2015
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ["plot_cdfs"]

cp = sns.color_palette()


def build_legend(x, y):
    """helper function for building a legend

    Parameters
    ----------
    x : ndarray
    y : ndarray

    """
    proxies = []
    labels = []
    for i in range(np.max(y) + 1):
        proxy = plt.Line2D([0, 1], [0, 1], color=cp[i + 1])
        proxies.append(proxy)
        labels.append(f"{i} (N={x[y == i].shape[0]})")
    proxies.append(plt.Line2D([0, 1], [0, 1], lw=1, color="darkgrey"))
    labels.append("unconditioned")
    return proxies, labels


def plot_discrete_cdf(ax, unc, x, y, xticklabels_on, ccdf):
    """plot a discrete cdf on ax for data,
    grouping data by logical index.

    Parameters
    ----------
    ax : matplotlib axes
    unc : str
    x  : ndarray
    y : ndarray
    xticklabels_on : bool
    ccdf : bool

    """
    cats = sorted(set(x))
    n_cat = len(cats)
    for i in range(np.max(y) + 1):
        data_i = x[y == i]

        freqs = []
        for cat in cats:
            freq = data_i[data_i == cat].shape[0] / data_i.shape[0]
            freqs.append((cat, freq))

        freqs.sort(key=operator.itemgetter(1))
        cats = list(map(operator.itemgetter(0), freqs))
        freqs = list(map(operator.itemgetter(1), freqs))

        cum_freq = 0
        for j, freq in enumerate(freqs):
            cum_freq += freq

            freq = cum_freq

            if ccdf:
                freq = 1 - cum_freq

            x_plot = [j * 1, j * 1 + 1]
            y_plot = [freq] * 2

            ax.plot(x_plot, y_plot, c=cp[i + 1], label=i == 1, marker="o")

            # misnomer
            cum_freq_un = (j + 1) / n_cat
            if ccdf:
                cum_freq_un = (len(freqs) - j - 1) / n_cat

            ax.plot(
                x_plot, [cum_freq_un] * 2, lw=1, c="darkgrey", zorder=1, label="x==y", marker="o"
            )

    ax.set_xticklabels([])
    if xticklabels_on:
        for i, cat in enumerate(cats):
            ax.text(i * 1 + 0.5, -0.1, cat, ha="center", rotation=45)

    ax.set_ylim(bottom=-0.01, top=1.01)

    xmin = -0.02 * n_cat
    xmax = n_cat + 0.02 * n_cat
    ax.set_xticks(np.linspace(xmin, xmax, 4))
    ax.set_xlim(left=xmin, right=xmax)


def plot_continuous_cdf(ax, unc, x, y, xticklabels_on, ccdf):
    """plot a continuous cdf on ax for data,grouping data by the groups
    specified in y.

    Parameters
    ----------
    ax : matplotlib axes
    unc : str
    x  : ndarray
    y : ndarray
    xticklabels_on : bool
    ccdf : bool

    """

    for i in range(np.max(y) + 1):
        data_i = x[y == i]
        sorted_data = np.sort(data_i)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
        if ccdf:
            yvals = 1 - yvals
        ax.plot(sorted_data, yvals, color=cp[i + 1], label=f"{i}")

    x0 = x.min()
    x1 = x.max()

    sorted_data = x.sort_values()
    yvals = np.arange(len(x)) / float(len(x))
    if ccdf:
        yvals = 1 - yvals

    ax.plot(sorted_data, yvals, c="darkgrey", lw=1)

    ax.set_xlim(left=x0, right=x1)
    xticklocs = np.linspace(x0, x1, 4)
    ax.set_xticks(xticklocs)
    if xticklabels_on:
        ax.set_xticklabels([f"{entry:.2g}" for entry in xticklocs])
    else:
        ax.set_xticklabels([])


def plot_individual_cdf(
    ax,
    unc,
    x,
    y,
    discrete=False,
    legend=False,
    xticklabels_on=False,
    yticklabels_on=False,
    ccdf=False,
):
    """plot cdf for x conditional on y

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

    """

    if discrete:
        plot_discrete_cdf(ax, unc, x, y, xticklabels_on, ccdf)
    else:
        plot_continuous_cdf(ax, unc, x, y, xticklabels_on, ccdf)

    if legend:
        proxies, labels = build_legend(x, y)
        ax.legend(proxies, labels, loc="best")

    yticklocs = np.linspace(0, 1, 4)
    ax.set_yticks(yticklocs)

    if yticklabels_on:
        ax.set_yticklabels(["$0$", "$\\frac{1}{3}$", "$\\frac{2}{3}$", "$1$"])
    else:
        ax.set_yticklabels([])

    if xticklabels_on:
        ax.set_xlabel(str(unc))
    else:
        x0, x1 = ax.get_xlim()
        ax.text(x0 + 0.01 * x1, 1, str(unc), va="top", ha="left")


def plot_cdfs(x, y, ccdf=False):
    """plot cumulative density functions for each column in x, based on
    the  classification specified in y.

    Parameters
    ----------
    x : DataFrame
        the experiments to use in the cdfs
    y : ndaray
        the categorization for the data
    ccdf : bool, optional
           if true, plot a complementary cdf instead of a normal cdf.


    Returns
    -------
    a matplotlib Figure instance

    """
    x = x.copy()

    try:
        x = x.drop("scenario_id", axis=1)
    except KeyError:
        pass

    for entry in ["model", "policy"]:
        if x.loc[:, entry].unique().shape != (1,):
            continue
        try:
            x = x.drop(entry, axis=1)
        except KeyError:
            pass

    uncs = x.columns.tolist()
    cp = sns.color_palette()

    n_col = 4
    n_row = math.ceil(len(uncs) / n_col)
    size = 3
    aspect = 1
    figsize = n_col * size * aspect, n_row * size
    fig, axes = plt.subplots(n_row, n_col, figsize=figsize, squeeze=False)

    for i, unc in enumerate(uncs):
        discrete = False

        i_col = i % n_col
        i_row = i // n_col
        ax = axes[i_row, i_col]

        data = x[unc]
        if data.dtype.name == "category":
            discrete = True
        plot_individual_cdf(ax, unc, data, y, discrete, ccdf=ccdf)

    # last row might contain empty axis,
    # let's make them disappear
    for j_col in range(i_col + 1, n_col):
        ax = axes[i_row, j_col]
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_yticks([])

        sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)

    proxies, labels = build_legend(x, y)

    fig.legend(proxies, labels, loc="upper center")

    return fig
