'''
This module implements logistic regression for scenario discovery.


The module draws its inspiration from Quinn et al (2018) 10.1029/2018WR022743
and Lamontagne et al (2019). The implementation here generalizes their work
and embeds it in a more typical scenario discovery workflow with a posteriori
selection of the appropriate number of dimensions to include. It is modeled
as much as possible on the api used for PRIM and CART.


'''
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns

from ..util import get_module_logger
from . import scenario_discovery_util as sdutil
from .prim_util import CurEntry


__all__ = ['Logit']

_logger = get_module_logger(__name__)

# Created on 14 Mar 2019
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


def calculate_covden(fitted_model, x, y, step=0.1):
    '''Helper function for calculating coverage and density across a
    number of levels

    Parameters
    ----------
    fitted_model
    x : DataFrame
    y : numpy Array
    step : float, optional

    '''

    predicted = fitted_model.predict(x.loc[:, fitted_model.params.index])
    coverage = []
    density = []
    thresholds = np.arange(0, 1 + step, step)
    for threshold in thresholds:
        precision, recall = calculate_covden_for_treshold(predicted, y,
                                                          threshold)

        density.append(precision)
        coverage.append(recall)
    return coverage, density, thresholds


def calculate_covden_for_treshold(predicted, y, threshold):
    '''Helper function for calculating coverage and density

    '''

    tp = np.sum(((predicted > threshold) == True) & (y == True))
    fp = np.sum(((predicted > threshold) == True) & (y == False))
    fn = np.sum(((predicted > threshold) == False) & (y == True))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall


def contours(ax, model, xlabel, ylabel, levels):
    '''helper function for plotting contours

    Parameters
    ----------
    ax : axes instance
    xlabel : str
    ylabel : str
    levels : list of floats in interval [0, 1]

    '''

    Xgrid, Ygrid = np.meshgrid(np.arange(-0.1, 1.1, 0.01),
                               np.arange(-0.1, 1.1, 0.01))

    xflatten = Xgrid.flatten()
    yflatten = Ygrid.flatten()

    shape = xflatten.shape[0], len(model.params.index)
    data = pd.DataFrame(np.ones(shape),
                        columns=model.params.index)
    cols = model.params.index.values.tolist()
    cols.remove('Intercept')

    base_data = data.copy()
    base_data.loc[:, cols] = data.loc[:, cols].multiply(0.5)

    X = base_data.copy()
    X[xlabel] = xflatten
    X[ylabel] = yflatten

    z = model.predict(X)
    Zgrid = np.reshape(z.values, Xgrid.shape)

    # rgb = [255*entry for entry in sns.color_palette()[0]]
    # hsl = 244, 80.9, 39
    # rgb = [255*entry for entry in sns.color_palette()[1]]
    # hsl = 28, 100, 52.7

    cmap = sns.diverging_palette(244, 28, s=99.9, l=52.7, n=len(levels) - 1,
                                 as_cmap=True)
    ax.contourf(Xgrid, Ygrid, Zgrid, levels,
                cmap=cmap, zorder=0)


class Logit(object):
    '''Implements an interactive version of logistic regression using
    BIC based forward selection


    Parameters
    ----------
    x : DataFrame
    y : numpy Array
    threshold : float


    Attributes
    ----------
    coverage : float
               coverage of currently selected model
    density : float
               density of currently selected model
    res_dim : int
              number of restricted dimensions of currently selected model
    peeling_trajectory : DataFrame
                         stats for each model in peeling trajectory
    models : list
               list of models associated with each model on the peeling
               trajectory

    '''
    # TODO:: peeling trajectory is a misnomer, requires fix to CurEntry

    coverage = CurEntry('coverage')
    density = CurEntry('density')
    res_dim = CurEntry('res_dim')
    models = []

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value

        for i, model in enumerate(self.models):
            predicted = model.predict(self._normalized.loc[:,
                                                           model.params.index])

            den, cov = calculate_covden_for_treshold(predicted, self.y, value)
            self.peeling_trajectory.loc[i, 'coverage'] = cov
            self.peeling_trajectory.loc[i, 'density'] = den

    def __init__(self, x, y, threshold=0.95):
        self.x = x
        self.y = y
        self.threshold = threshold

        normalized = (x - x.min()) / (x.max() - x.min())
        normalized['Intercept'] = np.ones(np.shape(x)[0])
        self._normalized = normalized
        colums = ['coverage', 'density', 'res_dim', 'id']
        self.peeling_trajectory = pd.DataFrame(columns=colums)

    def run(self):
        '''run logistic regression using forward selection using a Bayesian
        Information Criterion for selecting whether and if so which dimension
        to add

        '''
        remaining = set(self.x.columns)

        selected = []
        current_score, best_new_score = sys.float_info.max, sys.float_info.max
        while remaining and current_score == best_new_score:
            scores_with_candidates = []
            for candidate in remaining:
                data = self._normalized.loc[:,
                                            selected + [candidate, 'Intercept']]
                model = sm.Logit(self.y, data)
                model = model.fit()

                score = model._results.bic

                scores_with_candidates.append((score, candidate, model))

            scores_with_candidates.sort(reverse=True)
            best_new_score, best_candidate, model = scores_with_candidates.pop()

            if current_score > best_new_score:
                remaining.remove(best_candidate)
                selected.append(best_candidate)

                self.update(model, selected)

                current_score = best_new_score

    def update(self, model, selected):
        '''helper function for adding a model to the colection of models and
        update the associated attributes

        Parameters
        ----------
        model : statsmodel fitted logit model
        selected : list of str

        '''

        predicted = model.predict(self._normalized.loc[:,
                                                       selected + ['Intercept']])
        den, cov = calculate_covden_for_treshold(predicted, self.y,
                                                 self.threshold)

        self.models.append(model)
        i = self.peeling_trajectory.shape[0]

        data = {'coverage': cov,
                'density': den,
                'res_dim': len(selected),
                'id': i}
        new_row = pd.DataFrame([data])
        self.peeling_trajectory = self.peeling_trajectory.append(
            new_row, ignore_index=True, sort=True)

    def show_tradeoff(self, cmap=mpl.cm.viridis):  # @UndefinedVariable
        '''Visualize the trade off between coverage and density. Color
        is used to denote the number of restricted dimensions.

        Parameters
        ----------
        cmap : valid matplotlib colormap

        Returns
        -------
        a Figure instance

        '''
        return sdutil.plot_tradeoff(self.peeling_trajectory, cmap=cmap)

    # @UndefinedVariable
    def show_threshold_tradeoff(self, i, cmap=mpl.cm.viridis_r, step=0.1):
        '''Visualize the trade off between coverage and density for a given
        model i across the range of threshold values

        Parameters
        ----------
        i : int
        cmap : valid matplotlib colormap
        step : float, optional

        Returns
        -------
        a Figure instance

        '''
        # TODO:: might it be possible to flip the colorbar?

        fitted_model = self.models[i]
        x = self._normalized.loc[:, fitted_model.params.index.values]
        coverage, density, thresholds = calculate_covden(fitted_model, x,
                                                         self.y, step=step)

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        ncolors = cmap.N
        norm = mpl.colors.BoundaryNorm(thresholds, ncolors)

        p = ax.scatter(coverage,
                       density,
                       c=thresholds,
                       norm=norm,
                       cmap=cmap)
        ax.set_ylabel('density')
        ax.set_xlabel('coverage')
        ax.set_ylim(bottom=0, top=1.2)
        ax.set_xlim(left=0, right=1.2)

        ticklocs = thresholds
        cb = fig.colorbar(p, spacing='uniform', ticks=ticklocs,
                          drawedges=True)
        cb.set_label("thresholds")

        return fig

    def inspect(self, i, step=0.1):
        '''Inspect one of the models by showing the threshold tradeoff
        and summary2

        Parameters
        ----------
        i : int
        step : float between [0, 1]

        '''

        model = self.models[i]
        x = self._normalized.loc[:, model.params.index.values]
        coverage, density, thresholds = calculate_covden(model, x,
                                                         self.y, step=step)
        data = pd.DataFrame({'coverage': coverage,
                             'density': density,
                             'thresholds': thresholds})
        print(data)
        print()

        print(model.summary2())

    def plot_pairwise_scatter(self, i, threshold=0.95):
        '''plot pairwise scatter plot of data points, with contours as
        background


        Parameters
        ----------
        i : int
        threshold : float

        Returns
        -------
        Figure instance


        The lower triangle background is a binary contour based on the
        specified threshold. All axis not shown are set to a default value
        in the middle of their range

        The upper triangle shows a contour map with the conditional
        probability, again setting all non shown dimensions to a default value
        in the middle of their range.

        '''
        model = self.models[i]

        columns = model.params.index.values.tolist()
        columns.remove('Intercept')
        x = self._normalized[columns]
        data = x.copy()

        # TODO:: have option to change
        # diag to CDF, gives you effectively the
        # regional sensitivity analysis results

        data['y'] = self.y  # for testing
        grid = sns.PairGrid(data=data, hue='y', vars=columns)
        grid.map_lower(plt.scatter, s=5)
        grid.map_diag(sns.kdeplot, shade=True)
        grid.add_legend()

        contour_levels = np.arange(0, 1.05, 0.05)
        for i, j in zip(*np.triu_indices_from(grid.axes, 1)):
            ax = grid.axes[i, j]
            ylabel = columns[i]
            xlabel = columns[j]
            contours(ax, model, xlabel, ylabel, contour_levels)

        levels = [0, threshold, 1]
        for i, j in zip(*np.tril_indices_from(grid.axes, -1)):
            ax = grid.axes[i, j]
            ylabel = columns[i]
            xlabel = columns[j]
            contours(ax, model, xlabel, ylabel, levels)

        fig = plt.gcf()
        return fig
