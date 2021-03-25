"""
Created on Mar 13, 2012

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
"""
import matplotlib.pyplot as plt

from ema_workbench.analysis.pairs_plotting import (pairs_density, pairs_lines,
                                                   pairs_scatter)
from test import utilities


def test_pairs_lines():
    experiments, outcomes = utilities.load_eng_trans_data()
    pairs_lines(experiments, outcomes)

    pairs_lines(experiments, outcomes, group_by='policy')
    plt.draw()
    plt.close('all')


def test_pairs_density():
    experiments, outcomes = utilities.load_eng_trans_data()
    pairs_density(experiments, outcomes)
    plt.draw()

    pairs_density(experiments, outcomes, colormap='binary')
    plt.draw()

    pairs_density(experiments, outcomes, group_by='policy',
                  grouping_specifiers=['no policy'])
    plt.draw()
    plt.close('all')


def test_pairs_scatter():
    experiments, outcomes = utilities.load_eng_trans_data()

    pairs_scatter(experiments, outcomes)

    pairs_scatter(experiments, outcomes, group_by='policy',
                  grouping_specifiers='basic policy', legend=False)

    pairs_scatter(experiments, outcomes, group_by='policy',
                  grouping_specifiers=['no policy', 'adaptive policy'])
    plt.draw()
    plt.close('all')


if __name__ == '__main__':
    test_pairs_lines()
    test_pairs_density()
    test_pairs_scatter()
