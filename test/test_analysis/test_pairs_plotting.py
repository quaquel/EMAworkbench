'''
Created on Mar 13, 2012

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import matplotlib.pyplot as plt

from ema_workbench.analysis.pairs_plotting import (pairs_density, pairs_lines,
                                                   pairs_scatter)
from .. import utilities

def test_pairs_lines():
    results = utilities.load_eng_trans_data() 
    pairs_lines(results)
    
    pairs_lines(results, group_by='policy')
    plt.draw()
    plt.close('all')

def test_pairs_density():
    results =  utilities.load_eng_trans_data() 
    pairs_density(results)
    pairs_density(results, colormap='binary')

    pairs_density(results, group_by='policy', grouping_specifiers=['no policy'])
    plt.draw()
    plt.close('all')

def test_pairs_scatter():
    results = utilities.load_eng_trans_data()
    
    pairs_scatter(results)
    
    pairs_scatter(results, group_by='policy',
                  grouping_specifiers='basic policy', legend=False)
    
    pairs_scatter(results, group_by='policy', 
                  grouping_specifiers=['no policy', 'adaptive policy'])
    plt.draw()
    plt.close('all')


if __name__ == '__main__':
    test_pairs_lines()
    test_pairs_density()
    test_pairs_scatter()