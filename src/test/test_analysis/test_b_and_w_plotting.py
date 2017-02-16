'''
Created on Sep 13, 2013

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import matplotlib.pyplot as plt
import numpy as np

from ema_workbench.analysis.b_and_w_plotting import (set_fig_to_bw, HATCHING, 
                                                     GREYSCALE)


def test_scatter():
    x = np.random.rand(5)
    y = np.random.rand(5)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x,y, s=80, c='r')
    set_fig_to_bw(fig)
    
    plt.draw()

def test_fill_between():
    x = np.linspace(0, 1)
    y1 = np.sin(4 * np.pi * x) * np.exp(-5 * x)
    y2 = np.cos(4 * np.pi * x) * np.exp(-5 * x)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.fill_between(x, y1, y2)
    set_fig_to_bw(fig, style=HATCHING)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.fill_between(x, y1, y2,label='test')
    set_fig_to_bw(fig, style=GREYSCALE)
    
    plt.draw()

def test_fig_legend():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    x = np.random.rand(5)
    y = np.random.rand(5)
    ax.scatter(x,y, label='a', c='b')
    
    x = np.random.rand(5)
    y = np.random.rand(5)
    ax.scatter(x,y, label='b', c='r')
    
    x = np.random.rand(5)
    y = np.random.rand(5)
    ax.scatter(x,y, label='c', c='k')
    ax.legend()
    
    set_fig_to_bw(fig, style=GREYSCALE)
    plt.draw()
    

if __name__ == "__main__":
    test_scatter()
    test_fill_between()
    test_fig_legend()
#     plt.show()