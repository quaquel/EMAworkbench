'''
Created on Sep 13, 2013

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
import matplotlib.pyplot as plt
import numpy as np
import os

from analysis.b_and_w_plotting import set_fig_to_bw, HATCHING, GREYSCALE
from analysis.plotting_util import make_legend, PATCH

# figure_dir = r'./tmp_figs'
# d = os.path.dirname(figure_dir)
# if not os.path.exists(d):
#     os.makedirs(d)



def test_scatter():
    x = np.random.rand(5)
    y = np.random.rand(5)
    color = 'r'
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x,y, c=color, s=80, marker='x')
    set_fig_to_bw(fig)
    
    plt.draw()

def test_fill_between():
    x = np.linspace(0, 1)
    y1 = np.sin(4 * np.pi * x) * np.exp(-5 * x)
    y2 = np.cos(4 * np.pi * x) * np.exp(-5 * x)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.fill_between(x, y1, y2, c='b')
    set_fig_to_bw(fig, style=HATCHING)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.fill_between(x, y1, y2,label='test',c='b')
    set_fig_to_bw(fig, style=GREYSCALE)
    
    plt.draw()

def test_fig_legend():

    fig = plt.figure()
    ax = fig.add_subplot(111)
    make_legend(['a','b','c'], fig, legend_type=PATCH)
    set_fig_to_bw(fig, style=GREYSCALE)
    plt.draw()
    

if __name__ == "__main__":
#     test_scatter()
    test_fill_between()
#     test_fig_legend()
    plt.show()