'''
Created on Feb 28, 2012

@author: jhkwakkel
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def graph_pop_heatmap_raw(all, colormap="jet"):
    figure = plt.figure()
    ax = figure.add_subplot(111)
    cax = ax.imshow(all.T, aspect="auto", 
                 interpolation="gaussian", 
                 cmap=mpl.cm.__dict__[colormap])
    ax.set_title("Plot of pop. raw scores along the generations")
    ax.set_ylabel('Population')
    ax.set_xlabel('Generations')
    ax.grid(True)
    figure.colorbar(cax)
    return figure

def graph_errorbars_raw(all):
    x = []
    y = []
    
    yerr = np.zeros((all["rawMax"].shape[0], 2))

    a =  all["rawMax"] - all["rawAve"]
    yerr[:,0] = a[:, 0]
    yerr_max = list(a[:, 0])
    
    a = all["rawMax"] - all["rawAve"]
    yerr[:,1] = a[:, 0]
    yerr_min = list(a[:, 0])

    
    y =  all["rawAve"][:,0]
    x = np.arange(0, y.shape[0])
    
    figure = plt.figure()
    ax = figure.add_subplot(111)
   
    ax.errorbar(x, y, [yerr_min, yerr_max], ecolor="g")
    ax.set_xlabel('Generation (#)')
    ax.set_ylabel('Raw score Min/Avg/Max')
    ax.set_title("Plot of evolution identified by raw scores")
    ax.grid(True)
    return figure