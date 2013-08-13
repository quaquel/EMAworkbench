'''
Created on 18 sep. 2012

@author: localadmin
'''
import numpy as np
from matplotlib.colors import ColorConverter
import matplotlib as mpl
from types import ListType

__all__ = ['set_fig_to_bw']

COLORMAP = {
    'b': {'marker': None, 'dash': (None,None), 'fill':'0.1', 'hatch':'/'},
    'g': {'marker': None, 'dash': [5,5], 'fill':'0.25', 'hatch':'\\'},
    'r': {'marker': None, 'dash': [5,3,1,3], 'fill':'0.4', 'hatch':'|'},
    'c': {'marker': None, 'dash': [1,3], 'fill':'0.55', 'hatch':'-'},
    'm': {'marker': None, 'dash': [5,2,5,2,5,10], 'fill':'0.7', 'hatch':'o'},
    'y': {'marker': None, 'dash': [5,3,1,2,1,10], 'fill':'0.85', 'hatch':'O'},
    'k': {'marker': 'o', 'dash': (None,None), 'fill':'0.1', 'hatch':'.'} 
        }

MARKERSIZE = 3

def set_ax_lines_bw(ax):
    """
    Take each Line2D in the axes, ax, and convert the line style to be 
    suitable for black and white viewing.
    
    Derived from and expanded for use in the EMA workbench from:
    http://stackoverflow.com/questions/7358118/matplotlib-black-white-colormap-with-dashes-dots-etc
    
    :param ax: The ax of which the lines needs to be transformed to B&W. 
               Lines are transformed to different line styles. 
    
    """

    for line in ax.get_lines():
        orig_color = line.get_color()
        if orig_color in COLORMAP.keys():
            line.set_color('black')
            line.set_dashes(COLORMAP[orig_color]['dash'])
            line.set_marker(COLORMAP[orig_color]['marker'])
            line.set_alpha(1)
            line.set_markersize(MARKERSIZE)

def set_ax_patches_bw(ax):
    """
    Take each patch in the axes, ax, and convert the face color to be 
    suitable for black and white viewing.
    
    :param ax: The ax of which the patches needs to be transformed to B&W.
    
    """    
    
    color_converter = ColorConverter()
    colors = {}
    for key, value in color_converter.colors.items():
        colors[value] = key
    
    for patch in ax.patches:
        rgb_orig = color_converter.to_rgb(patch._facecolor)
        origColor = colors[rgb_orig]
        new_color = color_converter.to_rgba(COLORMAP[origColor]['fill'])
        
        patch._facecolor = new_color

def set_ax_polycollections_to_bw(ax):
    """
    Take each polycollection in the axes, ax, and convert the face color to be 
    suitable for black and white viewing.
    
    :param ax: The ax of which the polycollection needs to be transformed to 
               B&W.
    
    TODO:: this function has a misleading name for it transforms all collection 
    types. We need to make a decision on whether we can transform collections 
    in general or parcel it out to separate functions for each collection 
    class.
    
    """        
    
#    color_converter = ColorConverter()
#    colors = {}
#    for key, value in color_converter.colors.items():
#        colors[value] = key    
#    for polycollection in ax.collections:
#        rgb_orig = polycollection._facecolors_original
#        if rgb_orig in COLORMAP.keys():
#            new_color = color_converter.to_rgba(COLORMAP[rgb_orig]['fill'])
#            a = np.asarray([new_color])
#            polycollection.update({'facecolors' : a}) 
#            polycollection.update({'edgecolors' : 'black'}) 
    for polycollection in ax.collections:
        rgb_orig = polycollection._facecolors_original

        
        polycollection.update({'facecolors' : 'none'}) 
        polycollection.update({'edgecolors' : 'white'}) 
        polycollection.update({'alpha':1})
        
        for path in polycollection.get_paths():
            p1 = mpl.patches.PathPatch(path, fc="none", 
                                       hatch=COLORMAP[rgb_orig]['hatch'])
            ax.add_patch(p1)
            p1.set_zorder(polycollection.get_zorder()-0.1)


def set_legend_to_bw(leg):
    """
    Takes the figure legend and converts it to black and white. Note that
    it currently only converts lines to black and white, other artist 
    intances are currently not being supported, and might cause errors or
    other unexpected behavior.
    
    :param fig: The figure which needs to be transformed to B&W.
    
    """
    color_converter = ColorConverter()
    colors = {}
    for key, value in color_converter.colors.items():
        colors[value] = key
    
    if leg:
        if type(leg) == ListType:
            leg = leg[0]
    
        for element in leg.legendHandles:
            if isinstance(element, mpl.collections.PathCollection):
                rgb_orig = color_converter.to_rgb(element._facecolors[0])
                origColor = colors[rgb_orig]
                new_color = color_converter.to_rgba(COLORMAP[origColor]['fill'])
                element._facecolors = np.array((new_color,))
            elif isinstance(element, mpl.patches.Rectangle):
                rgb_orig = color_converter.to_rgb(element._facecolor)
                c = colors[rgb_orig]
                element.update({'alpha':1})
                element.update({'facecolor':'none'})
                element.update({'edgecolor':'black'})
                element.update({'hatch':COLORMAP[c]['hatch']})
            else:
                line = element
                origColor = line.get_color()
                line.set_color('black')
                line.set_dashes(COLORMAP[origColor]['dash'])
                line.set_marker(COLORMAP[origColor]['marker'])
                line.set_markersize(MARKERSIZE)

def set_ax_legend_to_bw(ax):
    legend = ax.legend_
    set_legend_to_bw(legend)

def set_fig_to_bw(fig):
    """
    TODO it would be nice if for lines you can select either markers, gray 
    scale, or simple black
    
    Take each axes in the figure and transform its content to black and white. 
    Lines are tranformed based on different line styles. Fills such as can 
    be used in `meth`:envelopes are gray-scaled. Heathmaps are also gray-scaled.
    
    
    derived from and expanded for my use from:
    http://stackoverflow.com/questions/7358118/matplotlib-black-white-colormap-with-dashes-dots-etc
    
    """
    for ax in fig.get_axes():
        set_ax_lines_bw(ax)
        set_ax_patches_bw(ax)
        set_ax_polycollections_to_bw(ax)
        set_ax_legend_to_bw(ax)
        
    set_legend_to_bw(fig.legends)
    
