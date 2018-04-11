'''

This module offers basic functionality for converting a matplotlib figure
to black and white. The provided functionality is largely determined by
what is needed for the workbench. 

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import itertools
import math
import six

import matplotlib as mpl
import numpy as np
from matplotlib.collections import PolyCollection, PathCollection
from matplotlib.colors import ColorConverter

from ema_workbench.util import ema_logging
from ..util import EMAError


# Created on 18 sep. 2012
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['set_fig_to_bw']

bw_mapping = [{'marker': None, 'dash': (None,None), 'fill':'0.1', 'hatch':'/'},
              {'marker': None, 'dash': [5,5], 'fill':'0.25', 'hatch':'\\'},
              {'marker': None, 'dash': [5,3,1,3], 'fill':'0.4', 'hatch':'|'},
              {'marker': None, 'dash': [1,3], 'fill':'0.55', 'hatch':'-'},
              {'marker': None, 'dash': [5,2,5,2,5,10], 'fill':'0.7', 'hatch':'o'},
              {'marker': None, 'dash': [5,3,1,2,1,10], 'fill':'0.85', 'hatch':'O'},
              {'marker': 'o', 'dash': (None,None), 'fill':'0.1', 'hatch':'.'}]

MARKERSIZE = 3
HATCHING = 'hatching'
GREYSCALE = 'grey_scale'

def _identify_colors(fig):
    '''Identify the various colors that are used in the figure and
    return as a set
    
    '''
    
    color_converter = ColorConverter()
    all_colors = set()
    
    for ax in fig.axes:
        for line in ax.get_lines():
            orig_color = line.get_color()
            all_colors.add(orig_color)
        
        for patch in ax.patches:
            rgb_orig = color_converter.to_rgb(patch._facecolor)
            all_colors.add(rgb_orig)
                
        for collection in ax.collections:
            for color in collection.get_facecolor():
                rgb_orig = color_converter.to_rgb(color)
                all_colors.add(rgb_orig)
    
    return all_colors

def set_ax_lines_bw(ax, colormap, line_style='continuous'):
    """
    Take each Line2D in the axes, ax, and convert the line style to be 
    suitable for black and white viewing.
    
    Derived from and expanded for use in the EMA workbench from:
    http://stackoverflow.com/questions/7358118/matplotlib-black-white-colormap-with-dashes-dots-etc
    
    Parameters
    ----------
    ax : axes
         The ax of which the lines needs to be transformed to B&W. Lines are 
         transformed to different line styles.
    colormap : dict 
    line_style: str
                linestyle to use for converting, can be continuous, black
                or None
                    
    """

    for line in ax.get_lines():
        orig_color = line.get_color()
        try:
            mapping = colormap[orig_color]
        except:
            ema_logging.warning('no mapping specified for color: {}'.format(orig_color))
        else:
            if line_style == 'continuous':
                line.set_color('black')
                alpha = 1/(math.log(len(ax.get_lines()))+1)
                line.set_alpha(alpha)
            elif line_style == 'black':
                line.set_color('black')
                line.set_alpha(0.5)
            else:
                line.set_color('black')
                line.set_dashes(mapping['dash'])
                line.set_marker(mapping['marker'])
                line.set_alpha(1)
                line.set_markersize(MARKERSIZE)


def set_ax_patches_bw(ax, colormap):
    """
    Take each patch in the axes, ax, and convert the face color to be 
    suitable for black and white viewing.
    
    Parameters
    ----------
    ax : axes
         The ax of which the patches needs to be transformed to B&W.
    colormap : dict
               mapping of color to B&W rendering
    
    """    
    
    color_converter = ColorConverter()
    
    for patch in ax.patches:
        rgb_orig = color_converter.to_rgb(patch._facecolor)
        new_color = color_converter.to_rgba(colormap[rgb_orig]['fill'])
        
        patch._facecolor = new_color


def set_ax_collections_to_bw(ax, style, colormap):
    """
    Take each polycollection in the axes, ax, and convert the face color to be 
    suitable for black and white viewing.

    Parameters
    ----------
    ax : axes
        The ax of which the polycollection needs to be transformed to B&W.
    style : {HATCHING, GREYSCALE}
    colormap : dict
               mapping of color to B&W rendering
    
    """
    for collection in ax.collections:
        collection_type = type(collection).__name__
        try:
            converter_func = _collection_converter[collection_type]
        except KeyError:
            raise EMAError("converter for {} not implemented".format(collection_type))
        else:
            converter_func(collection, ax, style, colormap)
        
 
def _set_ax_polycollection_to_bw(collection, ax, style, colormap):
    '''helper function for converting a polycollection to black and white
    
    Parameters
    ----------
    collection : polycollection
    ax : axes
    style : {GREYSCALE, HATCHING}
    colormap : dict
               mapping of color to B&W rendering

    
    '''

    if style==GREYSCALE:
        color_converter = ColorConverter()
        for polycollection in ax.collections:
            orig_color = polycollection._original_facecolor

            try:
                mapping = colormap[orig_color]
            except:
                ema_logging.warning('no mapping specified for color: {}'.format(orig_color))
            else:
                new_color = color_converter.to_rgba(mapping['fill'])
                new_color = np.asarray([new_color])
                polycollection.update({'facecolors' : new_color}) 
                polycollection.update({'edgecolors' : new_color})
    elif style==HATCHING:
        orig_color = collection._original_facecolor
        
        try:
            mapping = colormap[orig_color]
        except:
            ema_logging.warning('no mapping specified for color: {}'.format(orig_color))
        else:
            collection.update({'facecolors' : 'none'}) 
            collection.update({'edgecolors' : 'white'}) 
            collection.update({'alpha':1})
        
            for path in collection.get_paths():
                p1 = mpl.patches.PathPatch(path, fc="none", 
                                           hatch=colormap[orig_color]['hatch'])
                ax.add_patch(p1)
                p1.set_zorder(collection.get_zorder()-0.1)


def _set_ax_pathcollection_to_bw(collection, ax, style, colormap):
    '''helper function for converting a pathcollection to black and white
    
    Parameters
    ----------
    collection : pathcollection
    ax : axes
    style : {GREYSCALE, HATCHING}
    colormap : dict
               mapping of color to B&W rendering
    
    '''
    color_converter = ColorConverter()
    colors = {}
    for key, value in color_converter.colors.items():
        colors[value] = key    

    rgb_orig = collection._original_facecolor

    if isinstance(rgb_orig, six.string_types):
        rgb_orig = [rgb_orig]
    rgb_orig = [color_converter.to_rgb(row) for row in rgb_orig]
    
    new_color = [color_converter.to_rgba(colormap[entry]['fill']) for entry 
                 in rgb_orig]
    new_color = np.asarray(new_color)
    
    collection.update({'facecolors' : new_color}) 
    collection.update({'edgecolors' : new_color}) 


_collection_converter = {PathCollection.__name__: _set_ax_pathcollection_to_bw,  # @UndefinedVariable
                         PolyCollection.__name__: _set_ax_polycollection_to_bw}  # @UndefinedVariable


def set_legend_to_bw(leg, style, colormap, line_style='continuous'):
    """
    Takes the figure legend and converts it to black and white. Note that
    it currently only converts lines to black and white, other artist 
    intances are currently not being supported, and might cause errors or
    other unexpected behavior.
    
    Parameters
    ----------
    leg : legend
    style : {GREYSCALE, HATCHING}
    colormap : dict
               mapping of color to B&W rendering
    line_style: str
                linestyle to use for converting, can be continuous, black
                or None
                
    """
    color_converter = ColorConverter()

    if leg:
        if isinstance(leg, list):
            leg = leg[0]
    
        for element in leg.legendHandles:
            if isinstance(element, mpl.collections.PathCollection):
                rgb_orig = color_converter.to_rgb(element._facecolors[0])
                new_color = color_converter.to_rgba(colormap[rgb_orig]['fill'])
                element._facecolors = np.array((new_color,))
            elif isinstance(element, mpl.patches.Rectangle):
                rgb_orig = color_converter.to_rgb(element._facecolor)
                
                if style==HATCHING:
                    element.update({'alpha':1})
                    element.update({'facecolor':'none'})
                    element.update({'edgecolor':'black'})
                    element.update({'hatch':colormap[rgb_orig]['hatch']})
                elif style==GREYSCALE:
                    ema_logging.info(colormap.keys())
                    element.update({'facecolor':colormap[rgb_orig]['fill']})
                    element.update({'edgecolor':colormap[rgb_orig]['fill']})
            else:
                line = element
                orig_color = line.get_color()
                
                line.set_color('black')
                if not line_style=='continuous':
                    line.set_dashes(colormap[orig_color]['dash'])
                    line.set_marker(colormap[orig_color]['marker'])
                    line.set_markersize(MARKERSIZE)


def set_ax_legend_to_bw(ax, style, colormap, line_style='continuous'):
    '''convert axes legend to black and white
    
    Parameters
    ----------
    ax : axes
    style : {GREYSCALE, HATCHING}
    colormap : dict
               mapping of color to B&W rendering
    line_style: str
                linestyle to use for converting, can be continuous, black
                or None
    
    '''
    
    legend = ax.legend_
    set_legend_to_bw(legend, style, colormap, line_style)


def set_fig_to_bw(fig, style=HATCHING, line_style='continuous', all_colors=None):
    """
    TODO it would be nice if for lines you can select either markers, gray 
    scale, or simple black
    
    Take each axes in the figure and transform its content to black and white. 
    Lines are tranformed based on different line styles. Fills such as can 
    be used in `meth`:envelopes are gray-scaled. Heathmaps are also gray-scaled.
    
    
    derived from and expanded for my use from:
    http://stackoverflow.com/questions/7358118/matplotlib-black-white-colormap-with-dashes-dots-etc
    
    Parameters
    ----------
    fig : figure
          the figure to transform to black and white
    style : {HATCHING, GREYSCALE}
            parameter controlling how collections are transformed.
    line_style: str
                linestyle to use for converting, can be continuous, black
                or None  
    
    """
    all_colors = _identify_colors(fig)
    
    if len(all_colors)>len(bw_mapping):
        mapping_cycle = itertools.cycle(bw_mapping)
        ema_logging.warning('more colors used than provided in B&W mapping, cycling over mapping')
    else:
        mapping_cycle = bw_mapping 
    colormap = dict(zip(all_colors, mapping_cycle))
    ema_logging.debug(colormap.keys())
    
    max_shade=0.9
    for i, color in enumerate(colormap.keys()):
        relative_color=max_shade*((i+1)/len(all_colors))
        colormap[color]['fill'] = str(relative_color)
    
    for ax in fig.get_axes():
        set_ax_lines_bw(ax, colormap)
        set_ax_patches_bw(ax, colormap)
        set_ax_collections_to_bw(ax, style, colormap)
        set_ax_legend_to_bw(ax, style, colormap)
        
    set_legend_to_bw(fig.legends, style, colormap)
    
