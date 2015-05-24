'''
Created on May 24, 2015

@author: jhkwakkel
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as recfunctions
import pandas as pd

from expWorkbench import ema_logging
from analysis.plotting_util import COLOR_LIST


def _get_sorted_box_lims(boxes, box_init):
        
        # determine the uncertainties that are being restricted
        # in one or more boxes
        uncs = set()
        for box in boxes:
            us  = _determine_restricted_dims(box, box_init).tolist()
            uncs = uncs.union(us)
        uncs = np.asarray(list(uncs))

        # normalize the range for the first box
        box_lim = boxes[0]
        nbl = _normalize(box_lim, box_init, uncs)
        box_size = nbl[:,1]-nbl[:,0]
        
        # sort the uncertainties based on the normalized size of the 
        # restricted dimensions
        uncs = uncs[np.argsort(box_size)]
        box_lims = [box for box in boxes]
        
        return box_lims, uncs


def _make_box(x):
    '''
    Make a box that encompasses all the data
    
    Parameters
    ----------
    x : structured numpy array
    
    
    '''
    
    box = np.zeros((2, ), x.dtype)
    
    names = recfunctions.get_names(x.dtype)
    
    for name in names:
        dtype = x.dtype.fields.get(name)[0] 
        values = x[name]
        
        if dtype == 'object':
            try:
                values = set(values) - set([np.ma.masked])
                box[name][:] = values
            except TypeError as e:
                ema_logging.warning("{} has unhashable values".format(name))
                raise e
        else:
            box[name][0] = np.min(values, axis=0) 
            box[name][1] = np.max(values, axis=0)    
    return box  


def _normalize(box_lim, box_init, unc):
    
    # normalize the range for the first box
    norm_box_lim = np.zeros((len(unc), box_lim.shape[0]))
    
    for i, u in enumerate(unc):
        dtype = box_lim.dtype.fields[u][0]
        if dtype ==np.dtype(object):
            nu = len(box_lim[u][0])/ len(box_init[u][0])
            nl = 0
        else:
            lower, upper = box_lim[u]
            dif = (box_init[u][1]-box_init[u][0])
            a = 1/dif
            b = -1 * box_init[u][0] / dif
            nl = a * lower + b
            nu = a * upper + b
        norm_box_lim[i, :] = nl, nu
    return norm_box_lim


def _determine_restricted_dims(box_lims, box_init):
    '''
    
    determine which dimensions of the given box are restricted compared 
    to compared to the initial box that contains all the data
    
    Parameters
    ----------
    box_lims : structured numpy array
    
    '''

    logical = _compare(box_init, box_lims)
    u = np.asarray(recfunctions.get_names(box_lims.dtype), 
                   dtype=object)
    dims = u[logical==False]
    return dims


def _compare(a, b):
    '''compare two boxes, for each dimension return True if the
    same and false otherwise'''
    dtypesDesc = a.dtype.descr
    logical = np.ones((len(dtypesDesc,)), dtype=np.bool)
    for i, entry in enumerate(dtypesDesc):
        name = entry[0]
        logical[i] = logical[i] &\
                    (a[name][0] == b[name][0]) &\
                    (a[name][1] == b[name][1])
    return logical


def _setup_figure(uncs):
    '''
    
    helper function for creating the basic layout for the figures that
    show the box lims.
    
    '''
    nr_unc = len(uncs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # create the shaded grey background
    rect = mpl.patches.Rectangle((0, -0.5), 1, nr_unc+1.5,
                                 alpha=0.25,  
                                 facecolor="#C0C0C0",
                                 edgecolor="#C0C0C0")
    ax.add_patch(rect)
    ax.set_xlim(xmin=-0.2, xmax=1.2)
    ax.set_ylim(ymin= -0.5, ymax=nr_unc-0.5)
    ax.yaxis.set_ticks([y for y in range(nr_unc)])
    ax.xaxis.set_ticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(uncs[::-1]) 
    return fig, ax


class OutputFormatterMixin(object):
    
    def boxes_to_dataframe(self):
        boxes = self.boxes
            
        # determine the restricted dimensions
        # print only the restricted dimension
        box_lims, uncs = _get_sorted_box_lims(boxes, _make_box(self.x))
        nr_boxes = len(boxes)
        dtype = float
        index = ["box {}".format(i+1) for i in range(nr_boxes)]
        for value in box_lims[0].dtype.fields.values():
            if value[0] == object:
                dtype = object
                break
                
        columns = pd.MultiIndex.from_product([index,
                                              ['min', 'max',]])
        df_boxes = pd.DataFrame(np.zeros((len(uncs), nr_boxes*2)),
                               index=uncs,
                               dtype=dtype,
                               columns=columns)

        for i, box in enumerate(box_lims):
            for unc in uncs:
                values = box[unc][:]
                values = pd.Series(values, 
                                   index=['min','max'])
                df_boxes.ix[unc][index[i]] = values   
        return df_boxes 
        
    
    def stats_to_dataframe(self):
        pass
    
    def display_boxes(self, together=False):
        box_init = _make_box(self.x)
        box_lims, uncs = _get_sorted_box_lims(self.boxes, box_init)

        # normalize the box lims
        # we don't need to show the last box, for this is the 
        # box_init, which is visualized by a grey area in this
        # plot.
        norm_box_lims =  [_normalize(box_lim, box_init, uncs) for 
                                        box_lim in box_lims[0:-1]]
                        
        if together:
            fig, ax = _setup_figure(uncs)
            
            for i, u in enumerate(uncs):
                # we want to have the most restricted dimension
                # at the top of the figure
                xi = len(uncs) - i - 1
                
                for j, norm_box_lim in enumerate(norm_box_lims):
                    self._plot_unc(box_init, xi, i, j, norm_box_lim,
                                   box_lims[j], u, ax)
           
            plt.tight_layout()
            return fig
        else:
            figs = []
            for j, norm_box_lim in enumerate(norm_box_lims):
                fig, ax = _setup_figure(uncs)
                figs.append(fig)
                for i, u in enumerate(uncs):
                    xi = len(uncs) - i - 1
                    self._plot_unc(xi, i, j, norm_box_lim, box_lims[j], u, ax)
        
                plt.tight_layout()
            return figs
        
    def _plot_unc(self, box_init, xi, i, j, norm_box_lim, box_lim, u, ax):
        '''
        
        Parameters:
        ----------
        xi : int 
             the row at which to plot
        i : int
            the indexo of the uncertainty being plotted
        j : int
            the index of the box being plotted
        u : string
            the uncertainty being plotted:
        ax : axes instance
             the ax on which to plot
        
        '''

        dtype = box_init[u].dtype
            
        y = xi-j*0.1
        
        if dtype == object:
            elements = sorted(list(box_init[u][0]))
            max_value = (len(elements)-1)
            box_lim = box_lim[u][0]
            x = [elements.index(entry) for entry in 
                 box_lim]
            x = [entry/max_value for entry in x]
            y = [y] * len(x)
            
            ax.scatter(x,y,  edgecolor=COLOR_LIST[j],
                       facecolor=COLOR_LIST[j])
            
        else:
            ax.plot(norm_box_lim[i], (y, y),
                    COLOR_LIST[j])
        
        