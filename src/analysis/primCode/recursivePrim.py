'''
Created on 10 okt. 2011

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

This file contains a pythonic implementation of the PRIM algorithm. It is
effectively a reimplementation of the translation of PRIM from R to Python. 
Compared to that translation, this implementation is completely based on 
recursion. Moreover, the paste and peel methods now look analogous. 

planned enhancements:
 - modify peel and paste to handle oridnal and categorical data appropriately
 - add sdtoolkit coverage and support metrics
 

'''
from __future__ import division
import numpy as np
import copy
from scipy.stats.mstats import mquantiles
from operator import itemgetter

from expWorkbench.ema_logging import info, debug

class Prim(object):
    def __init__(self, 
                 x,
                 y,
                 box,
                 box_mass):
        self.x = x
        self.y = y
        self.y_mean = np.mean(y)
        self.box = box
        self.box_mass = box_mass

def perform_prim(x,
                 y,
                 box_init = None,
                 peel_alpha = 0.05,
                 paste_alpha = 0.01,
                 mass_min = 0.05,
                 threshold = None,
                 pasting = False,
                 threshold_type = 1):
    '''
    
    :param x: a 2-d array containing the features (rows are observations) 
    :param y: a 1-d array of class scores (discrete or continuous)
    :param peel_alpha: parameter controlling the peeling stage (default = 0.05). 
    :param paste_alpha: parameter controlling the pasting stage (default = 0.05).
    :param mass_min: minimum mass of a box (default = 0.05). 
    :param threshold: the threshold of the output space that boxes should meet. 
    :param pasting: perform pasting stage (default=True) 
    :param threshold_type: If 1, the boxes should go above the threshold, if -1
                           the boxes should go below the threshold, if 0, the 
                           algorithm looks for both +1 and -1.
    :return: a list of PRIM objects.
    
    '''
    
    if threshold==None:
        threshold = np.mean(y)
   
    k_max = np.ceil(1/mass_min)
    k_max = int(k_max)
    info("k max: %s" %(k_max))
    
    if box_init == None:
        #if no initial box, make initial box
        box_init = np.array([np.min(x, axis=0),np.max(x, axis=0)])
        box_diff = box_init[1, :] - box_init[0, :]
        box_init[0,:] = box_init[0, :] - 10*paste_alpha*box_diff
        box_init[1,:] = box_init[1, :] + 10*paste_alpha*box_diff
    else:
        #else, identify all points in initial box, rest is discarded
        logical =  in_box(x, box_init)
        x = x[logical]
        y = y[logical]


    n = y.shape[0]
    y = y * threshold_type
    boxes = find_boxes(x, y, box_init, 
                       peel_alpha, paste_alpha, mass_min, 
                       np.min(y)-0.1*np.abs(np.min(y)), 
                       pasting, 0, k_max, n)
    
    # adjust for negative hdr  
    for box in boxes:
        box.y = threshold_type*box.y
        box.y_mean = threshold_type*box.y_mean

    # the list of found boxes has the dump box as first element
    # we need to reverse the ordering to get the correct order in which
    # the boxes have been found
    boxes.reverse()
    boxes = prim_hdr(boxes, threshold, threshold_type)
    
    return boxes

def prim_hdr(prims,
             threshold,
             threshold_type):
    '''
    Highest density region for PRIM boxes
    
    prim        list of prim objects
    threshold    
    threshold_type
    
    '''
    
    n = 0
    for entry in prims:
        n += entry.y.shape[0]
    info("number of items in boxes: %s" %n)
  
    boxes = [(entry.y_mean, entry) for entry in prims]
    
    final_list = []
    dump_entries = []
    for entry in boxes:
        if entry[0]*threshold_type >= threshold*threshold_type:
            final_list.append(entry[1])
        else:
            dump_entries.append(entry[1])

    x_temp = None
    for entry in dump_entries: 
        if x_temp == None:
            x_temp = entry.x
            y_temp = entry.y
        else:
            x_temp = np.append(x_temp, entry.x, axis=0) 
            y_temp = np.append(y_temp, entry.y, axis=0)

    dump_box = Prim(x_temp, y_temp, prims[-1].box, 
                        y_temp.shape[0]/n)
        
    final_list.append(dump_box)

    return final_list
    
def find_boxes(x_remaining,
               y_remaining,
               box_init,
               peel_alpha,
               paste_alpha,
               mass_min,
               threshold,
               pasting,
               k, 
               k_max,
               n):
    '''    
     Finds box
    
     Parameters
     x - matrix of explanatory variables
     y - vector of response variable
     box.init - initial box (should cover range of x)
     mass.min - min box mass
     threshold - min box mean
     pasting - TRUE - include pasting step (after peeling)
             - FALSE - dont include pasting
    
     Returns
     List with fields
     x - data still inside box after peeling
     y - corresponding response values
     y.mean - mean of y
     box - box limits
     mass - box mass
    '''
    k+=1
    
    info("%s points remaining" % (y_remaining.shape[0]))
    
    new_box = peel(x_remaining, y_remaining, box_init, peel_alpha, 
                   mass_min, threshold, n)

    info("peeling completed")

    if pasting:
        logical = in_box(x_remaining, new_box, bool=True)
        x_inside = x_remaining[logical]
        y_inside = y_remaining[logical]

        new_box = paste(x_inside, y_inside, x_remaining, y_remaining, 
                           new_box, paste_alpha, mass_min, 
                           threshold, n)
        info("pasting completed")

    
    logical = in_box(x_remaining, new_box, bool=True)
    x_inside = x_remaining[logical]
    y_inside = y_remaining[logical]
    box_mass = y_inside.shape[0]/n

    # update data in light of found box
    x_remaining_temp = x_remaining[logical==False]
    y_remaining_temp = y_remaining[logical==False]

    if (y_remaining_temp.shape[0] != 0) &\
       (k < k_max) &\
       (compare(box_init, new_box)==False):

        # make a primObject
        prim_object = Prim(x_inside, y_inside, new_box, box_mass)
        info("Found box %s: y_mean=%s, mass=%s" % (k, 
                                                   prim_object.y_mean, 
                                                   prim_object.box_mass))
        info("%s points in new box" % (y_inside.shape[0]))
        
        boxes = find_boxes(x_remaining_temp, y_remaining_temp, 
                           box_init, peel_alpha, paste_alpha, mass_min, 
                           threshold, 
                           pasting, k, k_max, n)
        boxes.append(prim_object)
        return boxes
    else:
        info("Bump "+str(k)+" includes all remaining data")
        #make dump box
        box_mass = y_remaining.shape[0]/n
        dump_box = Prim(x_remaining, y_remaining, box_init, box_mass)
        return [dump_box]

def peel(x,
         y,
         box,
         peel_alpha,
         mass_min,
         threshold,
         n):
    ''' Peeling stage of PRIM '''
    mass_old = y.shape[0]/n
   
    #identify all possible peels
    possible_peels = []
    
    for j in range(box.shape[1]):
        possible_peels.append(try_peel(x, y, j, peel_alpha, box, direction='lower'))
        possible_peels.append(try_peel(x, y, j, peel_alpha, box, direction='upper'))
    
    possible_peels.sort(key=itemgetter(0,1), reverse=True)
    box_new = possible_peels[0][-1]
   
    logical = in_box(x, box_new) 
    x_new = x[logical]
    y_new = y[logical]
    mass_new = y_new.shape[0]/n
    y_mean_new =  np.mean(y_new)
    
    if (y_mean_new >= threshold) &\
       (mass_new >= mass_min) &\
       (mass_new < mass_old) &\
       (x_new.shape[0] != 0):
        # if best peel leaves remaining data
        # call peel again with updated box, x, and y
        return peel(x_new, y_new, box_new, peel_alpha, mass_min, threshold, n)
    else:
        #else return received box
        return box

def try_peel(x,y,j,peel_alpha, box,direction):
    '''
    make a test peel box
    
    returns a tuple (mean, volume, box)
    '''
    alpha = 1/3
    beta = 1/3
    
    i=0
    if direction=='upper':
        peel_alpha = 1-peel_alpha
        i=1
    
    box_peel = mquantiles(x[:, j], [peel_alpha], alphap=alpha, betap=beta)[0]
    
    if direction=='lower':
        y_mean_peel = np.mean(y[ x[:, j] >= box_peel])
    if direction=='upper':
        y_mean_peel = np.mean(y[ x[:, j] <= box_peel])
    
    temp_box = copy.deepcopy(box)
    temp_box[i,j] = box_peel

    box_vol = vol_box(temp_box)
    
    return (y_mean_peel, box_vol, temp_box)

def paste(x,
          y,
          x_init,
          y_init,
          box,
          paste_alpha,
          mass_min,
          threshold,
          n):
    '''
     Pasting stage for PRIM
    
    '''
    mass = y.shape[0]/n
    y_mean = np.mean(y)
    
    box_init = np.array([np.min(x_init, axis=0),np.max(x_init, axis=0)])
    
    possible_pastes = []
    for j in range(x.shape[1]):
        possible_pastes.append(try_paste(x_init, y_init, y, j, box, box_init, 
                                    paste_alpha, n,direction='lower'))
        possible_pastes.append(try_paste(x_init, y_init, y, j, box, box_init, 
                                    paste_alpha, n, direction='upper'))

    #break ties by choosing box with largest mass                 
    possible_pastes.sort(key=itemgetter(0,1), reverse=True)
    y_mean_new, mass_new, box_new = possible_pastes[0]
    logical = in_box(x_init, box_new)
    x_new = x_init[logical]
    y_new = y_init[logical]
    
    if (y_mean_new > threshold) & (mass_new >= mass_min) &\
       (y_mean_new >= y_mean) & (mass_new > mass):
        
        return paste(x_new, y_new, x_init, y_init, box_new, paste_alpha,
                     mass_min, threshold,n)
    else:
        return box

def try_paste(x_init,y_init, y, j,
              box,box_init, paste_alpha, n, direction='lower'):
    
    box_diff = (box_init[1,:] - box_init[0,:])[j]
    if direction == 'lower':
        i = 0
        box_diff = -1*box_diff
    if direction == 'upper':
        i = 1
    
    box_paste = copy.deepcopy(box)
    box_paste[i,j] = box[i,j]+box_diff*paste_alpha
    
    logical = in_box(x_init,box_paste,True)
    y_paste = y_init[logical]
    
    if direction == 'lower':
        while (y_paste.shape[0] <= y.shape[0]) &\
              (box_paste[i,j] >= box_init[i,j]):
            box_paste[i,j] = box_paste[i,j] + box_diff*paste_alpha
            logical = in_box(x_init, box_paste,True)
            y_paste = y_init[logical]
    
    if direction == 'upper':
        while (y_paste.shape[0] <= y.shape[0]) &\
              (box_paste[i,j] <= box_init[i,j]):
            box_paste[i,j] = box_paste[i,j] + box_diff*paste_alpha
            logical = in_box(x_init, box_paste,True)
            y_paste = y_init[logical]

    # y means of pasted boxes
    y_mean_paste = np.mean(y_paste)
    
    # mass of pasted boxes
    mass_paste = y_paste.shape[0]/n

    return (y_mean_paste, mass_paste, box_paste)


def vol_box(box):
    '''return volume of box'''
    return np.prod(np.abs(box[1, :]-box[0,:]))

def compare(a, b):
    '''compare two arrays, return true if identical, false otherwise'''
    return np.all(np.equal(a,b))


def in_box(x, box, bool=True):
    '''
     Find points of x that are in a single box
    
     Parameters
     x - data matrix
     box_paste ranges - matrix of min and max values which define a box 
     d - dimension of data
    
     Returns
     Data points which lie within the box
    '''
    
    x_box_ind = np.ones(x.shape[0], dtype=np.bool)

    for i in range(x.shape[1]):
        x_box_ind = x_box_ind & (box[0,i] <= x[:, i] )& (x[:, i] <= box[1, i])
    
    if bool:
        return x_box_ind   
    else:
        return x[x_box_ind] 
