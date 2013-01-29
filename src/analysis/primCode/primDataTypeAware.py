'''
Created on 10 okt. 2011

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

This file contains a pythonic implementation of the PRIM algorithm. It is
effectively a reimplementation of the translation of PRIM from R to Python. 
Compared to that translation, this implementation is completely based on 
recursion. Moreover, the paste and peel methods now look analogous. 

'''
from __future__ import division
import numpy as np
import copy
from scipy.stats.mstats import mquantiles
from operator import itemgetter

from expWorkbench.ema_logging import info
from types import StringType, FloatType, IntType

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
                 paste_alpha = 0.05,
                 mass_min = 0.05,
                 threshold = None,
                 pasting = False,
                 threshold_type = 1,
                 cases_of_interest = None,
                 obj_func = None):
    if threshold==None:
        threshold = np.mean(y)
   
    k_max = np.ceil(1/mass_min)
    k_max = int(k_max)
    info("max number of boxes: %s" %(k_max))
    
    if box_init == None:
        box_init = make_box(x)
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
                       pasting, 0, k_max, n, cases_of_interest, obj_func)
    
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

def perform_prim_specific(x,
                 y,
                 box_init = None,
                 peel_alpha = 0.05,
                 paste_alpha = 0.05,
                 mass_min = 0.05,
                 threshold = None,
                 pasting = False,
                 threshold_type = 1,
                 cases_of_interest = None,
                 obj_func = None):
    if threshold==None:
        threshold = np.mean(y)
   
    k_max = np.ceil(1/mass_min)
    k_max = int(k_max)
    info("max number of boxes: %s" %(k_max))
    
    if box_init == None:
        box_init = make_box(x)
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
                       pasting, 0, k_max, n, cases_of_interest, obj_func)
    
    # adjust for negative hdr  
    exps = []
    for box in boxes:
        box.y = threshold_type*box.y
        box.y_mean = threshold_type*box.y_mean
        exps.append(box.x)
    # the list of found boxes has the dump box as first element
    # we need to reverse the ordering to get the correct order in which
    # the boxes have been found
    boxes.reverse()
    exps.reverse()
    boxes = prim_hdr(boxes, threshold, threshold_type)
    
    return boxes, exps

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


    dump_box = Prim(x_temp, y_temp, make_box(x_temp), 
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
               n,
               cases_of_interest,
               obj_func):
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
                   mass_min, threshold, n, obj_func)

    info("peeling completed")

    if pasting:
        logical = in_box(x_remaining, new_box)
        x_inside = x_remaining[logical]
        y_inside = y_remaining[logical]

        new_box = paste(x_inside, y_inside, x_remaining, y_remaining, 
                           box_init, new_box, paste_alpha, mass_min, 
                           threshold, n, obj_func)
        info("pasting completed")

    
    logical = in_box(x_remaining, new_box)
    x_inside = x_remaining[logical]
    y_inside = y_remaining[logical]
    box_mass = y_inside.shape[0]/n

    # update data in light of found box
    x_remaining_temp = x_remaining[logical==False]
    y_remaining_temp = y_remaining[logical==False]

    if (y_remaining_temp.shape[0] != 0) &\
       (k < k_max) &\
       (equal(box_init, new_box)==False):

#        #some debugging stuff
#
#        name = 'total productivity growth main switch'
#        print "new box"
#        print new_box[name][0]
#        print "data"
#        
#        for entry in set(x_remaining_temp[name]):
#            
#            nr =  x_remaining_temp[x_remaining_temp[name]==entry].shape[0]
#            print "%s\t%s" %(entry, nr)
#        
#        #end of debugging stuff

        # make a primObject
        prim_object = Prim(x_inside, y_inside, new_box, box_mass)
        coverage = (n * prim_object.y_mean * prim_object.box_mass)/cases_of_interest
        info("Found box %s: y_mean=%s, mass=%s, coverage=%s" % (k, 
                                                                abs(prim_object.y_mean), #because of threshold_type == -1, y can become negative
                                                                prim_object.box_mass,
                                                                abs(coverage)))
        info("%s points in new box" % (y_inside.shape[0]))
        box_init = make_box(x_remaining)
        boxes = find_boxes(x_remaining_temp, y_remaining_temp, 
                           box_init, peel_alpha, paste_alpha, mass_min, 
                           threshold, 
                           pasting, k, k_max, n, cases_of_interest, obj_func)
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
         n, 
         obj_func):
    ''' Peeling stage of PRIM '''
    mass_old = y.shape[0]/n
   
    #identify all possible peels
    possible_peels = []
    
    for entry in x.dtype.descr:
            name = entry[0]
            value = x.dtype.fields.get(entry[0])[0] 
            if value == 'object':
                peels = categoryPeel(x, y, n, name, box, obj_func)
                [possible_peels.append(p) for p in peels]
            elif value == 'int':
                possible_peels.append(discretePeel(x, y, n, name, peel_alpha, box, 'lower', obj_func))
                possible_peels.append(discretePeel(x, y, n, name, peel_alpha, box, 'upper', obj_func))
            else:
                possible_peels.append(realPeel(x, y, n, name, peel_alpha, box, 'lower', obj_func))
                possible_peels.append(realPeel(x, y, n, name, peel_alpha, box, 'upper', obj_func))
    
    possible_peels.sort(key=itemgetter(0,1), reverse=True)
    obj, box_vol, box_new, logical = possible_peels[0]
    
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
        return peel(x_new, y_new, box_new, peel_alpha, mass_min, threshold, n, obj_func)
    else:
        #else return received box
        return box

def categoryPeel(x,y, n, name,box, obj_func):
    entries = box[name][0]

    
    if len(entries) > 1:
        peels = []
        for entry in entries:
            temp_box = np.copy(box)
            peel = copy.deepcopy(entries)
            peel.discard(entry)
            temp_box[name][:] = peel
            
            if type(list(entries)[0]) not in (StringType, FloatType, IntType):
                bools = []                
                for element in list(x[name]):
                    if element != entry:
                        bools.append(True)
                    else:
                        bools.append(False)
                logical = np.asarray(bools, dtype=bool)
            else:
                logical = x[name] != entry

            if y[logical].shape[0] != 0:
                obj = obj_func(y, y[logical])
            else:
                obj = 0
            
            
            box_vol = vol_box(temp_box)
            box_mass = y[logical].shape[0]/n
            box_vol = box_mass
            
            peels.append((obj, box_vol, temp_box, logical))
        return peels
    else:
        # no peels possible, return empty list
        return []


def discretePeel(x, y, n, name, peel_alpha, box,direction, obj_func):
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
    
    x_alpha = mquantiles(x[name], [peel_alpha], alphap=alpha, betap=beta)[0]
    
    x_alpha = int(x_alpha)
    if direction=='lower':
        if x_alpha == box[name][i]:
            logical = (x[name] > box[name][i]) & (x[name] <= box[name][i+1])
            
            if x[logical].shape[0] == 0:
                new_limit = np.min(x[name])
            else:
                new_limit = np.min(x[name][logical])
        else:
            logical = (x[name] >= x_alpha) & (x[name] <= box[name][i+1])
            if x[logical].shape[0] == 0:
                new_limit = np.min(x[name])
            else:
                new_limit = np.min(x[name][logical])
    if direction=='upper':
        if x_alpha == box[name][i]:
            logical = (x[name] < box[name][i]) & (x[name] >= box[name][i-1])
            if x[logical].shape[0] == 0:
                new_limit = np.max(x[name])
            else:
                new_limit = np.max(x[name][logical])
        else:
            logical = (x[name] <= x_alpha) & (x[name] >= box[name][i-1])
            if x[logical].shape[0] == 0:
                new_limit = np.max(x[name])
            else:
                new_limit = np.max(x[name][logical])      
    
    temp_box = np.copy(box)
    temp_box[name][i] = new_limit

    if y[logical].shape[0] != 0:
        obj = obj_func(y,  y[logical])
        box_mass = y[logical].shape[0]/n
        box_vol = box_mass
    else:
        obj = obj_func(y, y)
        box_vol = y.shape[0]/n
       
    return (obj, box_vol, temp_box, logical)
    

def realPeel(x,y,n,name,peel_alpha, box,direction, obj_func):
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

    box_peel = mquantiles(x[name], [peel_alpha], alphap=alpha, betap=beta)[0]
    if direction=='lower':
        logical = x[name] >= box_peel
    if direction=='upper':
        logical = [x[name] <= box_peel]

    obj = obj_func(y,  y[logical])    
    temp_box = np.copy(box)
    temp_box[name][i] = box_peel
    box_mass = y[logical].shape[0]/n
    box_vol = box_mass
#    box_vol = vol_box(temp_box)
    return (obj, box_vol, temp_box, logical)

def paste(x,
          y,
          x_init,
          y_init,
          box_init,
          box,
          paste_alpha,
          mass_min,
          threshold,
          n,
          obj_func):
    '''
     Pasting stage for PRIM
    
    '''
    mass = y.shape[0]/n
    y_mean = np.mean(y)
    
    possible_pastes = []
    for entry in x.dtype.descr:
        name = entry[0]
        value = x.dtype.fields.get(entry[0])[0]
        
        if value == 'object':
            peels = categoryPaste(x_init, y_init, y, name, box,n, obj_func)
            [possible_pastes.append(p) for p in peels]
        elif value == 'int':
            possible_pastes.append(discretePaste(x_init, y_init, y, name, box, box_init, paste_alpha, n, 'lower', obj_func))
            possible_pastes.append(discretePaste(x_init, y_init, y, name, box, box_init, paste_alpha, n, 'upper', obj_func))
        else:
            possible_pastes.append(realPaste(x_init, y_init, y, name, box, box_init, paste_alpha, n, 'lower', obj_func))
            possible_pastes.append(realPaste(x_init, y_init, y, name, box, box_init, paste_alpha, n, 'upper', obj_func))

    #break ties by choosing box with largest mass                 
    possible_pastes.sort(key=itemgetter(0,1), reverse=True)
    obj, mass_new, box_new = possible_pastes[0]
    logical = in_box(x_init, box_new)
    x_new = x_init[logical]
    y_new = y_init[logical]
    y_mean_new = np.mean(y_new)
    
    
    if (y_mean_new > threshold) & (mass_new >= mass_min) &\
       (y_mean_new >= y_mean) & (mass_new > mass):
        
        return paste(x_new, y_new, x_init, y_init, box_init,box_new, paste_alpha,
                     mass_min, threshold,n, obj_func)
    else:
        return box

def categoryPaste(x_init,y_init, y, name, box,n, obj_func):
    c_in_b = box[name][0]
    c_t = set(x_init[name])
    
    if len(c_in_b) < len(c_t):
        pastes = []
        possible_cs = c_t - c_in_b
        for entry in possible_cs:
            temp_box = np.copy(box)
            paste = copy.deepcopy(c_in_b)
            paste.add(entry)
            temp_box[name][:] = paste
            logical = in_box(x_init, box)
            obj = obj_func(y,  y_init[logical])
            mass_paste = y_init[logical].shape[0]/n
            pastes.append((obj, mass_paste, temp_box))
        return pastes
    else:
        # no pastes possible, return empty list
        return []

def discretePaste(x_init,y_init, y, name,
              box,box_init, paste_alpha, n, direction, obj_func):
    box_diff = box_init[name][1]-box_init[name][0]
    if direction == 'lower':
        i = 0
        paste_alpha = 1-paste_alpha
        box_diff = -1*box_diff
    if direction == 'upper':
        i = 1
    
    box_paste = np.copy(box)
    y_paste = y
    test_box = np.copy(box)
  
    if direction == 'lower':
        test_box[name][i+1] = test_box[name][i]
        test_box[name][i] = box_init[name][i]
        logical = in_box(x_init, test_box)
        data = x_init[logical][name]
        if data.shape[0] > 0:
            a = paste_alpha * y.shape[0]
            b = (data.shape[0]-a)/data.shape[0]
            paste_value = mquantiles(data, [b], alphap=1/3, betap=1/3)[0]
            paste_value = int(round(paste_value))
            box_paste[name][i] = paste_value
            logical = in_box(x_init, box_paste)
            y_paste = y_init[logical]
    
    if direction == 'upper':
        test_box[name][i-1] = test_box[name][i]
        test_box[name][i] = box_init[name][i]
        logical = in_box(x_init, test_box)
        data = x_init[logical][name]
        if data.shape[0] > 0:
            a = paste_alpha * y.shape[0]
            b = a/data.shape[0]
            paste_value = mquantiles(data, [b], alphap=1/3, betap=1/3)[0]
            paste_value = int(round(paste_value))
            box_paste[name][i] = paste_value
            logical = in_box(x_init, box_paste)
            y_paste = y_init[logical]

    # y means of pasted boxes
    obj = obj_func(y,  y_paste)
    
    # mass of pasted boxes
    mass_paste = y_init[logical].shape[0]/n

    return (obj, mass_paste, box_paste)

def realPaste(x_init,y_init, y, name,
              box,box_init, paste_alpha, n, direction, obj_func):
    
    box_diff = box_init[name][1]-box_init[name][0]
    if direction == 'lower':
        i = 0
        box_diff = -1*box_diff
    if direction == 'upper':
        i = 1
    
    box_paste = np.copy(box)
    test_box = np.copy(box)
    y_paste=y

    if direction == 'lower':
        test_box[name][i+1] = test_box[name][i]
        test_box[name][i] = box_init[name][i]
        logical = in_box(x_init, test_box)
        data = x_init[logical][name]
        if data.shape[0] > 0:
            a = paste_alpha * y.shape[0]
            b = (data.shape[0]-a)/data.shape[0]
            paste_value = mquantiles(data, [b], alphap=1/3, betap=1/3)[0]
            box_paste[name][i] = paste_value
            logical = in_box(x_init, box_paste)
            y_paste = y_init[logical]
    elif direction == 'upper':
        test_box[name][i-1] = test_box[name][i]
        test_box[name][i] = box_init[name][i]
        logical = in_box(x_init, test_box)
        data = x_init[logical][name]
        if data.shape[0] > 0:
            a = paste_alpha * y.shape[0]
            b = (a)/data.shape[0]
            paste_value = mquantiles(data, [b], alphap=1/3, betap=1/3)[0]
            box_paste[name][i] = paste_value
            logical = in_box(x_init, box_paste)
            y_paste = y_init[logical]

    # y means of pasted boxes
    obj = obj_func(y,  y_paste)
    
    # mass of pasted boxes
    mass_paste = y_init[logical].shape[0]/n

    return (obj, mass_paste, box_paste)

def vol_box(box):
    '''return volume of box'''
    
    a = []
    for entry in box.dtype.descr:
        name = entry[0]
        value = box.dtype.fields.get(entry[0])[0] 
        if value == 'object':
            a.append(len(box[name][0]))
        else:
            a.append(np.abs(box[name][1]-box[name][0]))
    a = np.asarray(a)
    return np.prod(a)

def equal(a,b):
    return np.all(compare(a,b))

def make_box(data):
    x = data
    box_init = np.zeros((2, ), x.dtype)
    #if no initial box, make initial box
    
    for entry in x.dtype.descr:
        name = entry[0]
        value = x.dtype.fields.get(entry[0])[0] 
        if value == 'object':
            box_init[name][:] = set(x[name])
        else:
            minimum = np.min(x[name], axis=0)
            maximum = np.max(x[name], axis=0)
            box_init[name][0] = minimum 
            box_init[name][1] = maximum    
    return box_init  

def compare(a, b):
    '''compare two boxes, return true if identical, false otherwise'''
    dtypesDesc = a.dtype.descr
    logical = np.ones((len(dtypesDesc,)), dtype=np.bool)
    for i, entry in enumerate(dtypesDesc):
        name = entry[0]
        
        test_1a = a[name][0]
        test_1b = b[name][0]
        test_1 = (a[name][0] == b[name][0])
        test_2 = (a[name][1] == b[name][1])
        
        logical[i] = logical[i] &\
                    (a[name][0] == b[name][0]) &\
                    (a[name][1] == b[name][1])
    return logical


def in_box(x, box):
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
    encompasing_box = make_box(x)

    for entry in x.dtype.descr:
        name = entry[0]
        value = x.dtype.fields.get(entry[0])[0]
        
        #test whether box is different from encompassing box
        if np.any(box[name] != encompasing_box[name]):
            if value == 'object':
                entries = box[name][0]
                logical = np.ones( (x.shape[0], len(entries)), dtype=np.bool)
                for i,entry in enumerate(entries):
                    if type(list(entries)[0]) not in (StringType, FloatType, IntType):
                        bools = []                
                        for element in list(x[name]):
                            if element == entry:
                                bools.append(True)
                            else:
                                bools.append(False)
                        logical[:, i] = np.asarray(bools, dtype=bool)
                    else:
                        logical[:, i] = x[name] == entry
                logical = np.any(logical, axis=1)
                x_box_ind = x_box_ind & logical
            else:
                x_box_ind = x_box_ind & (box[name][0] <= x[name] )&\
                                        (x[name] <= box[name][1])                
    return x_box_ind


#def in_box(x, box):
#    '''
#     Find points of x that are in a single box
#
#    alternative implementation which is suprisingly slower than the above
#
#     Parameters
#     x - data matrix
#     box_paste ranges - matrix of min and max values which define a box 
#     d - dimension of data
#    
#     Returns
#     Data points which lie within the box
#    '''
#
#    #identify by data type
#    objs = []
#    other = []
#    for entry in x.dtype.descr:
#        name = entry[0]
#        value = x.dtype.fields.get(entry[0])[0]
#        if value==float:
#            other.append(name)
#        else:
#            objs.append(name)
#    
#    #create a view on the data that contains only float dtype
#    float_box = box[other].view(float).reshape(2, len(other))
#    float_in_x = x[other].view(float).reshape(x.shape[0], len(other))
#    
#    #given these views, figure out which data points are inside the box
#    float_x_in_box = np.all( (float_in_x >= float_box[0,:][:, np.newaxis].T) &\
#                             (float_in_x <= float_box[1,:][:, np.newaxis].T),axis=1 )
#    
#    encompasing_box = make_box(x)
#    x_box_ind = float_x_in_box
#    for entry in objs:
#        name = entry
#        value = x.dtype.fields.get(name)[0]
#        
#        #test whether box is different from encompassing box
#        if np.any(box[name] != encompasing_box[name]):
#            if value == 'object':
#                entries = box[name][0]
#                logical = np.ones( (x.shape[0], len(entries)), dtype=np.bool)
#                for i,entry in enumerate(entries):
#                    logical[:, i] = x[name] == entry
#                logical = np.any(logical, axis=1)
#                x_box_ind = x_box_ind & logical
#            else:
#                x_box_ind = x_box_ind & (box[name][0] <= x[name] )&\
#                                        (x[name] <= box[name][1])   
#    x_box_ind = x_box_ind & float_x_in_box
#    
#    return x_box_ind


       
