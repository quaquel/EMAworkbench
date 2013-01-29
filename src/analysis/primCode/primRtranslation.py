from __future__ import division
'''
Created on 14 jul. 2011

@author: localadmin
'''
import numpy as np
import copy
from scipy.stats.mstats import mquantiles

from expWorkbench.ema_logging import info, debug

# TODO
# test code met random numbers shows boxes inside each other (without apparent
# overal..)
#

__all__ = ['prim_box',
           'prim_one',
           'find_box',
           'peel_one',
           'paste_one',
           'prim_which_box',
           'vol_box',
           'in_box',
           'overlap_box_seq',
           'overlap_box'
           ]

class PrimException(BaseException):
    pass

class Prim(object):
    def __init__(self, 
                 x,
                 y,
                 y_mean,
                 box,
                 box_mass,
                 threshold = 0):
        self.x = x
        self.y = y
        self.y_mean = y_mean
        self.box = box
        self.box_mass = box_mass
        self.threshold = threshold #vermoedelijk overbodig
    
def prim_box(x, 
             y, 
             box_init = None, 
             peel_alpha = 0.05, 
             paste_alpha = 0.01,
             mass_min = 0.05, 
             threshold = None, 
             pasting=True, 
             threshold_type=0):
    '''
     PRIM (Patient rule induction method)
    
     Parameters
     x - matrix of explanatory variables
     y - vector of response variable
     box.init - initial box (should cover range of x)
     mass.min - min. size of box mass
     y.mean.min - min. threshold of mean of y within a box
     pasting - TRUE - include pasting step (after peeling)
             - FALSE - don't include pasting
    
     Returns
     list with k fields, one for each box
     each field is in turn is a list with fields
     x - data inside box
     y - corr. response values
     y.mean - mean of y
     box - limits of box
     box.mass - box mass
     num.boxes - total number of boxes with box mean >= y.mean.min
     '''
    
    if (threshold_type == 1) or (threshold_type == -1):
        if threshold==None:
            threshold = np.mean(y)
        prim_temp, num_hdr = prim_one(x, y*threshold_type, box_init, peel_alpha, 
                             paste_alpha, mass_min, threshold, pasting, 
                             threshold_type)
            
    else:
        if threshold == None:
            threshold = np.array([np.mean(y), np.mean(y)])
        elif threshold.shape[0] == 1:
            raise PrimException("Need both upper and lower values for threshold")

        prim_pos, num_hdr_pos = prim_one(x=x, y=y, box_init=box_init, peel_alpha=peel_alpha,
                                         paste_alpha=paste_alpha, mass_min=mass_min,
                                         threshold_type=1, threshold=threshold[0],
                                         pasting=pasting)
        prim_neg, num_hdr_neg = prim_one(x=x, y=-1*y, box_init=box_init, peel_alpha=peel_alpha,
                                         paste_alpha=paste_alpha,mass_min=mass_min,
                                         threshold_type=-1, threshold=threshold[1],
                                         pasting=pasting)
        prim_temp = prim_combine(prim_pos, prim_neg, num_hdr_pos, num_hdr_neg)    

    ## re-do prim to ensure that no data points are missed from the `dump' box 
    prim_reg = prim_temp
    prim_labels =  prim_which_box(x=x, box_seq=prim_reg)
    
    for k, prim in enumerate(prim_reg):
        primk_ind = np.where(prim_labels == k)
        prim.x = x[primk_ind]
        prim.y = y[primk_ind]
        prim.logical = prim_labels == k
        prim.y_mean = np.mean(prim.y)
        prim.box_mass = prim.y.shape[0]/x.shape[0] #number of rows
    
    return prim_reg

def prim_one(x,
             y,
             box_init = None,
             peel_alpha = 0.05,
             paste_alpha = 0.01,
             mass_min = 0.05,
             threshold = None,
             pasting = False,
             threshold_type = 1):

    d = x.shape[1]
    n = x.shape[0]
    
    k_max = np.ceil(1/mass_min)
    info("k max: %s" %(k_max))
    num_boxes = int(k_max)
    
    y_mean =  np.mean(y)
    mass_init =  y.shape[0]/n #should default to 1 if I read the code correctly
    
    if box_init == None:
        box_init = np.array([np.min(x, axis=0),np.max(x, axis=0)])
        box_diff = box_init[1, :] - box_init[0, :]
        box_init[0,:] = box_init[0, :] - 10*paste_alpha*box_diff
        box_init[1,:] = box_init[1, :] + 10*paste_alpha*box_diff
  
    # find first box
    k = 1

    a = x.shape[0]
    debug("remaing items: %s" % (a))

    boxk = find_box(x=x, y=y, box=box_init, peel_alpha=peel_alpha,
                   paste_alpha=paste_alpha, mass_min=mass_min,
                   threshold=np.min(y)-0.1*np.abs(np.min(y)), d=d, n=n, 
                   pasting=pasting)

    b = boxk.x.shape[0]
    debug("removed items: %s" %b)


    if boxk == None:
        info("unable to find box 1")  
        x_prim = Prim(x, threshold_type*y, y_mean=threshold_type*y_mean, 
                      box=box_init, box_mass=mass_init, threshold=np.mean(y)) 
        return x_prim
    else:
        info("Found box %s: y_mean=%s, mass=%s" % (k, threshold_type*boxk.y_mean, boxk.box_mass))
        boxes = []
        boxes.append(boxk)
    
    # find subsequent boxes
    if num_boxes > 1:

        #  data still under consideration
        x_out_ind_mat = np.empty(x.shape)
    
        for j in range(d):
            x_out_ind_mat[:, j] = (x[:, j] < boxk.box[0,j]) | (x[:,j] >boxk.box[1,j]) 
        
        x_out_ind = np.any(x_out_ind_mat,axis=1)
        
        x_out =  x[x_out_ind,:]
        y_out =  y[x_out_ind]
     
        a = x_out.shape[0]
        debug("remaing items: %s" % (a))
   
        while (y_out.shape[0] > 0) & (k < num_boxes):
            k = k+1
            
            boxk = find_box(x=x_out, y=y_out, box=box_init,
                           peel_alpha=peel_alpha, paste_alpha=paste_alpha,
                           mass_min=mass_min, 
                           threshold=np.min(y)-0.1*np.abs(np.min(y)), d=d, n=n,
                           pasting=pasting) 
            if boxk == None:
                info("Bump "+str(k)+" includes all remaining data")
                boxk = Prim(x_out, y_out, np.mean(y_out), box_init, 
                            y_out.shape[0]/n)
                b += boxk.x.shape[0]
                debug("removed items: %s" %b)
                
                boxes.append(boxk)
                break
            else:
                b += boxk.x.shape[0]
                debug("removed items: %s" %b)
                
                # update x and y
                debug("Found box %s: y_mean=%s, mass=%s" % (k, threshold_type*boxk.y_mean, boxk.box_mass))
        
                #  data still under consideration
                x_out_ind_mat = np.empty(x.shape)
            
                for j in range(d):
                    x_out_ind_mat[:, j] = (x[:, j] < boxk.box[0,j]) | (x[:,j] >boxk.box[1,j]) 
                
                x_out_ind_mat = np.any(x_out_ind_mat,axis=1)
                x_out_ind = x_out_ind & x_out_ind_mat
                
                x_out =  x[x_out_ind, :]
                y_out =  y[x_out_ind]
                
                a = x_out.shape[0]
                debug("remaing items: %s" %(a))
    
    
                boxes.append(boxk)

    # adjust for negative hdr  
    for box in boxes:
        box.y = threshold_type*box.y
        box.y_mean = threshold_type*box.y_mean

    prim_res, num_hdr=  prim_hdr(boxes, threshold, threshold_type)
    
    return prim_res, num_hdr

def prim_hdr(prim,
             threshold,
             threshold_type):
    '''
    Highest density region for PRIM boxes
    
    prim        list of prim objects
    threshold    
    threshold_type
    
    '''
    
    n = 0
    for entry in prim:
        n += entry.y.shape[0]
    info("number of items in boxes: %s" %n)

    y_means = np.asarray([entry.y_mean for entry in prim])
    hdr_ind =  np.where(y_means * threshold_type >= threshold*threshold_type)[0]
    
    if hdr_ind.shape[0] > 0:
        hdr_ind = np.max(hdr_ind)
    else:
        if threshold_type ==1:
            raise Warning("No prim box found with mean >= "+str(threshold))
        elif threshold_type ==-1:
            raise Warning("No prim box found with mean <= "+str(threshold))
        return None

    #highest density region  
    x_prim_hdr = []
    
    for k in range(hdr_ind+1):
        hdr = prim[k]
        x_prim_hdr.append(hdr)

    #combine non-hdr into a `dump' box
    if hdr_ind < len(prim)-1:
        info("making a dumpbox")
        x_temp = None
        for k in range(hdr_ind+1,len(prim)): #dit moet via een slice veel sneller kunnen
            if x_temp == None:
                x_temp = prim[k].x
                y_temp = prim[k].y
            else:
                x_temp = np.append(x_temp, prim[k].x, axis=0) 
                y_temp = np.append(y_temp, prim[k].y, axis=0)

        dump_box = Prim(x_temp, y_temp, np.mean(y_temp), prim[-1].box, 
                        y_temp.shape[0]/n)
        
        x_prim_hdr.append(dump_box)

    #dit kan niet wat hdr is een list
    x_prim_hdr_num_class = len(x_prim_hdr)
    x_prim_hdr_num_hdr_class = hdr_ind+1
    x_prim_hdr_threshold = threshold
    
    x_prim_hdr_ind = np.zeros((x_prim_hdr_num_class)) 
    x_prim_hdr_ind[:] = threshold_type

    return x_prim_hdr, x_prim_hdr_num_hdr_class


def prim_combine(prim1,
                 prim2,
                 num_hdr_pos,
                 num_hdr_neg):
    '''
     Combine (disjoint) PRIM box sequences - useful for joining
     positive and negative estimates
    
     Parameters
     prim1 - 1st PRIM box sequence
     prim2 - 2nd PRIM box sequence
    
     Returns
     same as for prim()
    '''
    M1 = num_hdr_pos
    M2 = num_hdr_neg

    if (M1==0) & (M2!=0):
        return prim2
    if (M1!=0) & (M2==0):
        return prim1
    if (M1==0) & (M2==0):
        return None
    
    overlap = overlap_box_seq(prim1, prim2)
    
    x = None
    for i in range(len(prim1)):
        if x==None:
            x= prim1[i].x
            y= prim1[i].y
        else:
            x = np.append(x, prim1[i].x, axis=0)
            y = np.append(y, prim1[i].y, axis=0)

    if overlap.any():
        Warning("Class boundaries overlap - will return NULL")
        return None
    else:
        prim_temp = [prim for prim in prim1]
        [prim_temp.append(prim) for prim in prim2]
        
    
    dumpx_ind = prim_which_box(x, prim1)==prim1[-1] & prim_which_box(x, prim2)==prim2[-1]
    
    
    
    prim = Prim(x[dumpx_ind,:], y[dumpx_ind], np.mean(y[dumpx_ind]), prim1[-1].box, 
                y[dumpx_ind].shape[0]/y.shape[0], None)
    prim_temp.append(prim)
    return prim_temp

def find_box(x,
             y,
             box,
             peel_alpha,
             paste_alpha,
             mass_min,
             threshold,
             d,
             n,
             pasting):
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
    
    y_mean =  np.mean(y)
    mass = y.shape[0]/n 

    if (y_mean >= threshold) & (mass >= mass_min):
        boxk_peel = peel_one(x, y, box, peel_alpha, mass_min, threshold, d, n)
    else:
        boxk_peel = None
         
    boxk_temp = None
        
    while boxk_peel:
        boxk_temp = copy.deepcopy(boxk_peel)
        boxk_peel = peel_one(boxk_temp.x, boxk_temp.y, boxk_temp.box, 
                             peel_alpha, mass_min, threshold, d, n)
    

    info("peeling completed")

    if pasting:
        boxk_paste = boxk_temp
        
        while boxk_paste:
            boxk_temp = boxk_paste
            boxk_paste = paste_one(boxk_temp.x, boxk_temp.y, x, y, 
                                   boxk_temp.box, paste_alpha, mass_min, 
                                   threshold, d, n)
        info("pasting completed")
            
    boxk = boxk_temp
    return boxk

def peel_one(x,
             y,
             box,
             peel_alpha,
             mass_min,
             threshold,
             d,
             n):
    '''
     Peeling stage of PRIM
    
     Parameters
     x - data matrix
     y - vector of response variables
     box
     peel.alpha - peeling quantile
     paste.alpha - peeling proportion
     mass.min - minimum box mass
     threshold - minimum y mean
     d - dimension of data
     n - number of data
     
     Returns
     List with fields
     x - data still inside box after peeling
     y - corresponding response values
     y.mean - mean of y
     box - box limits
     mass - box mass
    '''
    alpha = 1/3
    beta = 1/3
   
    box_new = copy.deepcopy(box)
    mass = y.shape[0]/n
    
    if x.shape[1] == 0: return None 
    y_mean_peel = np.zeros((2, d))
    box_vol_peel = np.zeros((2,d))

    for j in range(d):
        box_min_new = mquantiles(x[:, j], [peel_alpha], alphap=alpha, betap=beta)[0]
        box_max_new = mquantiles(x[:, j], [1-peel_alpha], alphap=alpha, betap=beta)[0]
        
        y_mean_peel[0, j] = np.mean(y[ x[:, j] >= box_min_new])
        y_mean_peel[1, j] = np.mean(y[ x[:, j] <= box_max_new])
        
        box_temp1 = copy.deepcopy(box)
        box_temp2 = copy.deepcopy(box)
        
        box_temp1[0,j] = box_min_new
        box_temp2[1,j] = box_max_new
        
        box_vol_peel[0,j] = vol_box(box_temp1)
        box_vol_peel[1,j] = vol_box(box_temp2)
    
    
    #break ties by choosing box with largest volume
    y_mean_peel_max_ind = np.where(y_mean_peel == np.max(y_mean_peel))
    nrr  = y_mean_peel_max_ind[0].shape[0]
 
    if nrr > 1:
        box_vol_peel2 = box_vol_peel[y_mean_peel_max_ind] 
        row_ind= np.argmax(box_vol_peel2)
    else:
        row_ind = 0
     
    y_mean_peel_max_ind = [y_mean_peel_max_ind[0][row_ind], y_mean_peel_max_ind[1][row_ind]]
    
    # peel along dimension j.max
    j_max = y_mean_peel_max_ind[1]
    
    # peel lower
    if y_mean_peel_max_ind[0]==0:
        box_new[0, j_max] = mquantiles(x[:, j_max], [peel_alpha], alphap=alpha, betap=beta )[0]
        x_index = ( x[:, j_max] >= box_new[0, j_max] ) & ( x[:, j_max] <= box[1, j_max] )
    # peel upper 
    elif y_mean_peel_max_ind[0]==1:
        box_new[1, j_max] = mquantiles(x[:, j_max], [1-peel_alpha], alphap=alpha, betap=beta)[0]
        x_index = ( x[:, j_max] <= box_new[1, j_max] ) & ( x[:, j_max] >= box[0, j_max] )
    else:
        raise Warning("should not happen in peel")

    x_new = x[x_index, :]
#    info(x_new.shape)
    y_new = y[x_index]
    mass_new = y_new.shape[0]/n
    y_mean_new =  np.mean(y_new)

    # if min. y mean and min. mass conditions are still true, update
    # o/w return NULL  
    
    if (y_mean_new >= threshold) & (mass_new >= mass_min) & (mass_new < mass):
        return Prim(x_new, y_new, y_mean_new, box_new, mass_new)
    else:
        return None

def paste_one(x,
              y,
              x_init,
              y_init,
              box,
              paste_alpha,
              mass_min,
              threshold,
              d,
              n):
    '''
     Pasting stage for PRIM
    
     Parameters
     x - data matrix
     y - vector of response variables
     x.init - initial data matrix (superset of x) 
     y.init - initial response vector (superset of y) 
     box
     paste.alpha - peeling proportion
     mass.min - minimum box mass
     threshold - minimum y mean
     d - dimension of data
     n - number of data
     
     Returns
    
     List with fields
     x - data still inside box after pasting
     y - corresponding response values
     y.mean - mean of y
     box - box limits
     box.mass - box mass
    '''

    box_new = copy.deepcopy(box)
    mass = y.shape[0]/n
    y_mean = np.mean(y)
    
    box_init = np.array([np.min(x_init, axis=0),np.max(x_init, axis=0)])
    
    y_mean_paste = np.zeros((2,d))
    mass_paste = np.zeros((2,d))
    box_paste = np.zeros((2,d))
    
    x_paste1_list = []
    x_paste2_list = []
    y_paste1_list = []
    y_paste2_list = []
    
    box_paste1 = copy.deepcopy(box_new)
    box_paste2 = copy.deepcopy(box_new)
     
    for j in range(d):
        # candidates for pasting
        box_diff = (box_init[1,:] - box_init[0,:])[j]
        
        box_paste1[0,j] = box[0,j]-box_diff*paste_alpha
        box_paste2[1,j] = box[1,j]+box_diff*paste_alpha

        x_paste1_ind = in_box(x_init,box_paste1,d, True)
        x_paste1 = x_init[x_paste1_ind,:]
        y_paste1 = y_init[x_paste1_ind]
        
        x_paste2_ind = in_box(x_init,box_paste2,d, True)
        x_paste2 = x_init[x_paste2_ind]
        y_paste2 = y_init[x_paste2_ind]

        while (y_paste1.shape[0] <= y.shape[0]) & (box_paste1[0,j] >= box_init[0,j]):
            box_paste1[0,j] = box_paste1[0,j]- box_diff*paste_alpha
            x_paste1_ind = in_box(x_init, box_paste1, d, True)
            x_paste1 = x_init[x_paste1_ind,:]
            y_paste1 = y_init[x_paste1_ind]
        
        while (y_paste2.shape[0] <= y.shape[0]) & (box_paste2[1,j] <= box_init[1,j]):
            box_paste2[1,j] = box_paste2[1,j]+ box_diff*paste_alpha
            x_paste2_ind = in_box(x_init, box_paste1, d, True)
            x_paste2 = x_init[x_paste2_ind,:]
            y_paste2 = y_init[x_paste2_ind]

        # y means of pasted boxes
        y_mean_paste[0,j] = np.mean(y_paste1)
        y_mean_paste[1,j] = np.mean(y_paste2)
        
        # mass of pasted boxes
        mass_paste[0,j] = y_paste1.shape[0]/n
        mass_paste[1,j] = y_paste2.shape[0]/n

        x_paste1_list.append(x_paste1)
        x_paste2_list.append(x_paste2)
        y_paste1_list.append(y_paste1)
        y_paste2_list.append(y_paste2)
    
        box_paste[0,j] = box_paste1[0,j]
        box_paste[1,j] = box_paste2[1,j]

#    #break ties by choosing box with largest volume                 
  
    y_mean_paste_max_ind = np.where(y_mean_paste==np.max(y_mean_paste))
    
    if y_mean_paste_max_ind[0].shape[0]>1:
#        print "more then one box in paste"
        
        #figure out where mass is max
        mass_max =  mass_paste[y_mean_paste_max_ind]
        index = np.where(np.max(mass_max) == mass_max)[0][0]

        #return column index of b where a = max
        y_mean_paste_max_ind = np.asarray(y_mean_paste_max_ind).T[index, :]
    else:
        y_mean_paste_max_ind = np.asarray(y_mean_paste_max_ind).T[0, :]
  
    # paste along dimension j.max
    j_max = y_mean_paste_max_ind[1]
    
    # paste lower
    if y_mean_paste_max_ind[0] == 0:
        x_new = x_paste1_list[j_max]
        y_new = y_paste1_list[j_max]
        box_new[0, j_max] = box_paste[0, j_max] 
    # paste upper
    elif y_mean_paste_max_ind[0]==1:
        x_new = x_paste2_list[j_max]
        y_new = y_paste2_list[j_max]
        box_new[1,j_max] = box_paste[1,j_max]
    else:
        raise Warning("paste behaves strange, more then 2 rows")
    
    mass_new = y_new.shape[0]/n
    y_mean_new = np.mean(y_new)
    
    if (y_mean_new > threshold) & (mass_new >= mass_min) &\
        (y_mean_new >= y_mean) & (mass_new > mass):
        return Prim(x_new, y_new, y_mean_new, box_new, mass_new)
    else:
        return None

def prim_which_box(x, box_seq):
    '''
     Returns the box number which the data points belong in
    
     Parameters
    
     x - data matrix
     box.seq - list of prim objects
    
     Returns
    
     Vector of box numbers
    '''
    pass
    
    m = len(box_seq)
    d = x.shape[1]
    n = x.shape[0]
    
    x_ind = np.ones(n, dtype=np.bool)
    x_which_box = np.zeros(n)

    for k in range(m):
        x_ind_curr = x_ind
        box_curr = box_seq[k].box
        for j in range(d):
            x_ind_curr = x_ind_curr &\
                         (x[:, j] >= box_curr[0, j]) &\
                         (x[:, j] <= box_curr[1, j])
        x_which_box[x_ind_curr & x_ind] = k
        
        x_ind = x_ind & (x_ind_curr==False)
    return x_which_box



def in_box(x, box, d, bool=False):
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

    for i in range(d):
        x_box_ind = x_box_ind & (box[0,i] <= x[:, i] )& (x[:, i] <= box[1, i])
    
    if bool:
        return x_box_ind   
    else:
        return x[x_box_ind]

def overlap_box_seq(box_seq1, box_seq2, rel_tol = 0.01):
    '''
    Decide whether two box sequences overlap each other
    
    Input
    box.seq1 - first box sequence
    box.seq2 - second box sequence
    
    Returns
    TRUE if they overlap, FALSE o/w
    '''
    m1 = len(box_seq1)-1
    m2 = len(box_seq2)-1

    overlap_mat = np.zeros((m1, m2), dtype=np.bool)
    for i in range(m1):
        for j in range(m2):
            box1 = box_seq1[i].box
            box2 = box_seq2[j].box
            overlap_mat[i,j] = overlap_box(box1, box2, rel_tol)
    return overlap_mat

def overlap_box(box1, box2, rel_tol=0.01):
    '''
    Decide whether two boxes overlap each other
    
    Input
    box1 - first box 
    box2 - second box
    
    Returns
    TRUE if they overlap, FALSE o/w
    '''
    d = box1.shape[1]
    
    overlap = True
    box1_tol = box1
    box1_range = np.abs(np.diff(box1, axis=0))
    box1_tol[0, :] = box1[0, :] + rel_tol*box1_range
    box1_tol[1, :] = box1[1, :] - rel_tol*box1_range
    
    box2_tol = box2
    box2_range = np.abs(np.diff(box2, axis=0))
    box2_tol[0, :] = box2[0, :] + rel_tol*box2_range
    box2_tol[1, :] = box2[1, :] - rel_tol*box2_range


    for k in range(d):
        overlap = overlap & (   ((box1_tol[0,k] <= box2_tol[0,k]) & (box2_tol[0,k] <= box1_tol[1,k]))
                              | ((box1_tol[0,k] <= box2_tol[1,k]) & (box2_tol[1,k] <= box1_tol[1,k]))
                              | ((box2_tol[0,k] <= box1_tol[0,k]) & (box1_tol[0,k] <= box2_tol[1,k]))
                              | ((box2_tol[0,k] <= box1_tol[1,k]) & (box1_tol[1,k] <= box2_tol[1,k])))
    return overlap

def vol_box(box):
    return np.prod(np.abs(box[1, :]-box[0,:]))

if __name__ == '__main__':
    pass          