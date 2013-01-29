'''
Created on 18 jul. 2011

@author: localadmin

test functions for prim, order of functions is order of testing
at end more comprehensive test functions will be provided
testing is to be done using same dataset as used in the documentation
of the R-version of prim

'''
import numpy as np
import matplotlib.pyplot as plt

from analysis.primCode.copyOldPrim import *
from recursivePrim import perform_prim
from analysis import graphs
from expWorkbench import ema_logging

__all__ = []

def test_vol_box():
    x = np.loadtxt(r'\quasiflow x.txt')
    box = np.array([np.min(x, axis=0),np.max(x, axis=0)])
    vol = vol_box(box)
    print "vol_box is " + str(vol) + " should be 0.68494973985"

def test_peel_one():
    #x = np.loadtxt(r'C:\workspace\EMA workbench\src\analysis\prim\quasiflow x.txt')
    #y = np.loadtxt(r'C:\workspace\EMA workbench\src\analysis\prim\quasiflow y.txt')
    
    x = np.loadtxt(r'C:\workspace\EMA workbench\src\analysis\prim\data')[:, 0:2]
    y = np.loadtxt(r'C:\workspace\EMA workbench\src\analysis\prim\data', dtype=np.int)[:,2]
    
    peel_alpha = 0.05
    paste_alpha = 0.01
    
    box_init = np.array([np.min(x, axis=0),np.max(x, axis=0)])
    box_diff = box_init[1, :] - box_init[0, :]
    box_init[0,:] = box_init[0, :] - 10*paste_alpha*box_diff
    box_init[1,:] = box_init[1, :] + 10*paste_alpha*box_diff
    
    threshold = -1.1
    d = x.shape[1]
    n = x.shape[0]
    mass_min = 0.05
    
    prim = peel_one(x, y, box_init, peel_alpha, mass_min, threshold, d, n)
    
    print "mass: " + str(prim.box_mass)
    print prim.box
    print "mean: " + str(prim.y_mean)
    

def test_in_box():
    x = np.loadtxt(r'C:\workspace\EMA workbench\src\analysis\prim\quasiflow x.txt')
    d = x.shape[1]
    box = np.array([(0.6,0.4),(0.7,0.9)])
    a = str(in_box(x, box, d, False))
    print "got " + str(a) + " expected [[ 0.68192771  0.89779006]]"

def test_paste_one():
    x_init = np.loadtxt(r'C:\workspace\EMA workbench\src\analysis\prim\quasiflow x.txt')
    y_init = np.loadtxt(r'C:\workspace\EMA workbench\src\analysis\prim\quasiflow y.txt')

    peel_alpha = 0.05
    paste_alpha = 0.01
    mass_min = 0.05
    threshold = np.min(y_init)-0.1*np.abs(np.min(y_init))
    d = x_init.shape[1]
    n = x_init.shape[0]
    box_init = np.array([np.min(x_init, axis=0),np.max(x_init, axis=0)])
    box_diff = box_init[1, :] - box_init[0, :]
    box_init[0,:] = box_init[0, :] - 10*paste_alpha*box_diff
    box_init[1,:] = box_init[1, :] + 10*paste_alpha*box_diff   
    
    prim = peel_one(x_init, y_init, box_init, peel_alpha, mass_min, threshold, d, n, type=8)
    
    box = paste_one(prim.x, prim.y, x_init, y_init, prim.box, paste_alpha, mass_min, threshold, d, n)
    print box.box_mass
    print box.box
    print box.y_mean
    

def test_find_box():
    #x = np.loadtxt(r'C:\workspace\EMA workbench\src\analysis\prim\quasiflow x.txt')[0:200, :]
    #y = np.loadtxt(r'C:\workspace\EMA workbench\src\analysis\prim\quasiflow y.txt')[0:200]
    
    x = np.loadtxt(r'C:\workspace\EMA workbench\src\analysis\prim\data', dtype=np.float64)[:, 0:2]
    y = np.loadtxt(r'C:\workspace\EMA workbench\src\analysis\prim\data', dtype=np.int)[:,2]
    
    peel_alpha = 0.05
    paste_alpha = 0.01
    
    box_init = np.array([np.min(x, axis=0),np.max(x, axis=0)])
    box_diff = box_init[1, :] - box_init[0, :]
    box_init[0,:] = box_init[0, :] - 10*paste_alpha*box_diff
    box_init[1,:] = box_init[1, :] + 10*paste_alpha*box_diff
    
    threshold = -1.1
    d = x.shape[1]
    n = x.shape[0]
    mass_min = 0.05
    
    pasting= False
    verbose= True
    
    box = find_box(x, y, box_init, peel_alpha, paste_alpha, mass_min, threshold, d, n, pasting, verbose)
    print "mass: " + str(box.box_mass)
    print box.box
    print "mean: " + str(box.y_mean)
    

def test_prim_hdr():
    pass

#    testing requires multiple prim boxes with data
#    prim_hdr(prim, threshold, threshold_type)

def test_prim_one():
    x = np.loadtxt(r'C:\workspace\EMA workbench\src\analysis\prim\quasiflow x.txt')[0:200, :]
    y = np.loadtxt(r'C:\workspace\EMA workbench\src\analysis\prim\quasiflow y.txt')[0:200]
        
    peel_alpha = 0.05
    paste_alpha = 0.01
    box_init = None
    mass_min = 0.05
    threshold = 0.1
    pasting = False
    verbose = True
    threshold_type = 1
    y = y*threshold_type
           
    boxes, numhdr = prim_one(x, y, box_init, peel_alpha, paste_alpha, mass_min, threshold, 
                     pasting, threshold_type, verbose)
    print numhdr
    
    for box in boxes:
        print box.box_mass
        print box.box
        print box.y_mean
    
    return boxes

def test_overlap_box():
    
    box1 = np.array([[0.6461445000000000105089, 0.124309000000000002828],
                    [0.7557829499999999534054, 0.423756999999999939277]])
    box2 = np.array([[0.6472408845000000576775, 0.1273034799999999966413],
                    [0.7546865654999999062369, 0.4207625199999999177081]])
    print str(overlap_box(box1, box2)) 


    box1 = np.array([[0, 0.2],
                     [0.2, 0.4]])
    box2 = np.array([[0.21, 0.1],
                     [0.3, 0.19]])
    print str(overlap_box(box1, box2)) 
    
    
    box1 = np.array([[-0.1, -0.2],
                     [0.2, 0.4]])
    box2 = np.array([[0.21, 0.1],
                     [0.3, 0.19]])
    print str(overlap_box(box1, box2)) 
    
def test_overlap_box_seq():
    pass

def test_prim_which_box():
    x = np.loadtxt(r'C:\workspace\EMA workbench\src\analysis\prim\quasiflow x.txt')
    y = np.loadtxt(r'C:\workspace\EMA workbench\src\analysis\prim\quasiflow y.txt')
    
    x = x[0:200, :]
    y = y[0:200]
    
    boxes = test_prim_one()
    print (prim_which_box(x, boxes))
    
def test_combine():
    pass

def test_box():
    ema_logging.log_to_stderr(ema_logging.INFO)
    
    x = np.loadtxt(r'quasiflow x.txt')
    y = np.loadtxt(r'quasiflow y.txt')
    
#    prim = prim_box(x, y, pasting=True, threshold = 0, threshold_type = -1)
    prim = perform_prim(x, y, pasting=True, threshold = 0, threshold_type =-1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x[:,0], x[:, 1], c=y)
    
    
    print '           \tmass\tmean'
    for i, entry in enumerate(prim[0:-1]):
        print 'found box %s:\t%s\t%s' %(i, entry.box_mass, entry.y_mean)
    print 'rest box    :\t%s\t%s' %(prim[-1].box_mass, prim[-1].y_mean)
    
    colors = graphs.COLOR_LIST
    for i, box in enumerate(prim):
        box = box.box
#        print box
        x = np.array([box[0,0], box[1,0], box[1,0], box[0,0], box[0,0]])
        y = np.array([box[0,1], box[0,1], box[1,1], box[1,1], box[0,1]])
#        print x
#        print y
        ax.plot(x,y, c=colors[i%len(colors)], lw=4)
    
    plt.show()     

def hacks():
    a = np.array( [[True, True], 
                   [True, False], 
                   [False, False]])
    print a
    print a.any(axis=1)

if __name__ == '__main__':

#    hacks()
#    test_peel_one()
#    test_paste_one()
#    test_find_box()
#    test_prim_one()
#    test_overlap_box()
#    test_prim_which_box()
    test_box()
