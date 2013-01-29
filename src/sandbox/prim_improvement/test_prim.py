'''
Created on Mar 13, 2012

@author: jhkwakkel
'''
from __future__ import division

import numpy as np
np=np
import matplotlib.pyplot as plt

from expWorkbench import load_results, EMAlogging
from mpl_toolkits.axes_grid1 import host_subplot

import prim

#from analysis import prim

def classify_energy_trans(data):
    
    result = data['total fraction new technologies']    
    classes =  np.zeros(result.shape[0])
    classes[result[:, -1] < 0.35] = 1
    
    return classes

def classify_flu(data):
    
    result = data['infected fraction R1']    
    classes =  np.zeros(result.shape[0])
    classes[np.max(result, axis=1) > 0.05] = 1
    
    return classes


if __name__ == "__main__":
    EMAlogging.log_to_stderr(EMAlogging.INFO)

    results = load_results(r'./data/energy trans 1000 experiments.bz2')
  
    boxes = prim.perform_prim(results, 
                      classify=classify_energy_trans,
                      mass_min=0.05, 
                      threshold=0.8,
#                      obj_func=prim.orig_obj_func
                      )
    
#    results = load_results(r'./data/1000 flu cases no policy.bz2')
#  
#    boxes = prim.perform_prim(results, 
#                      classify=classify_flu,
#                      mass_min=0.05, 
#                      threshold=0.8)
    
    box = boxes[0]
    mean = box.p_and_p_trajectory["mean"]
    coverage = box.p_and_p_trajectory["coverage"]
    density = box.p_and_p_trajectory["density"]
    mass = box.p_and_p_trajectory["mass"]
    restricted_dim = box.p_and_p_trajectory["restricted_dim"]
    
    
    ax = host_subplot(111)
    ax.set_xlabel("peeling and pasting trajectory")
    
    par = ax.twinx()
    par.set_ylabel("nr. restricted dimensions")
        
    ax.plot(mean, label="mean")
    ax.plot(mass, label="mass")
    ax.plot(coverage, label="coverage")
    ax.plot(density, label="density")
    par.plot(restricted_dim, label="restricted_dim")
    ax.legend(loc='lower left')
    ax.grid(True, which='both')
    
#    prim.write_prim_to_stdout(boxes, screen=True)
    prim.show_boxes_together(boxes, results)
    
    plt.show()