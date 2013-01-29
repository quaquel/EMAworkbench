'''
Created in May 2012 - June 2012

@author: Bas M.J. Keijser
'''
import numpy as np
import distance_gonenc as dg

def ABP(behavior,filter):
    # Behavior is a numpy array with the values of the VOI.
    # Time is a numpy array with the sizes of the timesteps.
    # So one shorter than behavior.
    
    difference = behavior[1:] - behavior[0:-1]
    slope = np.abs(difference)
    if filter:
        dg.filter_series(slope,behavior,0.1)
    
    # Every positive value becomes +1, which stands for exponential.
    # The other values are 0 (linear) and -1 (logarithmic).
    curv = slope[1:] - slope[0:-1]
    if filter:
        curv = dg.filter_series(curv,slope,0.1)
    ABP = np.sign(curv)
    
    # This to fit the lengths of the behavior vector and the ABP vector.
    # It is assumed that the ABP at the first time step is the same as in the second.
    try:
        ABP = np.concatenate(([ABP[0],ABP[0]],ABP))
    except IndexError:
        raise

    return ABP


def ABP_intervals(series):
    first_element = series[0]
    start_interval = 0
    interval_list = []
    index = 1
    
    for entry in series[1:]:
        
        if entry != first_element:
            interval_list.append((start_interval,index-1))
            start_interval = index
            first_element = entry
        index += 1
        
    interval_list.append((start_interval,len(series)-1))
            
    return interval_list
            
            
def ABP_change(ABP,switch_ABP):
    cond = np.equal(ABP,switch_ABP)

    if not np.any(cond):
        dominant = True
    else:
        dominant = False

    return dominant