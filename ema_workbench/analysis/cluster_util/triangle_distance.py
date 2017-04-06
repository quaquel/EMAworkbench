'''
Created on Nov 24, 2011

@author: gonengyucel
'''
from __future__ import print_function

import numpy as np
from ema_workbench.util import info


def trdist(d1,d2):
    
    a = np.sum(d1*d2)
    b = np.sqrt(np.sum(d1*d1))
    c = np.sqrt(np.sum(d2*d2))
    d = a/(b*c)
    print(d)
    return d

def distance_triangle(data):
    '''
    The triangle distance is calculated as follows;
        Let ds1(.) and ds2(.) be two data series of length N. Then;
        A equals to the summation of ds1(i).ds2(i) from i=1 to N
        B equals to the square-root of the (summation ds1(i)^2 from i=1 to N)
        C equals to the square-root of the (summation ds1(i)^2 from i=1 to N)
        
        distance_triangle = A/(B.C)
     
     The triangle distance works only with data series of the same length
     
     In the literature, it is claimed that the triangle distance can deal with noise and amplitude scaling very well, and may yield poor
     results in cases of offset translation and linear drift.   
    '''
    
    
    
    runLogs = []
    #Generates the feature vectors for all the time series that are contained in numpy array data
    info("calculating distances")
    dRow = np.zeros(shape=(np.sum(np.arange(data.shape[0])), ))
    index = -1
    for i in range(data.shape[0]):
            
        # For each run, a log is created
        # Log includes a description dictionary that has key information 
        # for post-clustering analysis, and the data series itself. These 
        # logs are stored in a global array named runLogs
        behaviorDesc = {}
        behaviorDesc['Index'] = str(i)
        
        behavior = data[i]
        localLog = (behaviorDesc, behavior)
        runLogs.append(localLog)
    
        for j in range(i+1, data.shape[0]):
            index += 1
            distance = trdist(data[i],data[j]) 
            dRow[index] = distance
    return dRow, runLogs

if __name__ == '__main__':
    tester = np.array([(1,2,3,4,5,6,7,8,12,15),(11,22,32,32,26,24,23,16,12,10)])
