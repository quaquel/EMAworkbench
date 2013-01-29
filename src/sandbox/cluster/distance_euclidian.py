'''
Created on Dec 19, 2011

@author: gyucel
'''



import numpy as np

from expWorkbench import EMAError
from expWorkbench.ema_logging import info

def eucldist(d1,d2):
    d = ((d1-d2)**2).sum()
    ed = np.sqrt(d)
    return ed

def distance_euclidian(data):
    
    '''
    The Euclidian distance is equal to the square root of (the sum of squared-differences between corresponding dimensions of two N-dimensional vectors) 
    (i.e. two data series of length N).
    Let the data series be of length N; Then Euclidian distance between ds1 and ds2 equals to sqrt(the sum of the square of error terms from 1 to N), 
    where error_term(i) equals to ds1(i)-ds2(i) 
    
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
            distance = eucldist(data[i],data[j]) 
            dRow[index] = distance
    return dRow, runLogs

if __name__ == '__main__':
    tester = np.array([(12,4),(2,2)])
    t1 = np.array([1,2,4])
    t2 = np.array([2,0,1])
    print eucldist(t1,t2)