'''
Created on Dec 19, 2011

@author: gyucel
'''
import numpy as np

from ema_workbench.util import info

def msedist(d1,d2):
    sse = ((d1-d2)**2).sum()
    mse = np.average(sse)
    return mse

def distance_mse(data):
    '''
    The MSE (mean squared-error) distance is equal to the SSE distance divided by the number of data points in data series.
    
    The SSE distance between two data series is equal to the sum of squared-errors between corresponding data points of these two data series.
    Let the data series be of length N; Then SSE distance between ds1 and ds2 equals to the sum of the square of error terms from 1 to N, 
    where error_term(i) equals to ds1(i)-ds2(i) 
    
    Given that SSE is calculated as given above, MSE equals SSE divided by N.
    
    As SSE distance, the MSE distance only works with data series of equal length.
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
            distance = msedist(data[i],data[j]) 
            dRow[index] = distance
    return dRow, runLogs

if __name__ == '__main__':
    tester = np.array([(12,4),(2,2)])