'''
Created on Nov 8, 2011

@author: jhkwakkel
'''
from __future__ import division
import numpy as np

from expWorkbench.EMAlogging import info


def get_trend(timeSeries, trendThold):
    lenght = timeSeries.shape[0]
    slope = timeSeries[1::]-timeSeries[0:lenght-1]
    
    slopeMean = np.sum(slope/slope.shape[0])
    dsMean = np.sum(timeSeries/timeSeries.shape[0])
    
    if slopeMean/dsMean > trendThold :
        return 1
    elif slopeMean/dsMean < -1*trendThold:
        return -1
    else:
        return 0

def get_nr_of_crises(timeSeries, crisisThold):
    lenght = timeSeries.shape[0]
    slope = timeSeries[1::]-timeSeries[0:lenght-1]
    slope = slope/timeSeries[0:lenght-1]
    crises = np.zeros(slope.shape[0])        
    crises[slope>crisisThold]=1
    
    last = 0
    nrCrises = 0
    for i in range(crises.shape[0]):
        next = crises[i]
        if (next != last) & (last == 1):
            nrCrises += 1
        last = next
    if last==1:
        nrCrises+=1
    return nrCrises, np.max(np.abs(slope))

def trdist(d1,d2):
    
    a = np.sum(d1*d2)
    b = np.sqrt(np.sum(d1*d1))
    c = np.sqrt(np.sum(d2*d2))
    d = a/(b*c)
#    print d
    return d

def construct_feature_vector(timeSeries, trendThold, crisisThold):
    nrCrises, severityOfCrises = get_nr_of_crises(timeSeries, crisisThold)
#    trend = get_trend(timeSeries, trendThold)
#    bandwith = np.max(timeSeries) - np.min(timeSeries)
    if nrCrises > 0:
        return (1, nrCrises, severityOfCrises)
    else:
        return (0, nrCrises, severityOfCrises)


def construct_features(data, 
                       trendThold, 
                       crisisThold):
    info("calculating features")
    
    # Checks the parameters of the distance function that may be defined by the user in the distanceSetup dict
    
    
    features = np.zeros(shape=(data.shape[0], 3))
    for i in range(data.shape[0]):
        features[i,:] = construct_feature_vector(data[i, :], trendThold, crisisThold)
    return features

def distance_willem(data, 
                    trendThold=0.001, 
                    crisisThold=0.02,
                    wIfCrisis=1,
                    wNoOfCrises=1,
                    wTrend=1,
                    wBandwith=1,
                    wSevCrises=1,
                    wTriDist=0.5):
    '''
    
    
    :param data: the time series for which to calculate the distances
    :param trendThold: threshold for trend
    :param crisisThold: threshold for crisis
    :param wIfCrisis: weight of crisis
    :param wNoOfCrisis: weight of number of crises
    :param wTrend: weight of trend
        
    '''
    
    
    runLogs = []
    features = construct_features(data, trendThold, crisisThold)
    
    #normalize
    norm_features = features.copy()
    np.log(norm_features[:, 1]+1)
    minimum = np.min(features, axis=0) 
    maximum = np.max(features, axis=0)
    a = 1/(maximum-minimum)
    b = -minimum/maximum-minimum
    norm_features = a*features+b
    
    
    info('calculating distances')
    dRow = np.zeros(shape=(np.sum(np.arange(data.shape[0])), ))
    index = 0
    
    weights = np.array([wIfCrisis,
                       wNoOfCrises,
                       wSevCrises])
    max_distance = 0
    for i in range(data.shape[0]):
        feature_i = norm_features[i]
        # For each run, a log is created
        # Log includes a description dictionary that has key information for post-clustering analysis, and the data series itself
        # These logs are stored in a global array named runLogs
        behaviorDesc = {}
        behaviorDesc['Index'] = str(i)
        featVector = features[i] #this may not work due to data type mismatch
        featVector = tuple(featVector)
        behaviorDesc['Feature vector'] = "%d, %d, %s" % featVector
        behavior = data[i]
        localLog = (behaviorDesc, behavior)
        runLogs.append(localLog)
        
        for j in range(i+1, data.shape[0]):
            distance_tri = trdist(data[i],data[j])
            
            max_distance = max((max_distance, distance_tri))
             
            feature_j = norm_features[j]
            distance = np.abs(feature_i -feature_j)
            distance = weights*distance
            distance = np.sum(distance)+(distance_tri*wTriDist)
            dRow[index] = distance
            index += 1
        
#        distance = np.abs(feature_i - norm_features[i+1::])
#        distance = weights*distance
#        distance = np.sum(distance, axis=1)
#        dRow[index:index+distance.shape[0]] = distance
#        index += distance.shape[0]
    print max_distance
    info('distances determined')
    return dRow, runLogs

def test_trend():
    x = np.arange(0, 100)
    timeSeries = x*x
    trendTHold = 0.001
    print get_trend(timeSeries, trendTHold)
    
    timeSeries = -1*x
    print get_trend(timeSeries, trendTHold)
    
    timeSeries = x
    print get_trend(timeSeries, trendTHold)
    
    timeSeries = (x-50)^2
    print get_trend(timeSeries, trendTHold)
    
    timeSeries = np.zeros(100)
    print get_trend(timeSeries, trendTHold)


def test_nr_of_crises():
    x = np.arange(0, 100)
    crisisThold = 0.02
    
    timeSeries = np.sin(x)
    print get_nr_of_crises(timeSeries, crisisThold)
    

if __name__ == '__main__':
#    test_trend()
    test_nr_of_crises()
