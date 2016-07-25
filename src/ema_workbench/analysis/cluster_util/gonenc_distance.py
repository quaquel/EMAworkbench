'''

Created on Nov 8, 2011

.. codeauthor:: 
     gyucel <g.yucel (at) tudelft (dot) nl>,
     jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import division

import random

import numpy as np
from ema_workbench.util import info


def distance_same_length(series1, series2, wDim1, wDim2):
    '''
    Calculates the distance between two feature vectors of the same size.
    
    :param series1: Feature vector 1 (2-dimensional numpy array).
    :param series2: Feature vector 2 (2-dimensional numpy array).
    :param wDim1: Weight of the error between the 1st dimensions of the two 
                  feature vectors (i.e. Slope).
    :param wDim2: Weight of the error between the 2nd dimensions of the two 
                  feature vectors (i.e. Curvature).
    
    '''
    
    error = np.square(series1-series2)
    error = np.array([wDim1, wDim2])[np.newaxis].T * error
    error = np.sum(error)
    
    return error/series1.shape[1]

def distance_different_lenght(series1, series2, wDim1, wDim2, sisterCount):
    '''
    Calculates the distance between two feature vectors of different sizes.
    
    :param series1: Feature vector 1 (2-dimensional numpy array).
    :param series2: Feature vector 2 (2-dimensional numpy array).
    :param wDim1: Weight of the error between the 1st dimensions of the two 
                  feature vectors (i.e. Slope).
    :param wDim2: Weight of the error between the 2nd dimensions of the two 
                  feature vectors (i.e. Curvature).
    :param sisterCount: Number of long-versions that will be created for the 
                        short vector.
    '''
    
    # TODO Check why the initial value of error is different for the two 
    # distance methods?
    
    length1 = series1.shape[1]
    length2 = series2.shape[1]
#    toAdd = abs(length1-length2)
    
    if length1>length2:
        shortFV = series2
        longFV = series1
    else:
        shortFV = series1
        longFV = series2
    
    sisters = create_sisters(shortFV, longFV.shape, sisterCount)
    
    # to take advantage of the fact that the sisters are in a 3d array
    # I also vectorized the error calculation.
    # this means that calculation time is almost independent from the number
    # of sisters you want to use.
    error = np.square(sisters - longFV.T[np.newaxis,:,:])
    weights = np.array([wDim1, wDim2])
    
    error = error*weights[np.newaxis, np.newaxis, :]
    
    # this is sort of stupid, it should be possible to do in one line
    # but axis=(1,2) does not work, it requires numpy 1.7. I have 1.6.1 
    # at the moment
    error = np.sum(error, axis=1)
    error = np.sum(error, axis=1)
    return np.min(error)/longFV.shape[1]
#    for i in range(sisterCount):
#        sisterFV = createSister(shortFV, toAdd)
#        tempError = distance_same_length(sisterFV, longFV, wDim1, wDim2)
#        if tempError < error:
#            error = tempError
#    return error

def create_sisters(shortFV, desired_shape, sisterCount):
    '''
    Creates a new feature vector that is behaviorally identical to the given 
    vector by adding toAdd number of segments.
    
    :param shortFV: The feature vector to be extended.
    :param toAdd: Number of sections to be added to the input vector while 
                  creating the equivalent sister.
    ''' 
    
    #determine how much longer the vector has to become
    to_add = desired_shape[1]-shortFV.shape[1]
    
    #create a 2d array of indices
    indices = np.zeros(shape=(sisterCount, desired_shape[1]),dtype=int)
    
    #fill the first part of the indices array with random numbers
    #these are the indices that will be used to extent the short vector
    indices[:, 0:to_add] = np.random.randint(0, 
                                             shortFV.shape[1], 
                                             size=(sisterCount, to_add))
    
    #add the indices for the full vector to the rest
    indices[:, to_add::] = np.arange(0, shortFV.shape[1])[np.newaxis, :]
    
    #sort indices
    indices = np.sort(indices, axis=1)
    
    #this is where the real magic happens, we use the generated indices
    #in order to generate in one line of code all the sisters
    sisters = shortFV.T[indices,:] 
    
    return sisters


def createSister(shortFV, toAdd):
    '''
    Creates a new feature vector that is behaviorally identical to the given 
    vector by adding toAdd number of segments.
    
    :param shortFV: The feature vector to be extended.
    :param toAdd: Number of sections to be added to the input vector while 
                  creating the equivalent sister.
    ''' 
    sister = np.zeros(shape=(shortFV.shape[0], shortFV.shape[1]+toAdd))
    
    
    # while you have to add, add a random number of entries
    index = 0 
    i = 0
    while (toAdd>0) and (i<shortFV.shape[1]):
        
        x = random.randint(0, toAdd)
        sister[:, index:index+x+1] = shortFV[:,i][np.newaxis].T
        toAdd -= x
        index += x+1
        i+=1
    
    #fill up with the remaining values from short
    if i<shortFV.shape[1]:
        sister[:, index::] = shortFV[:, i::]

    return sister

def construct_features(data, filterSlope, tHoldSlope, filterCurvature, 
                       tHoldCurvature, addMidExtension, addEndExtension):
    '''
    Constructs a feature vector for each of the data-series contained in the 
    data. 
    
    '''
    info("calculating features")
    
    # TODO, the casting of each feature to a list of tuples might be 
    # removed at some stage, it will lead to a speed up, for you 
    # can vectorize the calculations that use the feature vector
    features = []
    for i in range(data.shape[0]):
        feature = construct_feature_vector(data[i, :], filterSlope, tHoldSlope, 
                                     filterCurvature, tHoldCurvature, 
                                     addMidExtension, addEndExtension)
#        feature =  [tuple(feature[0,:]),tuple(feature[1,:])]
        features.append(feature)
    return features

def filter_series(series, parentSeries, thold):
    '''
    Filters out a given time-series for insignificant fluctuations. For 
    example very small fluctuations due to numeric error of the simulator).
    '''
    absParent = np.absolute(parentSeries[0:parentSeries.shape[0]-1])
    absSeries = np.absolute(series)
    cond1a = np.not_equal(absParent, 0)
    cond1b = absSeries < thold*absParent
    cond2a = np.logical_not(cond1b)
    cond2b = absSeries < thold/10
    cond1 = np.logical_and(cond1a,cond1b)
    cond2 = np.logical_and(cond2a,cond2b)
    cond = np.logical_or(cond1,cond2)
    series[cond] = 0
    return series

def extend_mids(vector):
    sections = vector[0].size
    added = 0
    for i in range(sections-1):
        if(vector[0][i+added]*vector[0][i+1+added]==-1) and\
          (vector[0][i+added+1]!=0):
            vector = np.insert(vector, i+1+added, 0, axis=1)
            vector[1][i+1+added] = vector[1][i+added]
            added+=1
    return vector

def extend_ends(vector):
    sections = vector[0].size
    added = 0
    
    if(vector[0][0]*vector[1][0]==1):
        #Front extension
        vector = np.insert(vector, 0, 0, axis=1)
        vector[1][0] = vector[1][1]
        added+=1
    if(vector[0][sections-1+added]*vector[1][sections-1+added]==-1):
        #End extension
        vector = np.append(vector, [[0],[0]], axis=1)
        vector[1][sections+added] = vector[1][sections-1+added]
        added+=1
    return vector

def construct_feature_vector(dataSeries, filterSlope, tHoldSlope, 
                             filterCurvature, tHoldCurvature, addMidExtension, 
                             addEndExtension):
    '''
    Constructs a feature vector for the given dataSeries. Each element in 
    this 2-D vector represents a section along the time-series that can be 
    characterized as an atomic behaviour mode.
    '''
    dsLength = dataSeries.shape[0]
    slope = dataSeries[1::]-dataSeries[0:dsLength-1]
    curvature = slope[1::]-slope[0:dsLength-2]
    
    if filterSlope:
        slope = filter_series(slope, dataSeries, tHoldSlope)
    if filterCurvature:
        curvature = filter_series(curvature, slope, tHoldCurvature)
                
    signSlope = slope.copy()
    signSlope[signSlope>0] = 1
    signSlope[signSlope<0] = -1
    signSlope[signSlope==0] = 0
    signSlope = np.delete(signSlope, -1)
    
    signCurvature = curvature.copy()
    signCurvature[signCurvature>0] = 1
    signCurvature[signCurvature<0] = -1
    signCurvature[signCurvature==0] = 0         
    
    sections = 10*signSlope+signCurvature
    temp = sections[1::]-sections[0:sections.shape[0]-1]
    transPoints = np.nonzero(temp)
    numberOfSections = len(transPoints[0])+1
    
    featureVector = np.zeros(shape=(2,numberOfSections))
    
    for k in transPoints:
        featureVector[0][0:len(featureVector[0])-1] = signSlope[k]
        featureVector[1][0:len(featureVector[0])-1] = signCurvature[k]
    featureVector[0][numberOfSections-1]= signSlope[-1]
    featureVector[1][numberOfSections-1]= signCurvature[-1]
 
    
    if addMidExtension:
        featureVector = extend_mids(featureVector)
    if addEndExtension:
        featureVector = extend_ends(featureVector)
    return featureVector

def distance_gonenc(data,
                    sisterCount=50, 
                    wSlopeError=1, 
                    wCurvatureError=1,
                    filterSlope=True,
                    tHoldSlope = 0.1,
                    filterCurvature=True,
                    tHoldCurvature=0.1,
                    addMidExtension=True,
                    addEndExtension=True
                    ):
    
    '''
    The distance measures the proximity of data series in terms of their 
    qualitative pattern features. In order words, it quantifies the proximity 
    between two different dynamic behaviour modes.
    
    It is designed to work mainly on non-stationary data. It's current version 
    does not perform well in catching the proximity of two cyclic/repetitive 
    patterns with different number of cycles (e.g. oscillation with 4 cycle 
    versus oscillation with 6 cycles).
    
    :param data:
    :param sisterCount: Number of long-versions that will be created for the 
                        short vector while comparing two data series with 
                        unequal feature vector lengths. 
    :param wSlopeError: Weight of the error between the 1st dimensions of the 
                        two feature vectors (i.e. Slope). (default=1)
    :param wCurvatureError: Weight of the error between the 2nd dimensions of 
                            the two feature vectors (i.e. Curvature). 
                            (default=1)
    :param wFilterSlope: Boolean, indicating whether the slope vectors should 
                         be filtered for minor fluctuations, or not. 
                         (default=True)
    :param tHoldSlope: The threshold value to be used in filtering out 
                       fluctuations in the slope. (default=0.1)
    :param filterCurvature: Boolean, indicating whether the curvature vectors 
                            should be filtered for minor fluctuations, or not.
                            (default=True)
    :param tHoldCurvature: The threshold value to be used in filtering out 
                           fluctuations in the curvature. (default=0.1)
    :param addMidExtension: Boolean, indicating whether the feature vectors 
                            should be extended by introducing transition 
                            sections along the vector.
                            (default=True)
    :param addEndExtension: Boolean, indicating whether the feature vectors 
                            should be extended by introducing startup/closing 
                            sections at the beginning/end of the vector.
                            (default=True)
    '''
    
    
    runLogs = []
    #Generates the feature vectors for all the time series that are contained 
    # in numpy array data
    features = construct_features(data, filterSlope, tHoldSlope, 
                                  filterCurvature, tHoldCurvature, 
                                  addMidExtension, addEndExtension)
    info("calculating distances")
    dRow = np.zeros(shape=(np.sum(np.arange(data.shape[0])), ))
    index = -1
    for i in range(data.shape[0]):
        feature_i = features[i]
            
        # For each run, a log is created
        # Log includes a description dictionary that has key information 
        # for post-clustering analysis, and the data series itself. These 
        # logs are stored in a global array named runLogs
        behaviorDesc = {}
        behaviorDesc['Index'] = str(i)
        
        #this may not work due to data type mismatch
        featVector = feature_i
        
        behaviorDesc['Feature vector'] = str(featVector)
        behavior = data[i]
        localLog = (behaviorDesc, behavior)
        runLogs.append(localLog)
    
        for j in range(i+1, data.shape[0]):
            index += 1
            feature_j = features[j]
            if feature_i.shape[1] == feature_j.shape[1]:
                distance = distance_same_length(feature_i, feature_j, 
                                                wSlopeError, wCurvatureError)
    
            else:
                distance = distance_different_lenght(feature_i, 
                                                     feature_j, 
                                                     wSlopeError, 
                                                     wCurvatureError, 
                                                     sisterCount)
            dRow[index] = distance
    return dRow, runLogs


if __name__ == '__main__':
    tester = np.array([(0, 1, 4, 8,16,24,30,34,36,39,34,38,34)])
    #tester = np.array([(0, 3, 10,8,16,24,30,34,36,39,34,30,29),(0, 3, 10,8,16,24,30,34,36,39,34,30,29),(0, 3, 10,8,16,24,30,34,36,39,34,30,29)])
    distance_gonenc(tester, addMidExtension=True, addEndExtension=True)
    
    