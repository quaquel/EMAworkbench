"""

This module provides functions for identifying different behavior clusters 
within the experiment results. It is based on the standard hierarchical 
clustering algorithm of scipy library, which is identical to the clustering 
functions that come with Matlab. For further details about how hierarchical 
clustering works, refer to Matlab documentation. <Here come a link later>. 
The clustering can be based on different distance functions including the 
BM-distance (Behaviour Mode distance) developed by Gonenc.  

"""


import os
import jpype as javap
import matplotlib.pyplot as pyplot
import numpy as np

from scipy.cluster.hierarchy import fcluster, linkage, dendrogram, inconsistent

import clusterPlotter as plotter
import clusteringDistances as distances
import expWorkbench.util as util
from expWorkbench.ema_logging import info, debug

# Global variables
runLogs = []
varName = ""
clusterCount = 0

def cluster(data, outcome, distanceSetup={}, clusterSetup={}):
    '''
    
    Method that clusters time-series data from the specified cpickle file 
    according to a selected distance measure
    
     :param data: return from meth:`perform_experiments`
     :param outcome: Name of outcome/variable whose behavior is being analyzed
     :param distanceSetup: Dictionary that specifies the distance to be used in 
                          clustering, and the configuration of the distance 
                          (optional)  
    :param clusterSetup: Dictionary that specifies the configuration of the 
                         hierarchical clustering algorithm (optional)
    :rtype: A list of integers, which specify the clusters to which runs' 
            results are allocated
    
    The keys that can be specified in the distanceSetup are as follows;
    
    * 'distance': String that specifies the distance to be used. 
                  Options: bmd (default), mse, sse
    * 'filter?': Boolean that specifies whether the data series will be 
                 filtered (for bmd distance)
    * 'slope filter': A float number that specifies the filtering threshold 
                     for the slope (for every data point if change__in_the_
                     outcome/average_value_of_the_outcome < threshold, 
                     consider slope = 0) (for bmd distance)
    * 'curvature filter': A float number that specifies the filtering 
                          threshold for the curvature (for every data point if 
                          change__in_the_slope/average_value_of_the_slope < 
                          threshold, consider curvature = 0) (for bmd distance)
    * 'no of sisters': 50 (for bmd distance)
    
    The keys that can be specified in the clusterSetup are as follows:
    
    * 'plotClusters?': True
    * 'Plot type': 'single-window' #other option is 'single-window'
    * 'plotDendrogram?': True
    * 'inter-cluster distance': 'complete' # Other options are 'single' and 
                                'average'
    * 'cutoff criteria': 'inconsistent'   # Other options are 'distance' and 
                         'maxclust' 
    * 'cutoff criteria value': 0.5
    
    '''
    global varName 
    varName = outcome

    dataSeries = data[1][outcome]
    
    # Construct a n-by-n matrix of distances between behaviors of the selected 
    # outcome. It is a symmetrical matrix, with 0's along the 
    # nortwest-southeast diagonal. distanceSetup is a dictionary that specifies 
    # (and customizes) the distance function to be used.
    distMatrix = constructDMatrix(dataSeries, distanceSetup)
    info('done with distances')
    
    pyplot.ion()
    # Allocate individual runs into clusters using hierarchical agglomerative 
    # clustering. clusterSetup is a dictionary that customizes the clustering 
    # algorithm to be used.
    clusters = flatcluster(distMatrix, clusterSetup)
    
    if 'Plot type' in clusterSetup.keys():
        if clusterSetup['Plot type'] == 'multi-window':
            groupPlot = False
        elif clusterSetup['Plot type'] == 'single-window':
            groupPlot = True
        else:
            groupPlot = False
    else:
        groupPlot = False
    
    # Plots the clusters, unless it is specified not to be done in the setup
    if 'plotClusters?' in clusterSetup.keys():
        if clusterSetup['plotClusters?']:
            plotClusters(groupPlot)
        else:
            pass
    else:
        plotClusters(groupPlot)
    
    return clusters

def constructDMatrix(data, distanceSetup={}):
    """ 
    
    Constructs a n-by-n matrix of distances for n data-series in data 
    according to the specified distance
    
    distance argument specifies the distance measure to be used. Options, 
    which are defined in distances.py, are as follows;
    
    * gonenc: a distance based on qualitative dynamic pattern features 
    * willem: a disance mainly based on the presence of crisis-periods and 
              the overall trend of the data series
    * sse: regular sum of squared errors
    * mse: regular mean squared error
    
    others will be added over time
    
    """

    global runLogs    
    noSeries = len(data)
    
    # Sets up the distance function according to user specification
    if 'distance' in distanceSetup.keys():
        distance = distanceSetup['distance']
    else:
        distance = 'gonenc'
    distFunc = getattr(distances, "distance_%s" %distance)
    
    #dMatrix = []   
    dMatrix = np.zeros((noSeries,noSeries),dtype=float)
    if distance == 'gonenc':
    # gonenc-distance requires interfacing with the java class JavaDistance
    # Java version will be phased out, and it will be migrated into python as 
    # other distance functions
        try:
            javap.startJVM(javap.getDefaultJVMPath())
            print 'started jvm'
        except:
            print "cannot start jvm: try to find jvm.dll"
#           jpype.startJVM(r'C:\Program Files (x86)\Java\jdk1.6.0_22\jre\bin\client\jvm.dll')
        g_dist = javap.JClass('JavaDistance')
        dist = g_dist()
        
        # Checks the parameters of the distance function that may be defined by the user in the distanceSetup dict
        if 'filter?' in distanceSetup.keys():
            dist.setWithFilter(distanceSetup['filter?'])
        if 'slope filter' in distanceSetup.keys():
            dist.setSlopeFilterThold(distanceSetup['slope filter'])
        if 'curvature filter' in distanceSetup.keys():
            dist.setCurvatureFilterThold(distanceSetup['curvature filter'])
        if 'no of sisters' in distanceSetup.keys():
            dist.setSisterCount(distanceSetup['no of sisters'])
                
        # Calculate the distances and fill in the distance matrix. 
        # i is the index for each run, and is in sync with the indexing used in 
        # the cases, and results output structures.
        runLogs = []
        for i in range(noSeries):
            #dMatrix.append([])
            
            # For each run, a log is created
            # Log includes a description dictionary that has key information 
            # for post-clustering analysis, and the data series itself. These 
            # logs are stored in a global array named runLogs
            behaviorDesc = {}
            behaviorDesc['Index'] = str(i)
            featVector = dist.getFeatureVector(data[i]) #this may not work due to data type mismatch
            
            behaviorDesc['Feature vector'] = str(featVector)
            behavior = data[i]
            localLog = (behaviorDesc, behavior)
            runLogs.append(localLog)
          
            for j in range(noSeries):
                dMatrix[i,j]=dist.distance(data[i],data[j])
                #dMatrix[i].append(dist.distance(data[i], data[j]))
        javap.shutdownJVM()   
    elif distance == "willem":
        try:
            javap.startJVM(javap.getDefaultJVMPath())
        except:
            print "cannot start jvm: try to find jvm.dll"
#           jpype.startJVM(r'C:\Program Files (x86)\Java\jdk1.6.0_22\jre\bin\client\jvm.dll')
        w_dist = javap.JClass('WillemDistance')
        dist = w_dist()
        runLogs = []
        for i in range(noSeries):
            #dMatrix.append([])
            
            # For each run, a log is created
            # Log includes a description dictionary that has key information for post-clustering analysis, and the data series itself
            # These logs are stored in a global array named runLogs
            behaviorDesc = {}
            behaviorDesc['Index'] = str(i)
            featVector = dist.getFeatureVector(data[i]) #this may not work due to data type mismatch
            behaviorDesc['Feature vector'] = str(featVector)
            behavior = data[i]
            localLog = (behaviorDesc, behavior)
            runLogs.append(localLog)
          
            for j in range(noSeries):
                dMatrix[i,j]=dist.distance(data[i],data[j])
                #dMatrix[i].append(dist.distance(data[i], data[j]))
        javap.shutdownJVM()      
    
    else:    
        for i in range(noSeries):
            #dMatrix.append([])
            
            # For each run, a log is created
            # Log includes a description dictionary that has key information for post-clustering analysis, and the data series itself
            # These logs are stored in a global array named runLogs
            behaviorDesc = {}
            behaviorDesc['Index'] = i
            behavior = data[i]
            localLog = (behaviorDesc, behavior)
            runLogs.append(localLog)
            
            for j in range(noSeries):
                dMatrix[i,j]=dist.distance(data[i],data[j])
                #dMatrix[i].append(distFunc(data[i], data[j])) 
    return dMatrix

def prepareDRow(dMatrix):
    """ 
    
    Converts the distance matrix to to a reduced-form distance row that is 
    required for clustering algorithm
    
    """
    dRow = []
    noSeries = len(dMatrix)
    
    for i in range(noSeries):
        for j in range(noSeries):
            if j>i:
                dRow.append(dMatrix[i,j])
    return dRow

def flatcluster(dMatrix, clusterSetup):
    dRow = prepareDRow(dMatrix)
    print dRow
    
    # Checking user-specified options, if there is any. Otherwise the default 
    # values are assigned
    if 'inter-cluster distance' in clusterSetup.keys():
        method = clusterSetup['inter-cluster distance']
    else:
        method = 'complete'
    z = linkage(dRow, method)
    inc = inconsistent(z)
    print inc
    
    if 'plotDendrogram?' in clusterSetup.keys():
        if clusterSetup['plotDendrogram?']:
            plotdendrogram(z)
        else:
            pass
    else:
        plotdendrogram(z)
    
    if 'cutoff criteria' in clusterSetup.keys():
        cmethod = clusterSetup['cutoff criteria']
    else:
        cmethod = 'inconsistent'
    
    if 'cutoff criteria value' in clusterSetup.keys():
        cvalue = clusterSetup['cutoff criteria value']
    else:
        cvalue = 2.5
    
    clusters = fcluster(z, cvalue, cmethod)
    
    noClusters = max(clusters)
    print 'Total number of clusters:', noClusters
    for i in range(noClusters):
        counter = 0
        for j in range(len(clusters)):
            if clusters[j]==(i+1):
                counter+=1
        print "Cluster",str(i+1),":",str(counter)
    
    for runIndex in range(len(clusters)):
        global runLogs
        runLogs[runIndex][0]['Cluster'] = str(clusters[runIndex])
        global clusterCount
        if clusters[runIndex] > clusterCount:
            clusterCount = clusters[runIndex]
    return clusters
           
def plotdendrogram(z):
    dendrogram(z)
    pyplot.draw()

def plotClusters(groupPlot):
        
    # In order to show multiple plots at a time, multi-processing is used
    # Multi-processing fails in Mac OS X, therefore it is deactivated for that os
    if os.name == 'posix':
        multiprocessing = False
    else:
        multiprocessing = True
    
        
    global varName
    global runLogs
    global clusterCount
    noRuns = len(runLogs)
    processes = []
    
    clustersToPlot = []
    #print 'Cluster Sayisi ', clusterCount
    # For each cluster, create a data structure
    for clust in range(clusterCount):    
        runSubset = []
        for runIndex in range(noRuns):
            if runLogs[runIndex][0]['Cluster']==str(clust+1):
                runSubset.append(runLogs[runIndex])
        
        # Dumps data about each cluster to a different cpickle file
        if groupPlot:
            clustersToPlot.append(runSubset)
        else:
            callSinglePlotter(runSubset)
            #if multiprocessing:
            #    p = Process(target=callSinglePlotter, args=(runSubset,))
            #    processes.append(p)
            #    p.start()
            #else:
            #    callSinglePlotter(runSubset)
    
    if groupPlot:
        callGroupPlotter(clustersToPlot)
#        if multiprocessing:
#            p = Process(target=callGroupPlotter, args=(clustersToPlot,))
#            processes.append(p)
#            p.start()
#        else:
#            callGroupPlotter(clustersToPlot)
#        
   
def callSinglePlotter(data):
    plt = plotter.singlePlot()
    plt.setData(data)
    global varName
    plt.setVarName(varName)
    plt.configure_traits()

def callGroupPlotter(data):
    plt = plotter.groupPlot()
    plt.setData(data)
    global varName
    plt.setVarName(varName)
    plt.configure_traits()  
          
     
        
def jpypeTester():
    javap.startJVM(javap.getDefaultJVMPath())
    javap.java.lang.System.out.println("Deneme")
    javap.shutdownJVM()               

def compareClusterSets(cSetA, cSetB):
    if len(cSetA) != len(cSetB):
        print "Number of data series do not match! Comparison failed"
    else:
        ss = 0.0
        sd = 0.0
        ds = 0.0
        dd = 0.0
        noDataSeries = len(cSetA)
        matchA = np.zeros((noDataSeries,noDataSeries), dtype=int)
        matchB = np.zeros((noDataSeries,noDataSeries), dtype=int)
        
        for i in range(noDataSeries):
            for j in range(i+1):
                if cSetA[i]==cSetA[j]:
                    matchA[i,j]=1
                    matchA[j,i]=1
                else:
                    matchA[i,j]=0
                    matchA[j,i]=0
                
        for i in range(noDataSeries):
            for j in range(i+1):
                if cSetB[i]==cSetB[j]:
                    matchB[i,j]=1
                    matchB[j,i]=1
                else:
                    matchB[i,j]=0
                    matchB[j,i]=0
                
        for i in range(noDataSeries):
            for j in range(i+1):
                if (matchA[i,j]==1 and matchB[i,j]==1):
                    ss+=1
                elif (matchA[i,j]==1 and matchB[i,j]==0):
                    sd+=1
                elif (matchA[i,j]==0 and matchB[i,j]==1):
                    ds+=1
                elif (matchA[i,j]==0 and matchB[i,j]==0):
                    dd+=1
            
        rand = (ss+dd)/(ss+dd+sd+ds)
        jaccard = (ss)/(ss+sd+ds)
        print rand 
        print jaccard
        simIndices = {'rand':rand,'jaccard':jaccard}
        return simIndices
    
    


if __name__ == '__main__':
    clusterSetup = {}
    clusterSetup['plotClusters?'] = False
    clusterSetup['Plot type'] = 'single-window' #other option is 'single-window'
    clusterSetup['plotDendrogram?'] = False
    clusterSetup['inter-cluster distance'] = 'single' # Other options are 'complete', 'single' and 'average'
    clusterSetup['cutoff criteria'] = 'inconsistent'   # Other options are 'distance' and 'maxclust' 
    clusterSetup['cutoff criteria value'] = 0.5
    
    distanceSetup = {}
    distanceSetup['distance'] = 'gonenc'
    distanceSetup['filter?'] = True
    distanceSetup['slope filter'] = 0.001
    distanceSetup['curvature filter'] = 0.005
    distanceSetup['no of sisters'] = 20
    
    
    
    #cluster('chacoTest.cpickle', 'total population', distance='gonenc',plotClusters=True)
    cSet = cluster('PatternSet_Basics.cpickle', 'outcome', distanceSetup, clusterSetup)
    print max(cSet)
    
        
    cases, results = util.load_results('PatternSet_Basics.cpickle')
    cSetActual = cases['Class ID']
    
    compareClusterSets(cSet,cSetActual)   
    #jpypeTester()
    #multiprocessTester()
    for i in range(len(cSet)):
        print cSet[i]
  