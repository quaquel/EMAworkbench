'''
Created on Sep 8, 2011

.. codeauthor:: 
     gyucel <g.yucel (at) tudelft (dot) nl>,
     jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
                

a reworking of the cluster. The distance metrics have now their own .py file. 
The metrics available are currently stored in the distance_functions 
dictionary.

'''
from __future__ import division
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram, inconsistent
from scipy.spatial.distance import squareform
import numpy as np

from expWorkbench.EMAlogging import info
from expWorkbench import EMAError
from distance_gonenc import distance_gonenc
from distance_willem import distance_willem
from distance_triangle import distance_triangle
from distance_sse import distance_sse
from distance_mse import distance_mse


import clusterPlotter as plotter


distance_functions = {'gonenc': distance_gonenc,
                      'willem': distance_willem,
                      'triangle':distance_triangle,
                      'sse': distance_sse,
                      'mse': distance_mse}

# Global variables
runLogs = []
varName = ""
clusterCount = 0

def cluster(data, 
            outcome,
            distance='gonenc',
            interClusterDistance='complete',
            cMethod='inconsistent',
            cValue=2.5,
            plotDendrogram=True,
            plotClusters=True,
            groupPlot=False,
            **kwargs):
    '''
    
    Method that clusters time-series data from the specified cpickle file 
    according to a selected distance measure.
    
    :param data: return from meth:`perform_experiments`.
    :param outcome: Name of outcome/variable whose behavior is being analyzed
    :param distance: The distance metric to be used.
    :param interClusterDistance: How to calculate inter cluster distance.
                                 see `linkage <http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage>`_ 
                                 for details.
    :param cMethod: Cutoff method, 
                    see `fcluster <http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster>`_ 
                    for details.
    :param cValue: Cutoff value, see 
                   `fcluster <http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster>`_ 
                   for details.
    :param plotDendogram: Boolean, if true, plot dendogram.
    :param plotCluster: Boolean, true if you want to plot clusters.
    :param groupPlot: Boolean, if true plot clusters in a single window, 
                      else the clusters are plotted in separate windows.
    :rtype: A tuple containing the list of distances, the cluster allocation, 
            and a list of logged distance metrics for each time series.     
    
    The remainder of the arguments are passed on to the specified distance 
    function. See the distance functions for details on these parameters.
    
    '''
    
    global varName 
    varName = outcome
    
    data = data[1][outcome]
    
    # Construct a list with distances. This list is the upper triange
    # of the distance matrix
    dRow, runLogs = construct_distances(data, distance, **kwargs)
    info('finished distances')
    

    # Allocate individual runs into clusters using hierarchical agglomerative 
    # clustering. clusterSetup is a dictionary that customizes the clustering 
    # algorithm to be used.
    z, clusters, runLogs = flatcluster(dRow, 
                                    runLogs, 
                                    plotDendrogram=plotDendrogram,
                                    interClusterDistance=interClusterDistance,
                                    cMethod=cMethod,
                                    cValue=cValue)
    
    
    sample_indices = pick_csamples(clusters, dRow)
    
#    if 'Plot type' in clusterSetup.keys():
#        if clusterSetup['Plot type'] == 'multi-window':
#            groupPlot = False
#        elif clusterSetup['Plot type'] == 'single-window':
#            groupPlot = True
#        else:
#            groupPlot = False
#    else:
#        groupPlot = False
    
    # Plots the clusters, unless it is specified not to be done in the setup
#    if 'plotClusters?' in clusterSetup.keys():
#        if clusterSetup['plotClusters?']:
#            plotClusters(groupPlot, runLogs)
#        else:
#            pass
    if plotClusters:
        plot_clusters(groupPlot, runLogs)
    
    return dRow, clusters, runLogs, z

   

def construct_distances(data, distance='gonenc', **kwargs):
    """ 
        
    Constructs a n-by-n matrix of distances for n data-series in data 
    according to the specified distance.
    
    Distance argument specifies the distance measure to be used. Options, 
    which are defined in clusteringDistances.py, are as follows.
    
    
    * gonenc: a distance based on qualitative dynamic pattern features 
    * willem: a disance mainly based on the presence of crisis-periods and 
              the overall trend of the data series
    * sse: regular sum of squared errors
    * mse: regular mean squared error
    
    
    SSE and MSE are in clusterinDistances.py and don't work right now.
    
    others will be added over time
    
    """
    
    # Sets up the distance function according to user specification
    try:
        return distance_functions[distance](data, **kwargs)
    except KeyError:
        raise EMAError("trying to use an unknown distance: %s" %(distance))

def flatcluster(dRow, runLogs, 
                interClusterDistance='complete',
                plotDendrogram=True,
                cMethod='inconsistent',
                cValue=2.5):
#    if 'inter-cluster distance' in clusterSetup.keys():
#        method = clusterSetup['inter-cluster distance']
#    else:
#        method = 'complete'

    z = linkage(dRow, interClusterDistance)
    inc = inconsistent(z)
    print inc
    
    if plotDendrogram:
        plotdendrogram(z)
    
    clusters = fcluster(z, cValue, cMethod)
    
    noClusters = max(clusters)
    print 'Total number of clusters:', noClusters
    for i in range(noClusters):
        counter = 0
        for j in range(len(clusters)):
            if clusters[j]==(i+1):
                counter+=1
        print "Cluster",str(i+1),":",str(counter)
    
    global clusterCount
    clusterCount = noClusters
    print len(clusters)
    print len(runLogs)
    for i, log in enumerate(runLogs):
        log[0]['Cluster'] = str(clusters[i])
    

    return z, clusters, runLogs
           
def plotdendrogram(z):
    dendrogram(z)
    plt.show()

def plot_clusters(groupPlot, runLogs):
    global varName
    global clusterCount
    noRuns = len(runLogs)
    
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
   
    if groupPlot:
        callGroupPlotter(clustersToPlot)
   
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

if __name__ == '__main__':
    print 'tester'




