'''

a reworking of the cluster. The distance metrics have now their own .py file. 
The metrics available are currently stored in the distance_functions 
dictionary.

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram, inconsistent
from scipy.spatial.distance import squareform

from .cluster_util import (distance_mse, distance_sse, distance_triangle,
                           distance_gonenc)
from ..util import info, EMAError

# Created on Sep 8, 2011
# 
# .. codeauthor:: 
#      gyucel <g.yucel (at) tudelft (dot) nl>,
#      jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
#                 

distance_functions = {'gonenc': distance_gonenc,
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
    
    Parameters
    ----------
    
    data : tuple
           return from meth:`perform_experiments`.
    outcome : str
              Name of outcome/variable whose behavior is being analyzed
    distance : {'gonenc','triangle', 'sse', 'mse'}
               The distance metric to be used.
    interClusterDistance : str
                           How to calculate inter cluster distance.
                           see `linkage <http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage>`_ 
                           for details.
    cMethod : str
              Cutoff method, see `fcluster <http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster>`_ 
              for details.
    cValue : float
             Cutoff value, see `fcluster <http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster>`_ 
             for details.
    plotDendogram : bool
    plotCluster : bool
    groupPlot: bool
    
    Returns
    -------
    list
        distances
    list
        Clusters
    list
        distance metrics
    
    The remainder of the arguments are passed on to the specified distance 
    function.
    
    Gonenc Distance:
    
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
    
    info("tranforming to list of clusters")
    clusters = make_data_structure(clusters, dRow, runLogs)

    if plotClusters:
        plot_clusters(groupPlot, runLogs)
    
    return dRow, clusters, z

def make_data_structure(clusters, distRow, runLogs):
    nr_clusters = np.max(clusters)
    cluster_list = []
    for i in range(1, nr_clusters+1):
        info("starting with cluster %s" %i)
        #determine the indices for cluster i
        indices = np.where(clusters==i)[0]
        
        drow_indices = np.zeros((indices.shape[0]**2-indices.shape[0])/2, dtype=int)
        s = 0
        #get the indices for the distance for the runs in the cluster
        for q in range(indices.shape[0]):
            for r in range(q+1, indices.shape[0]):
                b = indices[q]
                a = indices[r]
                
                drow_indices[s] = get_drow_index(indices[r],
                                                 indices[q], 
                                                 clusters.shape[0])
                s+=1
        
        #get the distance for the runs in the cluster
        dist_clust = distRow[drow_indices]
        
        #make a distance matrix
        dist_matrix = squareform(dist_clust)

        #sum across the rows
        row_sum = dist_matrix.sum(axis=0)
        
        #get the index of the result with the lowest sum of distances
        min_cIndex = row_sum.argmin()
    
        # convert this cluster specific index back to the overall cluster list 
        # of indices
        originalIndices = np.where(clusters==i)
        originalIndex = originalIndices[0][min_cIndex]

        print(originalIndex)

        a = list(np.where(clusters==i)[0])
        a = [int(entry) for entry in a]
        
        cluster = Cluster(i, 
                          np.where(clusters==i)[0], 
                          originalIndex,
                          [runLogs[entry] for entry in a],
                          dist_clust)
        cluster_list.append(cluster)
    return cluster_list

def get_drow_index(i,j, size):
    '''
    Get the index in the distance row for the distance between i and j.
    
    Parameters
    ----------
    i : result i
    j : result j
    size : the number of results
    
    Returns
    -------
    int
    
    
    ...note:: i > j
    
    '''
    assert i > j

    index = 0
    for q in range(size-j, size):
        index += q
    index = index+(i-(1*j))-1

    return index

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
    #print inc
    
    if plotDendrogram:
        plotdendrogram(z)
    
    clusters = fcluster(z, cValue, cMethod)
    
    noClusters = max(clusters)
    print('Total number of clusters:', noClusters)
    for i in range(noClusters):
        counter = 0
        for j in range(len(clusters)):
            if clusters[j]==(i+1):
                counter+=1
        print("Cluster",str(i+1),":",str(counter))
    
    global clusterCount
    clusterCount = noClusters
    print(len(clusters))
    print(len(runLogs))
    for i, log in enumerate(runLogs):
        log[0]['Cluster'] = str(clusters[i])
    

    return z, clusters, runLogs
           
def plotdendrogram(z):
    
    dendrogram(z,
               truncate_mode='lastp',
               show_leaf_counts=True,
               show_contracted=True
               )
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
        
   
class Cluster(object):
    '''
    Contains information about a data-series cluster, as well as some methods to help analyzing a cluster.
    Basic attributes of a cluster (e.g. c) object are as follows;
     
     * c.no : Cluster number/index
     * c.indices : Original indices of the dataseries that are in cluster c
     * c.sample : Original index of the dataseries that is the representative of cluster c (i.e. median element of the cluster)
     * c.size : Number of elements (i.e. dataseries) in the cluster c
    '''

    def __init__(self, 
                 cluster_no, 
                 all_ds_indices, 
                 sample_ds_index,
                 runLogs,
                 dist_clust):
        '''
        Constructor
        '''
        self.no = cluster_no
        self.indices = all_ds_indices
        self.sample = int(sample_ds_index)
        self.size = self.indices.size
        self.runLogs = runLogs
        self.distances = dist_clust
        
        
    def error(self):
        return self.sample

if __name__ == '__main__':
    from ..util import load_results
    results = load_results('../sandbox/cluster/datasets/PatternSet_Basics.cPickle')
    
    
    distance, liste, obekler, kosulog, zet = cluster(results, 'outcome',cMethod='distance', cValue=1, groupPlot=True, plotDendrogram=False)
    
    #clusters = np.array([3, 1, 2, 2, 1,2])
    #dRow = np.array([1,2,3,6,2,1,0,3,9,6,2,1,3,2,1])
    #samples = pick_cSamples(clusters, dRow)
    #print "ornekler", samples
 
    
        
    


