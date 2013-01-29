'''
Created on Sep 8, 2011

@author: gonengyucel
'''
import time
import cPickle


from some_cluster_tests import cluster as cluster1, flatcluster, plotClusters

from expWorkbench import load_results
from expWorkbench import ema_logging

ema_logging.log_to_stderr(ema_logging.INFO)

clusterSetup = {}
clusterSetup['plotClusters?'] = True
clusterSetup['plotDendrogram?'] = False
clusterSetup['inter-cluster distance'] = 'complete' # Other options are 'single' and 'average'
clusterSetup['cutoff criteria'] = 'inconsistent'   # Other options are 'distance' and 'maxclust' 
clusterSetup['cutoff criteria value'] = 1.154700

distanceSetup = {}
distanceSetup['distance'] = 'gonenc'
distanceSetup['filter?'] = True
distanceSetup['slope filter'] = 0.0001
distanceSetup['curvature filter'] = 0.0005
distanceSetup['no of sisters'] = 50


data = load_results('1000 flu cases no policy.cPickle')

#result = cluster1(data,'deceased population region 1', distanceSetup, clusterSetup)
dRow, clusters, runLogs = cPickle.load(open('clustering.cPickle', 'r'))

cluster, runLogs = flatcluster(dRow, clusterSetup, runLogs)


plotClusters(False, runLogs)