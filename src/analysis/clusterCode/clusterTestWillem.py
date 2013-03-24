'''
Created on Sep 8, 2011

@author: gonengyucel
'''
import time

#from clusterer import cluster

from clustererV2 import cluster

from expWorkbench import load_results
from expWorkbench import ema_logging

ema_logging.log_to_stderr(ema_logging.INFO)


data = load_results(r'C:\workspace\ema_hacking\models\cPickle\result4500copperBottumUp.cPickle')



#dRow, clusters, runLogs = cluster(data, 
#                          outcome='Real price of copper', 
#                          distance='willem',
#                          interClusterDistance='average', 
#                          cMethod='maxclust', 
#                          cValue=15, 
#                          plotDendrogram=False, 
#                          plotClusters=True, 
#                          groupPlot=True, 
#                          trendThold=0.0001, 
#                          crisisThold=0.25,
#                          wIfCrisis=1,
#                          wNoOfCrises=0.7,
#                          wTrend=0.5,
#                          wBandwith=0.6,
#                          wSevCrises=0.2)

#for element in dRow:
#    print element

