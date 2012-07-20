'''
Created on Sep 8, 2011

@author: gonengyucel
'''

from analysis.clusterer import cluster

from expWorkbench import load_results
from expWorkbench import EMAlogging

EMAlogging.log_to_stderr(EMAlogging.INFO)

data = load_results(r'..\analysis\1000 flu cases.cPickle')

cluster(data=data, 
        outcome='deceased population region 1', 
        distance='gonenc', 
        interClusterDistance='complete', 
        plotDendrogram=True, 
        plotClusters=False, 
        groupPlot=False,
        sisterCount=100,
        )



