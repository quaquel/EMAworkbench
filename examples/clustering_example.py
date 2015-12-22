'''
Created on Sep 8, 2011

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
                gonengyucel

'''
import matplotlib.pyplot as plt

from ema_workbench.analysis.clusterer import cluster

from ema_workbench.util import ema_logging, load_results

ema_logging.log_to_stderr(ema_logging.INFO)

#load the data
data = load_results(r'..\examples\100 flu cases no policy.cPickle')

# specify the number of desired clusters
# note: the meaning of cValue is tied to the value for cMethod
cValue = 5

#perform cluster analysis
dist, clusteraloc, runlog, z = cluster(data=data, 
                                    outcome='deceased population region 1', 
                                    distance='gonenc', 
                                    interClusterDistance='complete', 
                                    cMethod = 'maxclust',
                                    cValue = cValue,
                                    plotDendrogram=False, 
                                    plotClusters=False, 
                                    groupPlot=False,
                                    sisterCount=100,
                                    tHoldCurvature = 0.1,
                                    tHoldSlope = 0.1
                                    )

#the plotting
fig = plt.figure()

#for each cluster
for i in range(1, cValue+1):
    #get the data
    values  = data[1]['deceased population region 1']
    values = values[clusteraloc==i]

    #some index mangling to get correct index for ax
    index = str(cValue) + "1"+str(i)
    index = int(index)
    
    #make an ax
    ax = plt.subplot(index)
    
    #plot data
    ax.plot(data[1]["TIME"].T, values.T, )

plt.show()