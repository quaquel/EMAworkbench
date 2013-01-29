'''
Created on Sep 8, 2011

@author: gonengyucel, jhkwakkel
'''
import matplotlib.pyplot as plt

from analysis.clusterer import cluster

from expWorkbench import load_results
from expWorkbench import EMAlogging

EMAlogging.log_to_stderr(EMAlogging.INFO)

#load the data
data = load_results(r'..\gallery\data\100 flu cases no policy.bz2')

# specify the number of desired clusters
# note: the meaning of cValue is tied to the value for cMethod
cValue = 5

#perform cluster analysis
dRow, clusters, z = cluster(data=data, 
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
for i, cluster in enumerate(clusters):
    #get the data
    values  = data[1]['deceased population region 1']
    values = values[cluster.indices]

    #some index mangling to get correct index for ax
    index = str(cValue) + "1"+str(i)
    index = int(index)
    
    #make an ax
    ax = plt.subplot(index)
    
    #plot data
    ax.plot(data[1]["TIME"].T, values.T, )

plt.savefig("./pictures/cluster_example.png", dpi=75)


