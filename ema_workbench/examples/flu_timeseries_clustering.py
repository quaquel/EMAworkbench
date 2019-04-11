'''
Created on 11 Apr 2019

@author: jhkwakkel
'''
import matplotlib.pyplot as plt
import seaborn as sns

from ema_workbench import load_results
from ema_workbench.analysis import clusterer, plotting, Density


experiments, outcomes = load_results('./data/1000 flu cases no policy.tar.gz')
data = outcomes['infected fraction R1']

# calcuate distances
distances = clusterer.calculate_cid(data)

# plot dedrog
clusterer.plot_dendrogram(distances)

# do agglomerative clustering on the distances
clusters = clusterer.apply_agglomerative_clustering(distances,
                                                    n_clusters=5)

# show the clusters in the output space
x = experiments.copy()
x['clusters'] = clusters.astype('object')
plotting.lines(x, outcomes, group_by='clusters',
               density=Density.BOXPLOT)

# show the input space
sns.pairplot(x, hue='clusters',
             vars=['infection ratio region 1', 'root contact rate region 1',
                   'normal contact rate region 1', 'recovery time region 1',
                   'permanent immune population fraction R1'],
             plot_kws=dict(s=7))
plt.show()
