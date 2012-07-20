'''
Created on 30 nov. 2011

@author: chamarat
'''

from expWorkbench.util import load_results
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats.kde as kde 

results = load_results('storeduncertainties.cPickle')

results = np.asarray(results)

fig = plt.figure()
ax = fig.add_subplot(111)

##For a KDE graph
#results = results[:-1,20]
#ymin = np.min(results)
#ymax = np.max(results)
#line = np.linspace(ymin, ymax, 1000)[::-1]
#b = kde.gaussian_kde(results)
#b = b.evaluate(line)
#b = np.log((b+1))
#ax.plot(b, line)

##Normal histogram graph
#results = results[:-1,20]
#ax.hist(results,30)

#A regular scatter graph 
myarray = results[:-1,21]
plt.scatter(range(len(myarray)), myarray)

plt.show()

