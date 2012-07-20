'''
Created on 30 nov. 2011

@author: chamarat
'''

from expWorkbench.util import load_results
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec 

results = load_results('storedresults_2.cPickle')
results = np.asarray(results)
results = results[:,0:4]


grid = gridspec.GridSpec(len(results[0]), len(results[0]))     
figure = plt.figure()

#FIELDS = ["ini cap T1" , "ini cap T2" , "ini cap T3" , "ini cap T4" , "ini cost T1" , "ini cost T2" , "ini cost T3" , "ini cost T4" , "ini cum decom cap T1" , "ini cum decom cap T2" , "ini cum decom cap T3" , "ini cum decom cap T4" , "average planning and construction period T1" , "average planning and construction period T2" , "average planning and construction period T3" , "average planning and construction period T4" , "ini PR T1" , "ini PR T2" , "ini PR T3" , "ini PR T4" , "lifetime T1" , "lifetime T2" , "lifetime T3" , "lifetime T4" , "ec gr t1" , "ec gr t2" , "ec gr t3" , "ec gr t4" , "ec gr t5" , "ec gr t6" , "ec gr t7" , "ec gr t8" , "ec gr t9" , "ec gr t10" , "random PR min" , "random PR max" , "seed PR T1" , "seed PR T2" , "seed PR T3" , "seed PR T4" , "absolute preference for MIC" , "absolute preference for expected cost per MWe" , "absolute preference against unknown" , "absolute preference for expected progress" , "absolute preference against specific CO2 emissions" , "SWITCH preference for MIC" , "SWITCH preference for expected cost per MWe" , "SWITCH preference against unknown" , "SWITCH preference for expected progress" , "SWITCH preference against specific CO2 emissions" , "performance expected cost per MWe T1" , "performance expected cost per MWe T2" , "performance expected cost per MWe T3" , "performance expected cost per MWe T4" , "performance CO2 avoidance T1" , "performance CO2 avoidance T2" , "performance CO2 avoidance T3" , "performance CO2 avoidance T4" , "SWITCH T3" , "SWITCH T4"]

combis = [(field1, field2) for field1 in range(len(results[0])) for field2 in range(len(results[0]))]
for field1, field2 in combis:
    i = field1
    j = field2
    ax = figure.add_subplot(grid[i,j])
    
    data1 = results[:-1,i]
    data2 = results[:-1,j]

    ax.scatter(data2, data1)
  
plt.show()

    