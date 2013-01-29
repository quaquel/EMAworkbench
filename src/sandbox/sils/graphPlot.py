'''
Created on 2 nov. 2011

@author: Bas Keijser
'''

import cPickle
import networkx as NX
import matplotlib.pyplot as plt

file_path = 'model of Oliva.cPickle'
graph = cPickle.load(open(file_path,'r'))
graph.remove_nodes_from(["INITIAL TIME","TIME STEP","SAVEPER","Time","TIME STEP","FINAL TIME"])

fig = plt.figure()
ax = fig.add_subplot(111)

NX.draw_networkx(graph, pos=NX.spring_layout(graph), ax=ax)
plt.show()