'''

Created on Dec 7, 2011

@author: jhkwakkel
'''
import networkx as nx
import matplotlib.pyplot as plt
import cPickle
from expWorkbench.vensimDLLwrapper import VensimWarning

try:
    from networkx import graphviz_layout
except ImportError:
    raise ImportError("This example needs Graphviz and either PyGraphviz or Pydot")


from expWorkbench import vensim
from expWorkbench import vensimDLLwrapper as venDLL

vensim.load_model(r'C:\workspace\EMA-workbench\src\sandbox\sils\MODEL.vpm')

vars = venDLL.get_varnames()

graph = nx.DiGraph()
graph.add_nodes_from(vars)

for var in vars:
    try:
        causes = venDLL.get_varattrib(var, attribute=4) #cause
        for cause in causes:
            graph.add_edge(cause, var)
    except VensimWarning:
        print var

cPickle.dump(graph, open("model of Oliva.cPickle",'wb'))

pos=nx.graphviz_layout(graph,prog='twopi',args='')
nx.draw_networkx(graph, pos=pos)
plt.show()