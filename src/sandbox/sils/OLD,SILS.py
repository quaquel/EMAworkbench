'''
Created on January 2012 - March 2012

@author: Bas M.J. Keijser
'''
import cPickle
import networkx as nx
import Partitions as prt
import numpy as np

import matplotlib.pyplot as plt

def cyclePartition(graph):
    a = prt.adj(graph)
    n = graph.nodes()
    r = prt.reach(a)
    l = prt.levels(r,n)
    c = prt.cycles(graph, l)
    return c

def cycleGraph(graph):
    c = cyclePartition(graph)
    graphs = []
    for entry in c:
        sub = graph.subgraph(entry)
        graphs.append(sub)
#        plot(sub)
    return graphs

def plot(graph):

    pos=nx.graphviz_layout(graph,prog='twopi',args='')
    nx.draw_networkx(graph, pos=pos)
    plt.show()

def distance(graph): 
    #Must be bettered if more than one cycle, works on Yeast
    adj = prt.adj(graph);
    B = prt.binAddI(adj); D = prt.binSub(B, np.identity(len(adj),float))
    prevS = B
    for i in range(2,len(adj),1):
        curS = prt.binPrd(B, prevS)
        D = D + i*(curS-prevS)
        prevS = curS
    return D

def predDistance(n,s,distance, graph): 
    # Identifies predecessors of node n that are s steps away 
    # (from distance matrix D)
    d = distance[:,n]; ind = [] 
    nodes = graph.nodes()
    for i in range(0,len(d),1):
        if d[i] == s:
            ind.append(nodes[i])
    return ind

def succDistance(n,s,distance, graph):
    d = np.asarray(distance[n])
    d = d[0]
    ind = []
    nodes = graph.nodes()
    for i in range(0,len(d),1):
        if d[i] == s:
            ind.append(nodes[i])
    return ind

def loopTrack(sourceDest,graph): 
    #sourceDest contains source and destination which are node names
    D = distance(graph)
    n = graph.nodes()
    u = n.index(sourceDest[0])
    v = n.index(sourceDest[1]) 
    # convert node names to indices in cycle
    loop = [sourceDest[0]]
    end = D[u,v]-1; end = int(end)
    # start loop with source and calculate length of source to destination
    for i in range(1,end+1,1): # the source to destination path
        rh = predDistance(v,D[u,v]-i,D, graph)
        lh = succDistance(n.index(loop[len(loop)-1]),1,D, graph)
        for item in lh:
            if item in rh:
                loop.append(item)
                break 
                # This break is the cause of the non-uniqueness of this 
                # function's results
    loop.append(sourceDest[1])
    end = D[v,u]-1
    end = int(end)
    for i in range(1,end+1,1):
        # the destination back to source path
        rh = predDistance(u,D[v,u]-i,D, graph)
        lh = succDistance(n.index(loop[len(loop)-1]),1,D, graph)
        for item in lh:
            if item in rh:
                loop.append(item)
                break
    return loop

def geodetic(graph):
    D = distance(graph)
    L = np.transpose(np.tril(D))+np.triu(D)
    loops = []
    nodes = graph.nodes()
    for i in range(2,int(np.max(L)+1)):
        indices = np.argwhere(L==i)
        pairs = []
        for entry in indices:
            pairs.append([nodes[entry[0,0]],nodes[entry[0,1]]])
        already_seen = []
        for j in range(len(pairs)):
            if j in already_seen:
                continue
            item = pairs[j]
            loop = loopTrack(item,graph)
            for k, item in enumerate(pairs):
                if item[0] and item[1] in loop:
                    already_seen.append(k)
            
            if len(set(loop)) == len(loop):
                loops.append(loop)
    return loops

def SILS(graph):
    A = prt.adj(graph)
    A = np.asarray(A)
    loops = geodetic(graph)
    nodes = graph.nodes()
    
    #make an adjacency matrix for the loop
    edges = [] #list of adjacency matrices for each loop
    for loop in loops:
        # For every loop, add edges to a matrix in adjacency-style
        l = len(loop)
        E = np.zeros(A.shape)
        for j in range(0,l-1):
            sourceName = loop[j]
            destName = loop[j+1]
            source = nodes.index(sourceName)
            dest = nodes.index(destName)
            E[source,dest] = 1
        firstName = loop[0]
        lastName = loop[l-1]
        first = nodes.index(firstName)
        last = nodes.index(lastName)
        E[last,first] = 1
        edges.append(E)
    
    B = np.zeros(A.shape);
    S = loops
    SILS = []
    # Here comes the construction of the SILS
    while S:
        toRemove = []
        
        #figure out which loops do not contribute any new edges
        
        copy_s = S[:]
        edges_copy = []
        for j, element in enumerate(edges):
            # Counting the number of contributions the loop can introduce
            entry = S[j]
            test = element - B
            test[test<0] = 0
            new_edges = np.sum(test)
            if new_edges==0:
                copy_s.remove(entry)
            else:
                edges_copy.append(element)
        S = copy_s
        edges = edges_copy
        
        #of the remaining loops, determine the contribution of edges
        numberOfNewEdges = []
        for element in edges:
            test = element - B
            test[test<0] = 0
            new_edges = int(np.sum(test))
            numberOfNewEdges.append(new_edges)
        
        if numberOfNewEdges:
            # The loop with minimum contribution should be added
            m = numberOfNewEdges.index(min(numberOfNewEdges))
    
            # Add edges to the SILS edge collection
            B = B + edges[m]
            SILS.append(S[m])
            
            # Remove the items from the loop collection
            S.pop(m)
            edges.pop(m)

    return SILS

if __name__ == "__main__":
    '''
    The following lines should be copied to a new file,
    this to alter the model or its graph into a form that can be used in the 
    SILS-routine developed.
    '''
    # The graph object which is used, is build and we get rid of the 
    # "nonsense" variables
#    file_path = r'C:\workspace\EMA-workbench\src\sandbox\sils\minerals and metals network.cPickle'
    file_path = r'model of Oliva.cPickle'
    graph = cPickle.load(open(file_path, 'r'))
    graph.remove_nodes_from(["INITIAL TIME","TIME STEP","SAVEPER","Time",
                             "TIME STEP","FINAL TIME"])    
    
    cycleGraphs = cycleGraph(graph)
    for graph in cycleGraphs:
        sils = SILS(graph)
        for entry in sils:
            print entry