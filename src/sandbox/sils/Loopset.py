'''
Created in January 2012 - March 2012

@author: Bas M.J. Keijser
'''
import cPickle
import networkx as nx
import Partitions as prt
import numpy as np
import LoopKO
import matplotlib.pyplot as plt


def cycle_partition(graph):
    a = prt.adj(graph)
    n = graph.nodes()
    r = prt.reach(a)
    l = prt.levels(r,n)
    c = prt.cycles(graph, l)

    return c


def cycle_graph(graph):
    c = cycle_partition(graph)
    graphs = []
    
    for entry in c:
        sub = graph.subgraph(entry)
        graphs.append(sub)
        
    return graphs


def graph_plot(graph):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    nx.draw_networkx(graph, pos=nx.spring_layout(graph), ax=ax)
    plt.show()

    
def distance(graph): 
    adj = prt.adj(graph);
    B = prt.binary_add_identity(adj)
    D = prt.binary_substraction(B, np.identity(len(adj),float))
    prev_S = B
    
    for i in range(2,len(adj),1):
        cur_S = prt.binary_product(B, prev_S)
        D = D + i*(cur_S-prev_S)
        prev_S = cur_S
        
    return D


def pred_distance(n,s,distance,graph):
    # Identifies predecessors of node n that are s steps away (from distance matrix D).
    d = distance[:,n]
    ind = [] 
    nodes = graph.nodes()
    
    for i in range(0,len(d),1):
        if d[i] == s:
            ind.append(nodes[i])
            
    return ind


def succ_distance(n,s,distance,graph):
    d = np.asarray(distance[n])
    d = d[0]
    ind = []
    nodes = graph.nodes()
    
    for i in range(0,len(d),1):
        if d[i] == s:
            ind.append(nodes[i])
            
    return ind


def loop_track(source_dest,graph): 
    # sourceDest contains source and destination which are node names.
    D = distance(graph)
    n = graph.nodes()
    u = n.index(source_dest[0])
    v = n.index(source_dest[1])
    
    # Convert node names to indices in cycle.
    loop = [source_dest[0]]
    end = int(D[u,v]-1)
    
    # Start loop with source and calculate length of source to destination.
    for i in range(1,end+1,1): # The source to destination path.
        rh = pred_distance(v,D[u,v]-i,D,graph)
        lh = succ_distance(n.index(loop[len(loop)-1]),1,D,graph)
        
        for item in lh:
            if item in rh:
                loop.append(item)
                break 
                # This break is the cause of the non-uniqueness of this 
                # function's results
                
    loop.append(source_dest[1])
    end = D[v,u]-1
    end = int(end)
    
    for i in range(1,end+1,1):
        # the destination back to source path
        rh = pred_distance(u,D[v,u]-i,D, graph)
        lh = succ_distance(n.index(loop[len(loop)-1]),1,D,graph)
        
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
            loop = loop_track(item,graph)
            
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
    
    # Make an adjacency matrix for the loop.
    edges = [] # List of adjacency matrices for each loop.
    
    for loop in loops:
        # For every loop, add edges to a matrix in adjacency-style.
        l = len(loop)
        E = np.zeros(A.shape)
        
        for j in range(0,l-1):
            source_name = loop[j]
            dest_name = loop[j+1]
            source = nodes.index(source_name)
            dest = nodes.index(dest_name)
            E[source,dest] = 1
            
        first_name = loop[0]
        last_name = loop[l-1]
        first = nodes.index(first_name)
        last = nodes.index(last_name)
        E[last,first] = 1
        edges.append(E)
    
    B = np.zeros(A.shape);
    S = loops
    
    SILS = []
    # Here comes the construction of the SILS.
    
    while S:
        
        # Figure out which loops do not contribute any new edges.

        copy_s = S[:]
        edges_copy = []
        for j, element in enumerate(edges):
            # Counting the number of contributions the loop can introduce.
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
        
        # Of the remaining loops, determine the contribution of edges.
        number_of_new_edges = []
        
        for element in edges:
            test = element - B
            test[test<0] = 0
            new_edges = int(np.sum(test))
            number_of_new_edges.append(new_edges)
        
        if number_of_new_edges:
            # The loop with minimum contribution should be added
            m = number_of_new_edges.index(min(number_of_new_edges))
    
            # Add edges to the SILS edge collection
            B = B + edges[m]
            SILS.append(S[m])
            
            # Remove the items from the loop collection
            S.pop(m)
            edges.pop(m)

    return SILS

if __name__ == "__main__":
    
    ''' The following lines should be copied to a new file,
    this to alter the model or its graph into a form that can be used in the 
    SILS-routine developed.'''
    
    # The graph object which is used, is build and we get rid of the 
    # "nonsense" variables.
    file_path = r'minerals and metals network.cPickle'
    graph = cPickle.load(open(file_path, 'r'))
    graph.remove_nodes_from(["INITIAL TIME","TIME STEP","SAVEPER","Time",
                             "TIME STEP","FINAL TIME"])
    
    cycle_graphs = cycle_graph(graph)
    
#    for graph in cycleGraphs:
#        graphPlot(graph)
    
    for graph in cycle_graphs:
        
        sils = SILS(graph)
#        for entry in sils:
#            print entry

        indices_edges,unique_loops = LoopKO.unique_edges(graph,sils)
        indices_cons_edges,unique_cons_loops = LoopKO.unique_cons_edges(graph,sils)
        unique,unique_edges,uniq_cons,unique_cons_edges,notOff = LoopKO.switchChoose(sils,
                                                                                 indices_edges,
                                                                                 unique_loops,
                                                                                 indices_cons_edges,
                                                                                 unique_cons_loops,
                                                                                 graph)
        
        for elem in indices_edges: print elem
#        for elem in uniqueLoops: print elem
#        for elem in extraLoops: print elem
        for elem in indices_cons_edges: print elem
#        for elem in uniqueConsLoops: print elem
        print 'The number of loops that cannot be turned off is {}.'.format(len(notOff))
        print 'Loops that cannot be turned off are the following.'
        for loop in notOff: print loop
        print 'Loops that can be turned off by unique edges are the following.'
        for elem in unique: print elem
        for name in unique_edges: print name
        print 'Loops that can be turned off by two unique consecutive edges are the following.'
        for elem in uniq_cons: print elem
        for name in unique_cons_edges: print name
