'''
Created in November 2011 - January 2012

@author: Bas M.J. Keijser
'''
import networkx as nx
import numpy as np

''' First the adjacency and the reach matrix are constructed.
These matrices can be used to partition the model into levels and/or cycles.
The functions binAddI and binPwr are used to construct reach. '''

def adj(graph): # Construction of the adjacency matrix #
    
    adj=nx.adj_matrix(graph, nodelist=None, weight=None)
    
    return adj

def binary_add_identity(adj): # Adds the identity matrix to the adjacency matrix, by means of Boolean addition #
    a = np.mat(adj);
    I = np.identity(len(adj),float);
    aI = a + I
    
    for i in range(0,len(adj),1):
        
        for j in range(0,len(adj),1):
            if aI[i,j]>1: aI[i,j]=1
            
    return aI 


def binary_substraction(A,B):
    A = np.mat(A);
    B = np.mat(B)
    S = A - B
    
    for i in range(0,len(S),1):
        
        for j in range(0,len(S),1):
            if S[i,j]<0: S[i,j]=0
            
    return S


def binary_power(A,n): # Multiplies the adjacency matrix with itself (number of times as specified), also Boolean #
    A = np.mat(A);
    m = np.linalg.matrix_power(A,n)
    
    for i in range(0,len(m),1):
        
        for j in range(0,len(m),1):
            if m[i,j]>1: m[i,j]=1
            
    return m


def binary_product(A,B):
    A = np.mat(A); B = np.mat(B)
    P = np.dot(A,B)
    
    for i in range(0,len(P),1):
        
        for j in range(0,len(P),1):
            if P[i,j]>1: P[i,j]=1
            
    return P


def reach(adj): # Defines the reach matrix derived from an adjacency matrix
    aI = binary_add_identity(adj)
    p = aI
    n = binary_power(aI,2)
    diff = n - p
    zeros = diff - diff;
    cond = np.equal(diff,zeros)
    
    while False in cond:
        p = n
        n = binary_power(p,2)
        diff = n - p
        cond = np.equal(diff,zeros)
        
    reach = np.array(n)
    
    return reach


''' In this section the successor and predecessor sets of every node are derived.
Two representations of these sets can be made: the succ and pred-functions use one of them.
These functions use a list of node indices. So it is very important that the right input is used.
This input should be a reachability matrix with columns and rows ordered in the same way as the nodes inside the graph-structure are.
Also the interior  of two sets is derived (in intsp), this function use the columns resp. rows of the reach matrix. '''

def succ(reach,n): # Gives list of indices of nodes that are successor to node n
    s = reach[n]
    nzi = list(np.nonzero(s)[0])
    succ = list(nzi[0])
    # Nzi stands for non-zero indices, i.e. indices of non-zero elements
    return succ


def pred(reach,n): # Gives list of indices of nodes that are predecessor to node n
    p = reach[:,n]
    nzi = np.nonzero(p)
    pred = list(nzi[0])
    return pred


def intsp(reach,n): # Defines the intersection of the successor and predecessor sets
    s = reach[n]; p = reach[:,n]; intsp = []
    intsp = s + p
    
    for i in range(0,len(intsp),1): # Equivalent of logical AND, couldn't find how to get that working
        if intsp[i]==1: intsp[i]=0
    
    for i in range(0,len(intsp),1):
        if intsp[i]>1: intsp[i]=1
        
    return s,intsp

  
''' Now the two different partitions, into levels and into cycles, can be constructed.
It uses most of the functions derived above. '''
 
def levels(reach,nodes): # Used to construct different levels derived from reach matrix, returns list of lists
    r = reach; nd = np.array(range(len(nodes))) # nd and r are used to delete nodes that have been assigned to a level, nd consists of node indices
    levels = []
    
    while len(nd) != 0: # While there are nodes not assigned to a level
        level = []
        j = []
        
        for i in range(len(nd)): # Iterate over remaining nodes
            (succ1,intsp1) = intsp(r,i)
            cond = np.equal(succ1,intsp1) # Gives list with all True's if interior of p and s is equal to s
            
            if not False in cond: # If node is in top level
                level.append(nd[i]) # Introduce to current level
                j.append(i)
                
        levels.append(level) # Make a list of nodes in current level
        nd = np.delete(nd,j)
        r = np.delete(r, j, axis=0)
        r = np.delete(r, j, axis=1)
        
    return levels


''' A NetworkX function is used to come to cycle partition(s)'''
def cycles(graph, l):
    cycle_partitions = []
    
    for level in l:
        # Get the subgraph with the nodes in level.
        nodes = graph.nodes()
        nodes = np.array(nodes)
        nodes = nodes[level]
        subgraph = graph.subgraph(nodes)
        
        # Get the strongly connected components in l.
        st_cc = nx.strongly_connected_components(subgraph)
        
        # Filter and remove anything of length 1.
        st_cc = [component for component in st_cc if len(component)>1]
            
        for entry in st_cc:
            cycle_partitions.append(entry)
            
    return cycle_partitions


''' This is used to "translate" between node indices and node names. '''
def node_names(indices,graph):
    node_names = []
    n = graph.nodes()
    
    for item in indices:
        name = n[item]
        node_names.append(name)
        
    return node_names