'''
Created in April 2012 - May 2012

@author: Bas M.J. Keijser
'''
import numpy as np
import Partitions as prt

def unique_edges(graph,sils):
    
    A = prt.adj(graph)
    A = np.asarray(A)
    nodes = graph.nodes()
    
    # Make an adjacency matrix for every loop in SILS.
    edges = [] # List of adjacency matrices for each loop
    
    for loop in sils:
        
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
    
    # Sum occurrence of edges in loops.
    sum_edges = np.zeros(A.shape)
    for entry in edges:
        # sumEdges += entry
        sum_edges = np.add(sum_edges,entry)
        
    # When sum is one, the edge is in only one loop, so get indices.
    indices_edges = np.where(sum_edges == 1)
    
    unique_loops = []
    
    for p in range(int(len(indices_edges[0]-1))):
        rowNo = int(indices_edges[0][p])
        columnNo = int(indices_edges[1][p])
        
        for q,edge in enumerate(edges):
            
            if edge[rowNo,columnNo] == 1:
                unique_loops.append(sils[q])
    
#    # When sum is two, the edge can be used to switch off two loops
#    extraIndicesEdges = np.where(sumEdges == 2)
#    
#    extraLoops = []
#    for i in range(0,int(len(extraIndicesEdges[0]-1))):
#        twoLoops = []
#        rowNo = int(extraIndicesEdges[0][i])
#        columnNo = int(extraIndicesEdges[1][i])
#        j = 0
#        for edge in edges:
#            if edge[rowNo,columnNo] == 1:
#                twoLoops.append(sils[j])
#            j = j+1
#        extraLoops.append(twoLoops)
    
    return indices_edges,unique_loops


def unique_cons_edges(graph,sils):
    
    # Get graph information needed.
    A = prt.adj(graph)
    A = np.asarray(A)
    nodes = graph.nodes()
    
    # Make an adjacency matrix for every loop in sils
    edges = [] # List of adjacency-style matrices for each loop
    
    for loop in sils:
        # For every loop, add unique consecutive edges to a matrix in adjacency-style
        l = len(loop)
        E = np.zeros(A.shape)
        
        # If length of loop is 2 the else procedure would give problems
        if l == 2:
            for node in loop:
                index = nodes.index(node)
                E[index,index] = 1
                
        else:
            for j in range(0,l-2):
                source_name = loop[j]
                dest_name = loop[j+2] # This line is different, plus 2 this time.
                source = nodes.index(source_name)
                dest = nodes.index(dest_name)
                E[source,dest] = 1
                
            second_name = loop[1]
            first_name = loop[0]
            last_name = loop[l-1]
            last_but_one_name = loop[l-2]
            
            second = nodes.index(second_name)
            first = nodes.index(first_name)
            last = nodes.index(last_name)
            last_but_one = nodes.index(last_but_one_name)
            
            E[last_but_one,first] = 1
            E[last,second] = 1
            
        edges.append(E)
    
    # Sum occurrence of consecutive edges in loops.
    sum_cons_edges = np.zeros(A.shape)
    for entry in edges:
        sum_cons_edges = np.add(sum_cons_edges,entry)
        
    # When sum is one, the edge is in only one loop, so get indices. 
    indices_cons_edges = np.where(sum_cons_edges == 1)
    unique_cons_loops = []
    
    # Construct the loop list and node indices list.
    for i in range(0,int(len(indices_cons_edges[0]-1))):
        rowNo = int(indices_cons_edges[0][i])
        columnNo = int(indices_cons_edges[1][i])
        j = 0
        
        for edge in edges:
            if edge[rowNo,columnNo] == 1:
                unique_cons_loops.append(sils[j])
            j = j+1
            
    return indices_cons_edges,unique_cons_loops


def switchChoose(sils,ind_edges,uniq_loops,ind_cons_edges,uniq_cons_loops,graph):

    # This function is used to come to a conclusion about which loop to switch off how.
    # First try to switch off as many loops as possible by finding unique edges.
    unique = []
    # Make element of the set uniqLoops unique.
    [unique.append(loop) for loop in uniq_loops if loop not in unique]
    
    # Get the right loop numbers.
    ind_source = []
    ind_dest = []
    loop_nos = []
    for loop in unique:
        loop_no = uniq_loops.index(loop)
        loop_nos.append(loop_no)
        
    # Construct the indices list with indices for every loop.    
    [ind_source.append(ind_edges[0][loop_no]) for loop_no in loop_nos]
    [ind_dest.append(ind_edges[1][loop_no]) for loop_no in loop_nos]
    uniq_indices = [ind_source,ind_dest]
    
    # Now switching node indices for variable names.
    names_uniq = []
    for item in uniq_indices:
        names = prt.node_names(item,graph)
        names_uniq.append(names)
    
    # And reordering them into edge style.
    unique_edges = []
    for i in range(0,len(names_uniq[0])):
        unique_edge = [names_uniq[0][i],names_uniq[1][i]]
        unique_edges.append(unique_edge)
    
    # Now eliminating loops from the total collection that needs to be turned off.
    SILS = sils
    [SILS.remove(loop) for loop in unique if loop in SILS]
    
    # Now deleting which loops can be switched off by the method of consecutive edges.
    uniq_cons = []
    [uniq_cons.append(loop) for loop in uniq_cons_loops if loop not in uniq_cons]
    # Removing all loops that can already be turned off simpler.
    [uniq_cons.remove(loop) for loop in unique if loop in uniq_cons]
    
    # Constructing indices lists for every loop.
    ind_source = []
    ind_dest = []
    loop_nos = []
    for loop in uniq_cons:
        loop_no = uniq_cons_loops.index(loop)
        loop_nos.append(loop_no)
    
    # List of indices for every loop with two unique cons edges.    
    [ind_source.append(ind_cons_edges[0][int(loopNo)]) for loopNo in loop_nos]
    [ind_dest.append(ind_cons_edges[1][int(loopNo)]) for loopNo in loop_nos]
    uniqConsIndices = [ind_source,ind_dest]
    
    # Now switching node indices for variable names.
    names_cons = []
    for item in uniqConsIndices:
        names2 = prt.node_names(item,graph)
        names_cons.append(names2)
        
    # And reordering them into edge style.
    unique_cons_edges = []
    for i in range(0,len(names_cons[0])):
        cons_edge = [names_cons[0][i],names_cons[1][i]]
        unique_cons_edges.append(cons_edge)
    
    # Eliminating loops again from SILS.
    [SILS.remove(loop) for loop in uniq_cons if loop in SILS]
    
    return (unique,unique_edges,uniq_cons,unique_cons_edges,SILS)