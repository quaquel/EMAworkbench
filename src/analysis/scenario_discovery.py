'''
.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


This module contains helper functions related to scenario discovery. These
functions can be used to transform a classification tree into a list of 
boxes, identical to those returned by the PRIM algorithm. There are also
functions here for calculating the scenario discovery metrics *coverage* and
*density*.

'''
from __future__ import division
import copy
from types import StringType

import numpy as np

from analysis.primCode.primDataTypeAware import make_box, Prim, in_box,\
                                                prim_hdr

__all__ = ['calculate_sd_metrics',
           'make_boxes',
           'find_branches']


def calculate_sd_metrics(boxes, y, threshold, threshold_type):
    r'''

    Function for calculating the coverage and density scenario discovery
    metrics.
    
    :param boxes: A list of PRIM boxes.
    :param y: The y vector used in generating the boxes. This is typically 
              the return from a classify function.   
    :param threshold: the threshold of the output space that boxes should meet. 
    :param threshold_type: If 1, the boxes should go above the threshold, if -1
                           the boxes should go below the threshold, if 0, the 
                           algorithm looks for both +1 and -1.
    :return: The list of PRIM boxes with coverage and density added to
             each box as additional attribute.
    
    
    Coverage and density are given below:
     
    .. math::
 
        coverage=\frac
                    {{\displaystyle\sum_{y_{i}\in{B}}y_{i}{'}}}
                    {{\displaystyle\sum_{y_{i}\in{B^I}}y_{i}{'}}}
    
    
    where :math:`y_{i}{'}=1` if :math:`x_{i}\in{B}` and :math:`y_{i}{'}=0`
    otherwise.
    
    Coverage is the ratio of cases of interest in a box to the total number
    of cases of interests. It thus provides insight into which fraction of 
    cases of interest is in a particular box.
    
    .. math::
 
        density=\frac
                    {{\displaystyle\sum_{y_{i}\in{B}}y_{i}{'}}}
                    {{\displaystyle\left|{y_{i}}\right|\in{B}}}
    
    where :math:`y_{i}{'}=1` if :math:`x_{i}\in{B}` and :math:`y_{i}{'}=0`
    otherwise, and :math:`{\displaystyle\left|{y_{i}}\right|\in{B}}` is the
    cardinality of :math:`y_{i}`.    
        
    Density is the ratio of the cases of interest in a box to the 
    total number of cases in that box. *density* is identical to the mean
    in case of a binary classification.  For more detail on these metrics see 
    `Bryant and Lempert (2010) <http://www.sciencedirect.com/science/article/pii/S004016250900105X>`_
    
    '''
    
    t_coi = __calculate_cases_of_interest(y, threshold, threshold_type)
    for box in boxes:
        coi = __calculate_cases_of_interest(box.y, threshold, threshold_type)
        box.coverage = coi/t_coi
        box.density = coi/box.y.shape[0]
    return boxes

def __calculate_cases_of_interest(y, threshold, threshold_type):
    cases_of_interest = np.sum(y[(y*threshold_type) >= (threshold_type *threshold)])
    return cases_of_interest


def make_boxes(tree, data, classify, threshold):
    '''
    Function that turns a classification tree into prim boxes, including
    the scenario discovery metrics. 
    
    :param tree: the return from :func:`orangeFunctions.tree`.
    :param data: the return from :meth:`perform_experiments`.
    :param classify: the classify function used in making the tree.
    :param threshold: the minimum mean that the boxes should meet.
    :return: a list of prim boxes.
    
    '''
    branches = find_branches(tree.tree, [("root", "")])
    data, results = data
    init_box = make_box(data)
    boxes = []
    
    for branch in branches:
        box = copy.copy(init_box)
        for name in branch[1:]:
            name, limit = name
            if name=="root" or type(name) != StringType:
                continue
            
            if limit.startswith('>'):
                limit = limit[1:]
                limit = float(limit)
                box[name][0] = limit
            else:
                limit = limit[2:]
                limit = float(limit)
                box[name][1] = limit 
        boxes.append(box)
    
    y = classify(results)
    n = y.shape[0]
    threshold=0.8
    threshold_type=1
    #turn boxes into prim objects
    new_boxes = []
    for box in boxes:
        logical = in_box(data, box)
        box = Prim(data[logical], y[logical], box, y[logical].shape[0]/n)
        new_boxes.append(box)
    boxes = prim_hdr(new_boxes, threshold=threshold, threshold_type=threshold_type)
    
    boxes = calculate_sd_metrics(boxes, y, threshold, threshold_type)
    
    return boxes

def find_branches(node, vars):
    '''
    Recursive function for finding branches in a tree.
    
    :param node: The node from which you want to find branches.
    :param vars: The variables found so far and their limits.
    :return: A list of branches. Each branch is in turn a list, starting
             from the distribution in the leaf, back to the root of the tree. 
             For each split in the tree, it gives the name of the variable,
             and the split condition. The split condition is given as a string.
             So. A branch is given from the bottom up.
    
    .. rubric:: example of use
    
    >>> tree = analysis.orangeFunctions.tree(data, classify_function)
    >>> startNode = tree.tree
    >>> branches = find_branches(startNode, ["root", ""]) 
    
    '''
    if node.branches != None:
        new_paths = []
        new_var = node.branch_selector.class_var.name
        for i, element in enumerate(node.branches):
            limit = node.branch_descriptions[i]
            new_vars = copy.copy(vars)
            new_vars.append((new_var, limit))
            paths = find_branches(element, new_vars)
            for path in paths:
                new_paths.append(path)
        return new_paths
    else:
        vars.append(node.distribution)
        return [vars]