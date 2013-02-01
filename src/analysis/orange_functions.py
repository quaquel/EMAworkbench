'''

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

This module contains some convenience functions that wrap machine learning
algorithms implemented in `orange <http://orange.biolab.si/doc/reference/Domain.htm>`_

The current wrappers use default values for the various parameters that can be
specified. Follow the provided links to the orange functions that are being
wrapped for more details. 

:func:`build_orange_data` can be used as a starting point if one wants to use 
other algorithms provided by orange.

Where appropriate, the relevant documentation from orange has been used. 

'''
from __future__ import division
import numpy as np

import orange, orngTree, orngEnsemble, orngFSS #@UnresolvedImport

from expWorkbench.ema_logging import info

__all__ = ['build_orange_data', 
           'random_forest', 
           'feature_selection',
           'random_forest_measure_attributes', 
           'tree']

FLOAT = orange.FloatVariable
ENUM = orange.EnumVariable

def build_orange_data(data,classify):
    '''
    
    helper function for turning the data from :meth:`perform_experiments` into 
    a data object that can be used by the various orange functions. 
    
    For more details see `orange domain <http://orange.biolab.si/doc/reference/Domain.htm>`_  
    
    :param data: return from :meth:`perform_experiments`.
    :param classify: function to be used for determining the class for each 
                     run.
    
    '''
    info("building orange data")
    
    experiments, results = data

    #build domain
    dtypes =  []
    for entry in experiments.dtype.descr:
        dtypes.append((entry[0], experiments.dtype.fields.get(entry[0])))
    
    attributes = []
    for entry in dtypes:
        name, dtype = entry
        dtype = dtype[0].name
        if dtype == 'int' or dtype =='object':
            attribute = ENUM(name)
            [attribute.addValue(str(value)) for value in\
                                            set(experiments[name].tolist())]
        else:
            attribute = FLOAT(name, startValue = np.min(experiments[name]), 
                              endValue = np.max(experiments[name]))
        attributes.append(attribute)

    data = np.array(experiments.tolist())
        
    #determine classes
    classes = classify(results)
    classVar = ENUM('class')
    #these numbers are merely referring to the possible classes
    [classVar.addValue(str(i)) for i in set(classes.tolist())] 
    #by default the last entry in the list should be the class variable
    attributes.append(classVar) 
    domain = orange.Domain(attributes)
    
    data = np.hstack((data, classes[:, np.newaxis]))
    data = data.tolist()
    data = orange.ExampleTable(domain, data)

    return data

def random_forest(data, classify, nrOfTrees=100, attributes=None):
    '''
    make a random forest using orange
    
    For more details see `orange ensemble <http://orange.biolab.si/doc/modules/orngEnsemble.htm>`_
    
    :param data: data from :meth:`perform_experiments`.
    :param classify: function for classifying runs.
    :param nrOfTrees: number of trees in the forest (default: 100).
    :param attributes: Number of attributes used in a randomly drawn subset 
                       when searching for best attribute to split the node in 
                       tree growing (default: None, and if kept this way, this 
                       is turned into square root of attributes in 
                       example set).
    :rtype: an orange random forest.
    
    '''
    data = build_orange_data(data, classify)
    
    #do the random forest
    #see http://orange.biolab.si/doc/modules/orngEnsemble.htm for details
    info("executing random forest")
    measure = orngEnsemble.MeasureAttribute_randomForests(trees=nrOfTrees, 
                                                        attributes=attributes)
    
    return measure

def feature_selection(data, classify, k=5, m=100):
    '''
    
    perform feature selection using orange
    
    For more details see `orange feature selection <http://orange.biolab.si/doc/modules/orngFSS.htm>`_ and
    `orange measure attribute <http://orange.biolab.si/doc/reference/MeasureAttribute.htm>`_
    
    the default measure is ReliefF ((MeasureAttribute_relief in Orange).
    
    :param data: data from :meth:`perform_experiments`.
    :param classify: function for classifying runs.
    :param k: the number of neighbors for each example (default 5).
    :param m: number of examples to use, Set to -1 to use all (default 100).
    :rtype: sorted list of tuples with uncertainty names and reliefF attribute 
            scores.
    
    Orange provides other metrics for feature selection
    
    * Information Gain
    * Gain ratio 
    * Gini index 
    * Relevance of attributes 
    * Costs
    
    If you want to use any of of these instead of ReliefF, use the code
    supplied here as a template, but modify the measure. That is replace::
    
        measure = orange.MeasureAttribute_relief(k=k, m=m)
        
    with the measure of choice. See the above provided links for more details.
    
    '''
    data = build_orange_data(data, classify)

    info("executing feature selection")
    measure = orange.MeasureAttribute_relief(k=k, m=m)
    ma = orngFSS.attMeasure(data, measure)
    
    results = [] 
    for m in ma:
        results.append((m[1], m[0]))
    results.sort(reverse=True)
    
    results = [(entry[1], entry[0]) for entry in results]
    return results
        
    
    
def random_forest_measure_attributes(data, classify):
    '''
    performs feature selection using random forests in orange.
    
    For more details see `orange ensemble <http://orange.biolab.si/doc/modules/orngEnsemble.htm>`_
    
    :param data: data from :meth:`perform_experiments`.
    :param classify: function for classifying runs.
    :param nrOfTrees: number of trees in the forest (default: 100).
    :param attributes: Number of attributes used in a randomly drawn subset 
                       when searching for best attribute to split the node in 
                       tree growing. (default: None, and if kept this way, this 
                       is turned into square root of attributes in example set)
    :rtype: sorted list of tuples with uncertainty names and importance values.
    
    '''
    data = build_orange_data(data, classify)
    
    #do the random forest
    #see http://orange.biolab.si/doc/modules/orngEnsemble.htm for details
    info("executing random forest for attribute selection")
    measure = orngEnsemble.MeasureAttribute_randomForests(trees=100)
    
    #calculate importance
    imps = measure.importances(data)
    
    #sort importance, using schwartzian transform
    results = [] 
    for i,imp in enumerate(imps): 
        results.append((imp, data.domain.attributes[i].name))
    results.sort(reverse=True)
    
    results = [(entry[1], entry[0]) for entry in results]
    return results


def tree(data, 
         classify,
         sameMajorityPruning=False,
         mForPruning=0,
         maxMajority= 1,
         minSubset = 0,
         minExamples = 0):
    '''
    make a classification tree using orange
    
    For more details see `orange tree <http://orange.biolab.si/doc/modules/orngTree.htm>`_
    
    :param data: data from :meth:`perform_experiments`
    :param classify: function for classifying runs
    :param sameMajorityPruning: If true, invokes a bottom-up post-pruning by 
                                removing the subtrees of which all leaves 
                                classify to the same class (default: False).
    :param mForPruning: If non-zero, invokes an error-based bottom-up 
                        post-pruning, where m-estimate is used to estimate 
                        class probabilities (default: 0).
    :param maxMajority: Induction stops when the proportion of majority class 
                        in the node exceeds the value set by this parameter
                        (default: 1.0). 
    :param minSubset: Minimal number of examples in non-null leaves 
                      (default: 0).
    :param minExamples: Data subsets with less than minExamples examples are 
                        not split any further, that is, all leaves in the tree 
                        will contain at least that many of examples 
                        (default: 0).
    :rtype: a classification tree
    
    in order to print the results one can for example use `graphiv <http://www.graphviz.org/>`_.
    
    >>> import orgnTree
    >>> tree = tree(input, classify)
    >>> orngTree.printDot(tree, r'..\..\models\\tree.dot', 
                      leafStr="%V (%M out of %N)") 
    
    this generates a .dot file that can be opened and displayed using graphviz.
    the leafStr keyword argument specifies the format of the string for
    each leaf. See on this also the more detailed discussion on the orange 
    web site.
    
    At some future state, a convenience function might be added for turning
    a tree into a `networkx graph <http://networkx.lanl.gov/>`_. However, this
    is a possible future addition. 
    
    '''

    data = build_orange_data(data, classify)

    #make the actually tree, for details on the meaning of the parameters, see 
    #the orange webpage
    info("executing tree learner")
    tree = orngTree.TreeLearner(data,
                                sameMajorityPruning=sameMajorityPruning,
                                mForPruning=mForPruning,
                                maxMajority=maxMajority,
                                minSubset = minSubset ,
                                minExamples = minExamples)
    info("tree contains %s leaves" % orngTree.countLeaves(tree))
    
    return tree
    

#def multi_dimensional_scaling(data,
#                              classify,
#                              runs=100):
#    '''
#    Perform multi dimensional scaling using the `orngMDS<http://orange.biolab.si/doc/modules/orngMDS.htm>`_
#    module. 
#    
#    :param data: data from :meth:`perform_experiments`
#    :param classify: function for classifying runs
#    :param runs: the number of iterations to be used in MDS
#    :returns: a `figure<http://matplotlib.sourceforge.net/api/figure_api.html#matplotlib.figure.Figure>`_ instance
#    
#    
#    '''
#    data = build_orange_data(data, classify)
#    
#    euclidean = orange.ExamplesDistanceConstructor_Euclidean(data) 
#    distance = orange.SymMatrix(len(data)) 
#    for i in range(len(data)): 
#        for j in range(i+1): 
#            distance[i, j] = euclidean(data[i], data[j]) 
#    
#    mds=orngMDS.MDS(distance) 
#    mds.run(runs)
#    
#
#    colors = graphs.COLOR_LIST
#    points = [] 
#    
#    figure = plt.figure()
#    ax = figure.add_subplot(111)
#    
#    for (i,d) in enumerate(data): 
#        points.append((mds.points[i][0], mds.points[i][1], d.getclass())) 
#    
#    for c in range(len(data.domain.classVar.values)): 
#        sel = filter(lambda x: x[-1]==c, points) 
#        x = [s[0] for s in sel] 
#        y = [s[1] for s in sel] 
#        ax.scatter(x, y, c=colors[c]) 
#    
#    return figure