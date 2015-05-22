'''
Created on May 22, 2015

@author: jhkwakkel
'''
import copy
import types
import pydot




import numpy as np
import numpy.lib.recfunctions as recfunctions

from sklearn import tree
from sklearn.tree import _tree
from sklearn.externals.six import StringIO


from expWorkbench import ema_logging

ema_logging.log_to_stderr(ema_logging.INFO)


def perform_cart(results, classify, incl_unc=[], mass_min=0.05):
    """helper function for performing cart
    
    Parameters
    ----------
    results : tuple of structured array and dict with numpy arrays
              the return from :meth:`perform_experiments`.
    classify : string, function or callable
               either a string denoting the outcome of interest to 
               use or a function. 
    incl_unc : list of strings
    mass_min : float
    
    
    Raises
    ------
    TypeError 
        if classify is not a string or a callable.
    
    """
    
    if not incl_unc:
        x = np.ma.array(results[0])
    else:
        drop_names = set(recfunctions.get_names(results[0].dtype))-set(incl_unc)
        x = recfunctions.drop_fields(results[0], drop_names, asrecarray = True)
    if type(classify)==types.StringType:
        y = results[1][classify]
    elif callable(classify):
        y = classify(results[1])
    else:
        raise TypeError("unknown type for classify")
    
    # we need to transform the structured array to a ndarray
    # for object dtype, we should return the mapping
    x_temp = np.zeros((x.shape[0], len(x.dtype.descr)))
    for i, entry in enumerate(x.dtype.descr):
        data_type = x.dtype.fields[entry[0]][0]
        data = x[entry[0]]
        if data_type == np.object:
            items = set(data)
            temp_data = np.zeros(data.shape)
            for index, item in enumerate(items):
                temp_data[data==item] = index
            x_temp[:, i] = temp_data
            
        else:
            dtype = zip(entry, ['float64'])
            x_temp[:, i] = np.array(data).view(dtype=dtype).copy()
    
    x = x_temp
    
    return build_tree(x, y, mass_min)
    

def build_tree(x, y, mass_min):
    clf = tree.DecisionTreeClassifier(min_samples_leaf = int(mass_min*x.shape[0]))
    clf = clf.fit(x,y)
    return clf


def show_tree(clf, feature_names):
    dot_data = StringIO() 
    tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_names) 
    graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
    img = graph.create_png()
    return img

def get_branches(tree, node_id, parent=None, depth=0, path=[]):
    path = copy.copy(path)
    path.append(node_id)
    depth += 1
    
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    # Add node with description
    if left_child != _tree.TREE_LEAF: # @SupressWarning
        new_paths = []
        
        left_paths = get_branches(tree, left_child, parent=node_id, depth=depth, path=path)
        right_paths = get_branches(tree, right_child, parent=node_id, depth=depth, path=path)
        for entry in [left_paths, right_paths]:
            for path in entry:
                new_paths.append(path)
        return new_paths
    else:
        return [path]


def make_box(x):
    '''
    Make a box that encompasses all the data
    
    :param x: a structured array
    
    '''
    
    box = np.zeros((2, ), x.dtype)
    
    names = recfunctions.get_names(x.dtype)
    
    for name in names:
        dtype = x.dtype.fields.get(name)[0] 
        values = x[name]
        
        if dtype == 'object':
            try:
                values = set(values) - set([np.ma.masked])
                box[name][:] = values
            except TypeError as e:
                ema_logging.warning("{} has unhashable values".format(name))
                raise e
        else:
            box[name][0] = np.min(values, axis=0) 
            box[name][1] = np.max(values, axis=0)    
    return box  
    

# http://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree

def tree_to_boxes(experiments, decision_tree, feature_names=None):
    """transform a classification tree to a PRIM like box representation

    """
    branches = get_branches(decision_tree.tree_, 0)
    tree = decision_tree.tree_
    
    # left betekent dat het below is, right betekent above
    # probleem is echter dat left en right gelden tov parent, en dus de
    # threshold die we daar opvragen
    left_children = set(tree.children_left)
    right_children = set(tree.children_right)
    
    boxes = []
    
    for branch in branches:
        box = make_box(experiments)
        results = None
        
        feature = None
        threshold = None
        for node in branch:
            if feature:
                if node in left_children:
                    box[feature][1] = threshold
                else:
                    box[feature][0] = threshold
            if tree.children_left[node] == _tree.TREE_LEAF:
                value = tree.value[node]
                if tree.n_outputs == 1:
                    value = value[0, :]
                results = value
            else:
                feature = feature_names[tree.feature[node]]
                threshold = tree.threshold[node]
                
        boxes.append((box, results))
    
    return boxes

    
if __name__ == '__main__':
    from test import util
 
    def flu_classify(data):
        #get the output for deceased population
        result = data['deceased population region 1']
        
        #make an empty array of length equal to number of cases 
        classes =  np.zeros(result.shape[0])
        
        #if deceased population is higher then 1.000.000 people, classify as 1 
        classes[result[:, -1] > 1000000] = 1
        
        return classes   
 
    results = util.load_flu_data()
    
    clf = perform_cart(results, flu_classify)
    img = show_tree(clf, np.lib.recfunctions.get_names(results[0].dtype))   
    
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    # treat the dot output string as an image file
    sio = StringIO()
    sio.write(img)
    sio.seek(0)
    img = mpimg.imread(sio)
    
    # plot the image
    imgplot = plt.imshow(img, aspect='equal')
    
    plt.show()