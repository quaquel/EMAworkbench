'''
Created on May 22, 2015

@author: jhkwakkel
'''
from __future__ import division, print_function

import pydot
import types

import numpy as np
import numpy.lib.recfunctions as recfunctions
from sklearn import tree
from sklearn.externals.six import StringIO

from expWorkbench import ema_logging
import scenario_discovery_util as sdutil


def setup_cart(results, classify, incl_unc=[], mass_min=0.05):
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
    
    return CART(x, y, mass_min)


class CART(sdutil.OutputFormatterMixin):
    sep = '?!?'
    
    def __init__(self, x,y, mass_min):
        self.x = x
        self.y = y
        self.mass_min = mass_min

        # we need to transform the structured array to a ndarray
        # we use dummy variables for each category in case of categorical 
        # variables. Integers are treated as floats
        self.feature_names = []
        columns = []
        for unc, dtype in x.dtype.descr:
            dtype = x.dtype.fields[unc][0]
            if dtype==np.object:
                categories =  sorted(list(set(x[unc])))
                for cat in categories:
                    label = '{}{}{}'.format(unc, self.sep,cat)
                    self.feature_names.append(label)
                    columns.append(x[unc]==cat)
            else:
                self.feature_names.append(unc)
                columns.append(x[unc])

        self._x = np.column_stack(columns)
        self._boxes = None
        self._stats = None

    @property
    def boxes(self):
        assert self.clf
        
        if self._boxes:
            return self._boxes
    
        # based on
        # http://stackoverflow.com/questions/20224526/how-to-extract-the-
        # decision-rules-from-scikit-learn-decision-tree
        assert self.clf
        
        left = self.clf.tree_.children_left
        right = self.clf.tree_.children_right
        threshold = self.clf.tree_.threshold
        features = [self.feature_names[i] for i in self.clf.tree_.feature]
    
        # get ids of leaf nodes
        leafs = np.argwhere(left == -1)[:,0]     
    
        def recurse(left, right, child, lineage=None):          
            if lineage is None:
                # lineage = [self.clf.tree_.value[child]]
                lineage = []
            
            if child in left:
                parent = np.where(left == child)[0].item()
                split = 'l'
            else:
                parent = np.where(right == child)[0].item()
                split = 'r'
    
            lineage.append((parent, split, threshold[parent],
                            features[parent]))
    
            if parent == 0:
                lineage.reverse()
                return lineage
            else:
                return recurse(left, right, parent, lineage)
            
        box_init = sdutil._make_box(self.x)
        boxes = []
        for leaf in leafs:
            branch = recurse(left, right, leaf)
            box = np.copy(box_init)
            for node in branch:
                direction = node[1]
                value = node[2]
                unc = node[3]
                
                if direction=='l':
                    try:
                        box[unc][1] = value
                    except ValueError:
                        unc, cat = unc.split(self.sep)
                        cats = box[unc]
                        cats.pop(cat)
                        box[unc][:]=cats
                else:
                    try:
                        box[unc][0] = value
                    except ValueError:
                        # we are in the right hand branch, so 
                        # the category is included
                        pass
                        
            boxes.append(box) 
        self._boxes = boxes
        return self._boxes       
    
    @property
    def stats(self):
        if self._stats:
            return self._stats
        
        boxes = self.boxes
        total_coi = np.sum(self.y)
        box_init = sdutil._make_box(self.x)
        
        self._stats = []
        for box in boxes:
            indices = sdutil._in_box(self.x, box)
            
            y_in_box = self.y[indices]
            box_coi = np.sum(y_in_box)
            
            boxstats = {'coverage': box_coi/total_coi,
                        'density': box_coi/y_in_box.shape[0],
                        'res dim':sdutil._determine_nr_restricted_dims(box,
                                                                       box_init),
                        'mass':y_in_box.shape[0]/self.y.shape[0]}
            self._stats.append(boxstats)
        return self._stats

    def build_tree(self):
        '''train CART on the data'''
        min_samples = int(self.mass_min*self.x.shape[0])
        self.clf = tree.DecisionTreeClassifier(min_samples_leaf=min_samples)
        self.clf.fit(self._x,self.y)

    def show_tree(self):
        '''return a png of the tree'''
        assert self.clf

        dot_data = StringIO() 
        tree.export_graphviz(self.clf, out_file=dot_data, 
                             feature_names=self.feature_names) 
        graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
        img = graph.create_png()
        return img

       
if __name__ == '__main__':
    from test import util
    import matplotlib.pyplot as plt

    ema_logging.log_to_stderr(ema_logging.INFO)

    def scarcity_classify(outcomes):
        outcome = outcomes['relative market price']
        change = np.abs(outcome[:, 1::]-outcome[:, 0:-1])
        
        neg_change = np.min(change, axis=1)
        pos_change = np.max(change, axis=1)
        
        logical = (neg_change > -0.6) & (pos_change > 0.6)
        
        classes = np.zeros(outcome.shape[0])
        classes[logical] = 1
        
        return classes
 
    results = util.load_scarcity_data()
    
    cart = perform_cart(results, scarcity_classify)
    cart.build_tree()
    
    print(cart.boxes_to_dataframe())
    print(cart.stats_to_dataframe())
    cart.display_boxes(together=True)
    
#     img = cart.show_tree()
#      
#     import matplotlib.pyplot as plt
#     import matplotlib.image as mpimg
#   
#     # treat the dot output string as an image file
#     sio = StringIO()
#     sio.write(img)
#     sio.seek(0)
#     img = mpimg.imread(sio)
#       
#     # plot the image
#     imgplot = plt.imshow(img, aspect='equal')
#       
    plt.show()