'''
Created on 22 feb. 2013

@author: localadmin
'''
import numpy as np
from scipy.stats._support import StringType


class PrimBox(object):
    pass

class PrimException(Exception):
    pass

def def_obj_func(y_old, y_new):
    r'''
    the default objective function used by prim, instead of the original
    objective function, this function can cope with continuous, integer, and
    categorical uncertainties.      
    
    .. math::
        
        obj = \frac
             {\text{ave} [y_{i}\mid x_{i}\in{B-b}] - \text{ave} [y\mid x\in{B}]}
             {|n(y_{i})-n(y)|}
    
    where :math:`B-b` is the set of candidate new boxes, :math:`B` the old box 
    and :math:`y` are the y values belonging to the old box. :math:`n(y_{i})` 
    and :math:`n(y)` are the cardinality of :math:`y_{i}` and :math:`y` 
    respectively. So, this objective function looks for the difference between
    the mean of the old box and the new box, divided by the change in the 
    number of data points in the box. This objective function offsets a problem 
    in case of categorical data where the normal objective function often 
    results in boxes mainly based on the categorical data.  
    
    :param old_y: the y's belonging to the old box
    :param new_y: the y's belonging to the new box
    
    '''
    
    mean_old = np.mean(y_old)
    mean_new = np.mean(y_new)
    obj = 0
    if mean_old != mean_new:
        if y_old.shape >= y_new.shape:
            obj = (mean_new-mean_old)/(y_old.shape[0]-y_new.shape[0])
        else:
            obj = (mean_new-mean_old)/(y_new.shape[0]-y_old.shape[0])
    return obj


class Prim(object):
    
    def __init__(self, 
                 results,
                 classify, 
                 obj_function=def_obj_func, 
                 peel_alpha = 0.05, 
                 paste_alpha = 0.05,
                 mass_min = 0.05, 
                 threshold = None, 
                 pasting=True, 
                 threshold_type=1):
        '''
        
        :param results: the return from :meth:`perform_experiments`.
        :param classify: either a string denoting the outcome of interest to 
                         use or a function. 
        :param peel_alpha: parameter controlling the peeling stage (default = 0.05). 
        :param paste_alpha: parameter controlling the pasting stage (default = 0.05).
        :param mass_min: minimum mass of a box (default = 0.05). 
        :param threshold: the threshold of the output space that boxes should meet. 
        :param pasting: perform pasting stage (default=True) 
        :param threshold_type: If 1, the boxes should go above the threshold, if -1
                               the boxes should go below the threshold, if 0, the 
                               algorithm looks for both +1 and -1.
        :param obj_func: The objective function to use. Default is 
                         :func:`def_obj_func`
        :raises: PrimException if data resulting from classify is not a 
                 1-d array.
                     
        '''
        
        self.x = results[0]
        
        if classify==StringType:
            self.y = results[1][classify]
        
        
        if len(self.y.shape) > 1:
            raise PrimException
        
        
        
        pass
    
    def perform_pca(self):
        '''
        
        Pre-process the data by performing a pca based rotation on it. 
        This effectively turns the algorithm into PCA-PRIM as described
        in the envsoft paper
        
        '''
        
        pass
    
    def find_box(self):
        '''
        
        Executate one iteration of the PRIM algorithm. That is, find one
        box, starting from the current state of Prim. 
        
        
        '''
        
        pass
    
    def __peel(self):
        '''
        
        Executes the peeling phase of the PRIM algorithm. Delegates peeling
        to data type specific helper methods.
        
        '''
        
        pass
    
    def __paste(self):
        '''
        
        Executates the pasting phase of the PRIM. Delegates pasting to data 
        type specific helper methods.
        
        '''
        
        
        pass