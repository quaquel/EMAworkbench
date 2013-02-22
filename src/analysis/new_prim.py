'''
Created on 22 feb. 2013

@author: localadmin
'''
from __future__ import division
from types import StringType, FloatType, IntType
from operator import itemgetter
import copy

import numpy as np
from scipy.stats.mstats import mquantiles #@UnresolvedImport
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.pyplot as plt

from analysis.plotting_util import make_legend
from expWorkbench import info

DEFAULT = 'default'
ABOVE = 1
BELOW = -1

class PrimBox(object):
    
    def __init__(self, prim, box_lims, indices):
        self.prim = prim
        
        # peeling and pasting trajectory
        self.coverage = []
        self.density = []
        self.mean = []
        self.res_dim = []
        self.box_lims = []
        self.mass = []
        
        # indices van data in box
        self.update(box_lims, indices)
        
    def select(self, i):
        '''
        
        select an entry from the peeling and pasting trajectory and update
        the prim box to this selected box.
        
        TODO: ideally, this should invoke a paste attempt.
        
        '''
        pass

    def update(self, box_lims, indices):
        '''
        
        update the box to the provided box limits.
        
        what should be updated?
        
        add to peeling trajectory
        update indices in box
        update all metrics
        
        '''
        self.yi = indices
        
        y = self.prim.y[self.yi]

        self.box_lims.append(box_lims)
        self.mean.append(np.mean(y))
        self.mass.append(y.shape[0]/self.prim.n)
        
        coi = self.prim.determine_coi(self.yi)
        self.coverage.append(coi/self.prim.t_coi)
        self.density.append(coi/y.shape[0])
        
        # determine the nr. of restricted dimensions
        # box_lims[0] is the initial box, box_lims[-1] is the latest box
        self.res_dim.append(self.prim.determine_restricted_dims(self.box_lims[-1]))
        
    def show_ppt(self):
        '''
        
        show the peeling and pasting trajectory in a figure
        
        '''
        
        ax = host_subplot(111)
        ax.set_xlabel("peeling and pasting trajectory")
        
        par = ax.twinx()
        par.set_ylabel("nr. restricted dimensions")
            
        ax.plot(self.mean, label="mean")
        ax.plot(self.mass, label="mass")
        ax.plot(self.coverage, label="coverage")
        ax.plot(self.density, label="density")
        par.plot(self.res_dim, label="restricted_dim")
        ax.grid(True, which='both')
        ax.set_ylim(ymin=0,ymax=1)
        
        fig = plt.gcf()
        
        make_legend(['mean', 'mass', 'coverage', 'density', 'restricted_dim'],
                    fig, ncol=5)
        return fig
    
    def write_ppt_stdout(self):
        '''
        
        write the peeling and pasting trajectory to stdout
        
        '''

        print "{0:<5}{1:>10}{2:>10}{3:>10}{4:>10}{5:>10}".format('box', 'mean', 'mass', 'coverage', 'density', 'res dim')
        for i in range(len(self.box_lims)):
            input = {'mean': self.mean[i], 
                     'mass': self.mass[i], 
                     'coverage': self.coverage[i], 
                     'density': self.density[i], 
                     'restricted_dim': self.res_dim[i]}
            row = "{0:<5}{mean:>10.2g}{mass:>10.2g}{coverage:>10.2g}{density:>10.2g}{restricted_dim:>10.2g}".format(i,**input)
            print row
        
        pass 

class PrimException(Exception):
    pass

class Prim(object):

    # parameters that control the mquantile calculation used
    # in peeling and pasting
    alpha = 1/3
    beta = 1/3
    
    def __init__(self, 
                 results,
                 classify, 
                 obj_function=DEFAULT, 
                 peel_alpha = 0.05, 
                 paste_alpha = 0.05,
                 mass_min = 0.05, 
                 threshold = None, 
                 pasting=True, 
                 threshold_type=ABOVE):
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
        :raises: TypeError if classify is not a string or a callable function
                     
        '''
        
        self.x = results[0]
        
        # determine y
        if type(classify)==StringType:
            self.y = results[1][classify]
        elif callable(classify):
            self.y = classify(results[1])
        else:
            raise TypeError("unknown type for classify")
        
        if len(self.y.shape) > 1:
            raise PrimException("y is not a 1-d array")
        
        # store the remainder of the parameters
        self.paste_alpha = paste_alpha
        self.peel_alpha = peel_alpha
        self.mass_min = mass_min
        self.threshold = threshold 
        self.pasting = pasting 
        self.threshold_type = threshold_type
        self.obj_func = self.__obj_functions[obj_function]
        
        # set the indices
        self.yi_remaining = np.arange(0, self.y.shape[0])
        
        # how many data points do we have
        self.n = self.yi_remaining.shape[0]
        
        # how many cases of interest do we have?
        self.t_coi = self.determine_coi(self.yi_remaining)
        
        # initial box that contains all data
        self.box_init = self.make_box(self.x)
    
        # make a list in which the identified boxes can be put
        self.boxes = []
    
    def perform_pca(self):
        '''
        
        Pre-process the data by performing a pca based rotation on it. 
        This effectively turns the algorithm into PCA-PRIM as described
        in the envsoft paper
        
        '''
        
        pass
    
    def find_box(self):
        '''
        
        Execute one iteration of the PRIM algorithm. That is, find one
        box, starting from the current state of Prim. 
        
        
        '''
        
        # TODO here we should assess the state of the algorithm
        # that is, go over all the identifed PrimBoxes, get there last
        # entry and update yi_remaining in light of this
        
        info("{} points remaining".format(self.yi_remaining.shape[0]))
        info("{} cases of interest remaining".format(self.determine_coi(self.yi_remaining)))
        
        box = PrimBox(self, self.box_init, self.yi_remaining)
        
        new_box = self.__peel(box)
        
#        new_box = peel(x_remaining, y_remaining, copy.copy(box_init), peel_alpha, 
#                       mass_min, threshold, n, obj_func)
    
        info("peeling completed")
    
#        if pasting:
#            logical = in_box(x_remaining, new_box.box)
#            x_inside = x_remaining[logical]
#            y_inside = y_remaining[logical]
#    
#            new_box = paste(x_inside, y_inside, x_remaining, y_remaining, 
#                               copy.copy(box_init), new_box, paste_alpha, mass_min, 
#                               threshold, n, obj_func)
#            info("pasting completed")
        
        self.boxes.append(new_box)
        
        return new_box


    def compare(self, a, b):
        '''compare two boxes, for each dimension return True if the
        same and false otherwise'''
        dtypesDesc = a.dtype.descr
        logical = np.ones((len(dtypesDesc,)), dtype=np.bool)
        for i, entry in enumerate(dtypesDesc):
            name = entry[0]
           
            logical[i] = logical[i] &\
                        (a[name][0] == b[name][0]) &\
                        (a[name][1] == b[name][1])
        return logical
    
    def in_box(self, box):
        '''
         
        returns the indices of the remaining data points that are within the 
        box_lims.
        
        '''
        x = self.x[self.yi_remaining]
        logical = np.ones(x.shape[0], dtype=np.bool)
    
        for entry in x.dtype.descr:
            name = entry[0]
            value = x.dtype.fields.get(entry[0])[0]
            
            if value == 'object':
                entries = box[name][0]
                l = np.ones( (x.shape[0], len(entries)), dtype=np.bool)
                for i,entry in enumerate(entries):
                    if type(list(entries)[0]) not in (StringType, FloatType, IntType):
                        bools = []                
                        for element in list(x[name]):
                            if element == entry:
                                bools.append(True)
                            else:
                                bools.append(False)
                        l[:, i] = np.asarray(bools, dtype=bool)
                    else:
                        l[:, i] = x[name] == entry
                l = np.any(l, axis=1)
                logical = logical & l
            else:
                logical = logical & (box[name][0] <= x[name] )&\
                                        (x[name] <= box[name][1])                
        
        return self.yi_remaining[logical]
   
    def determine_coi(self, indices):
        '''
        
        Given a set of indices on y, how many cases of interest are there in 
        this set.
        
        :param indices: a valid index for y
        :raises: ValueError if threshold_type is not either ABOVE or BELOW
        :returns: the nr. of cases of interest.
        
        '''
        
        y = self.y[indices]
        
        if self.threshold_type == ABOVE:
            coi = y[y >= self.threshold].shape[0]
        elif self.threshold_type == BELOW:
            coi = y[y <= self.threshold].shape[0]
        else:
            raise ValueError("threshold type is not one of ABOVE or BELOW")
        
        return coi
    
    def determine_restricted_dims(self, box_lims):
        '''
        
        determine the number of restriced dimensions of a box given
        compared to the inital box that contains all the data
        
        :param box_lims: 
        
        '''
    
        dims = np.ones((len(box_lims.dtype.descr),))
        logical = self.compare(self.box_init, box_lims)
        dims = dims[logical==False]
        return np.sum(dims)
    
    def make_box(self, x):
        box = np.zeros((2, ), x.dtype)
        for entry in x.dtype.descr:
            name = entry[0]
            value = x.dtype.fields.get(entry[0])[0] 
            if value == 'object':
                box[name][:] = set(x[name])
            else:
                box[name][0] = np.min(x[name], axis=0) 
                box[name][1] = np.max(x[name], axis=0)    
        return box  
    
    def __peel(self, box):
        '''
        
        Executes the peeling phase of the PRIM algorithm. Delegates peeling
        to data type specific helper methods.
        
        '''
        
        '''
        Peeling stage of PRIM 
        
        :param x: structured array of independent variables
        :param y: array of the independend variable
        :param box: box limits
        :param peel_alpha: param that controls the amount of data that is removed
                           in a single peel
        :param mass_min: the minimum mass that should be left inside the box
        :param threshold:
        :param n: the total number of data points
        :param obj_func: the objective function to be used in selecting the 
                         new box from a set of candidate peel boxes
        
        returns a tuple (mean, volume, box)
        '''
    
        mass_old = box.yi.shape[0]/self.n

        x = self.x[box.yi]
        y = self.y[box.yi]
       
        #identify all possible peels
        possible_peels = []
        
        for entry in x.dtype.descr:
            u = entry[0]
            dtype = x.dtype.fields.get(u)[0].name
            peels = self.__peels[dtype](self, box, u)
            [possible_peels.append(entry) for entry in peels] 

        possible_peels.sort(key=itemgetter(0,1), reverse=True)
        entry = possible_peels[0]
        box_new, indices, a_tmp = entry[2:]
        
        # this should only result in an update to yi_remaining
        # first as a temp / new, and if we continue, update yi_remaining
        mass_new = self.y[indices].shape[0]/self.n
        
        # what is the function of the fist condition?
        # self.threshold makes no sense here
        # second condition checks whether we are not peeling to much
        # third criterion makes sure that data actually has been removed
        # fourth criterion makes sure that the box is not empty
        # again this is rather silly given mass_min, unless mass_min==0
        # which would again be nonsensical
        if (mass_new >= self.mass_min) &\
           (mass_new < mass_old):
            # if best peel leaves remaining data
            # call peel again with updated box, x, and y
            self.yi_remaining = indices
            box.update(box_new, indices)
            return self.__peel(box)
        else:
            #else return received box
            return box
    
    
    def __real_peel(self, box, u):
        '''
        
        returns two candidate new boxes, peel along upper and lower dimension
        
        :param box: a PrimBox instance
        :param u: the uncertainty for which to peel
        :returns: two box_lims, associated value of obj_fuction, 
                  nr_restricted_dims
        
        
        '''
        
        x = self.x[box.yi]
        y = self.y[box.yi]

        peels = []
        for direction in ['upper', 'lower']:
            peel_alpha = self.peel_alpha
        
            i=0
            if direction=='upper':
                peel_alpha = 1-self.peel_alpha
                i=1
            
            box_peel = mquantiles(x[u], [peel_alpha], alphap=self.alpha, 
                                  betap=self.beta)[0]
            if direction=='lower':
                logical = x[u] >= box_peel
                indices = box.yi[logical]
            if direction=='upper':
                logical = x[u] <= box_peel
                indices = box.yi[logical]
        
            obj = self.obj_func(self, y,  self.y[indices])
            temp_box = copy.deepcopy(box.box_lims[-1])
            temp_box[u][i] = box_peel
            res_dim = self.determine_restricted_dims(temp_box)
            
            non_res_dim = len(x.dtype.descr)-res_dim

            a_tmp = np.sum(self.y[indices])
                        
            peels.append((obj, non_res_dim, temp_box, indices,a_tmp))
    
        return peels
    
    def __discrete_peel(self, box, u):
        '''
        
        returns two candidate new boxes, peel along upper and lower dimension
        
        :param box: a PrimBox instance
        :param u: the uncertainty for which to peel
        :returns: two boxlims, associated value of obj_fucntion, 
                  nr_restricted_dims

        
        '''
        print u
        pass
    
    def __categorical_peel(self, box, u):
        '''
        
        returns one candidate new box
        
        :param box: a PrimBox instance
        :param u: the uncertainty for which to peel
        :returns: one boxlims, associated value of obj_fucntion, 
                  nr_restricted_dims
        
        '''
        entries = box.box_lims[-1][u][0]

        x = self.x[box.yi]
        y = self.y[box.yi]
        
        if len(entries) > 1:
            peels = []
            for entry in entries:
                temp_box = np.copy(box)
                peel = copy.deepcopy(entries)
                peel.discard(entry)
                temp_box[u][:] = peel
                
                if type(list(entries)[0]) not in (StringType, FloatType, IntType):
                    bools = []                
                    for element in list(x[u]):
                        if element != entry:
                            bools.append(True)
                        else:
                            bools.append(False)
                    logical = np.asarray(bools, dtype=bool)
                else:
                    logical = x[u] != entry
    
                if y[logical].shape[0] != 0:
                    obj = self.obj_func(self, y, y[logical])
                else:
                    obj = 0
               
                res_dim = self.determine_restricted_dims(temp_box)
                indices = box.yi[logical]
                non_res_dim = len(x.dtype.descr)-res_dim
                
                peels.append((obj, non_res_dim, temp_box, indices))
            return peels
        else:
            # no peels possible, return empty list
            return []


    def __paste(self):
        '''
        
        Executates the pasting phase of the PRIM. Delegates pasting to data 
        type specific helper methods.
        
        '''
        
        
        pass
    
    def __def_obj_func(self, y_old, y_new):
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
        
        '''
        mean_old = np.mean(y_old)
        mean_new = np.mean(y_new)
        obj = 0
        if mean_old != mean_new:
            if y_old.shape[0] >= y_new.shape[0]:
                obj = (mean_new-mean_old)/(y_old.shape[0]-y_new.shape[0])
            else:
                obj = (mean_new-mean_old)/(y_new.shape[0]-y_old.shape[0])
        return obj



    __peels = {'object': __categorical_peel,
               'int': __discrete_peel,
               'float64': __real_peel}

#    __paste = {'object': __categorical_paste,
#               'int': __discrete_paste,
#               'float': __real_paste}

    # dict with the various objective functions available
    __obj_functions = {DEFAULT : __def_obj_func}    