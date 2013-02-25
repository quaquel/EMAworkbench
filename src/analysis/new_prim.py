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
from expWorkbench import info, debug

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
        self.yi = np.arange(0, self.y.shape[0])
       
        # how many data points do we have
        self.n = self.y.shape[0]
        
        # how many cases of interest do we have?
        self.t_coi = self.determine_coi(self.yi)
        
        # initial box that contains all data
        self.box_init = self.make_box(self.x)
    
        # make a list in which the identified boxes can be put
        self.boxes = []
        
        self.__update_yi_remaining()
    
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
        # set the indices
        self.__update_yi_remaining()

        info("{} points remaining".format(self.yi_remaining.shape[0]))
        info("{} cases of interest remaining".format(self.determine_coi(self.yi_remaining)))
        
        # make a new box
        box = PrimBox(self, self.box_init, self.yi_remaining[:])
        
        #  perform peeling phase
        new_box = self.__peel(box)
        debug("peeling completed")
    
#        if pasting:
#            logical = in_box(x_remaining, new_box.box)
#            x_inside = x_remaining[logical]
#            y_inside = y_remaining[logical]
#    
#            new_box = paste(x_inside, y_inside, x_remaining, y_remaining, 
#                               copy.copy(box_init), new_box, paste_alpha, mass_min, 
#                               threshold, n, obj_func)
#            info("pasting completed")
        
        # TODO check if box meets critiria, otherwise, return a 
        # dumpbox, and log that there is no box possible anymore
        
        self.boxes.append(new_box)
        self.__update_yi_remaining()
        
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
    
#    def __getattr__(self, name):
#        # TODO intercept gets on self.yi_remaining, call an update prior
#        # to returning the value
   
    def __update_yi_remaining(self):
        '''
        
        Update yi_remaining in light of the state of the boxes associated
        with this prim instance.
        
        
        '''
        
        # set the indices
        yi_remaining = self.yi
        
        logical = yi_remaining == yi_remaining
        for box in self.boxes:
            logical[box.yi] = False
        self.yi_remaining = yi_remaining[logical]
    
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
            peels = self.__peels[dtype](self, box, u, x)
            [possible_peels.append(entry) for entry in peels] 

        # determine the scores for each peel in order
        # to identify the next candidate box
        scores = []
        for entry in possible_peels:
            i, box_lim = entry
            obj = self.obj_func(self, y,  self.y[i])
            non_res_dim = len(x.dtype.descr)-\
                          self.determine_restricted_dims(box_lim)
            score = (obj, non_res_dim, box_lim, i)
            scores.append(score)

        scores.sort(key=itemgetter(0,1), reverse=True)
        entry = scores[0]
        box_new, indices = entry[2:]
        
        # this should only result in an update to yi_remaining
        # first as a temp / new, and if we continue, update yi_remaining
        mass_new = self.y[indices].shape[0]/self.n
        
        if (mass_new >= self.mass_min) &\
           (mass_new < mass_old):
            box.update(box_new, indices)
            return self.__peel(box)
        else:
            #else return received box
            return box
    
    
    def __real_peel(self, box, u, x):
        '''
        
        returns two candidate new boxes, peel along upper and lower dimension
        
        :param box: a PrimBox instance
        :param u: the uncertainty for which to peel
        :returns: two box_lims, associated value of obj_fuction, 
                  nr_restricted_dims
        
        
        '''
        

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
            temp_box = copy.deepcopy(box.box_lims[-1])
            temp_box[u][i] = box_peel
            peels.append((indices, temp_box))
    
        return peels
    
    def __discrete_peel(self, box, u, x):
        '''
        
        returns two candidate new boxes, peel along upper and lower dimension
        
        :param box: a PrimBox instance
        :param u: the uncertainty for which to peel
        :returns: two boxlims, associated value of obj_fucntion, 
                  nr_restricted_dims

        
        '''

        peels = []
        for direction in ['upper', 'lower']:
            peel_alpha = self.peel_alpha
        
            i=0
            if direction=='upper':
                peel_alpha = 1-self.peel_alpha
                i=1
            
            box_peel = mquantiles(x[u], [peel_alpha], alphap=self.alpha, 
                                  betap=self.beta)[0]
            box_peel = int(box_peel)

            # determine logical associated with peel value            
            if direction=='lower':
                if box_peel == box.box_lims[-1][u][i]:
                    logical = (x[u] > box.box_lims[-1][u][i]) &\
                              (x[u] <= box.box_lims[-1][u][i+1])
                else:
                    logical = (x[u] >= box_peel) &\
                              (x[u] <= box.box_lims[-1][u][i+1])
            if direction=='upper':
                if box_peel == box.box_lims[-1][u][i]:
                    logical = (x[u] < box.box_lims[-1][u][i]) &\
                              (x[u] >= box.box_lims[-1][u][i-1])
                else:
                    logical = (x[u] <= box_peel) &\
                              (x[u] >= box.box_lims[-1][u][i-1])

            # determine value of new limit given logical
            if x[logical].shape[0] == 0:
                new_limit = np.min(x[u])
            else:
                new_limit = np.min(x[u][logical])            
            
            indices= box.yi[logical] 
            temp_box = copy.deepcopy(box.box_lims[-1])
            temp_box[u][i] = new_limit
            peels.append((indices, temp_box))
    
        return peels
    
    def __categorical_peel(self, box, u, x):
        '''
        
        returns one candidate new box
        
        :param box: a PrimBox instance
        :param u: the uncertainty for which to peel
        :param x: the x in box
        :returns: one boxlims, associated value of obj_fucntion, 
                  nr_restricted_dims
        
        '''
        entries = box.box_lims[-1][u][0]
        
        if len(entries) > 1:
            peels = []
            for entry in entries:
                temp_box = np.copy(box.box_lims[-1])
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
                indices = box.yi[logical]
                peels.append((indices,  temp_box))
            return peels
        else:
            # no peels possible, return empty list
            return []


    def __paste(self, box):
        '''
        
        Executates the pasting phase of the PRIM. Delegates pasting to data 
        type specific helper methods.
        
        '''
        mass = self.y.shape[0]/self.n
        y_mean = np.mean(self.y)
        
        possible_pastes = []
        for entry in self.x.dtype.descr:
            u = entry[0]
            dtype = self.x.dtype.fields.get(u)[0].name
            pastes = self.__pastes[dtype](self, box, u)
            [possible_pastes.append(entry) for entry in pastes] 
        
        print "blaat"
    
#        #break ties by choosing box with largest mass                 
#        possible_pastes.sort(key=itemgetter(0,1), reverse=True)
        obj, mass_new, box_new = possible_pastes[0]
        logical = self.in_box(box_new)
        x_new = self.x[self.yi_remaining][logical]
        y_new = self.y[[self.yi_remaining]][logical]
        y_mean_new = np.mean(y_new)
#       
        if (y_mean_new > self.threshold) &\
           (mass_new >= self.mass_min) &\
           (y_mean_new >= y_mean) &\
           (mass_new > mass):
            
            box.update(x_new, y_new, box_new, mass_new)
            return self.__paste(box)
        else:
            return box

    def __real_paste(self, u, box):
       
        box_diff = self.box_init[u][1]-self.box_init[u][0]
        
        box_paste = np.copy(box.box_lims[-1])
        test_box = np.copy(box.box_lims[-1])
    
        pa = self.paste_alpha * box.yi.shape[0]
    
        pastes = []
        for direction in ['upper', 'lower']:
            if direction == 'lower':
                i = 0
                box_diff = -1*box_diff
                test_box[u][1] = test_box[u][i]
                test_box[u][i] = self.box_init[u][i]
                logical = self.in_box(test_box)
                data = self.x[self.yi_remaining][logical][u]
                
                if data.shape[0] > 0:
                    b = (data.shape[0]-pa)/data.shape[0]
                    paste_value = mquantiles(data, [b], alphap=self.alpha, 
                                             betap=self.beta)[0]
            elif direction == 'upper':
                i = 1
                test_box[u][0] = test_box[u][i]
                test_box[u][i] = self.box_init[u][i]
                logical = self.in_box(test_box)
                data = self.x[self.yi_remaining][logical][u]
                
                if data.shape[0] > 0:
                    b = (pa)/data.shape[0]
                    paste_value = mquantiles(data, [b], alphap=self.alpha, 
                                             betap=self.beta)[0]
            box_paste[u][i] = paste_value
            indices = self.in_box(box_paste)
            pastes.append(indices, box_paste)
    
        return pastes        
    
    def __discrete_paste(self, x_init,y_init, y, name,
                  box,box_init, paste_alpha, n, direction, obj_func):
        pass
#        box_diff = box_init[name][1]-box_init[name][0]
#        if direction == 'lower':
#            i = 0
#            paste_alpha = 1-paste_alpha
#            box_diff = -1*box_diff
#        if direction == 'upper':
#            i = 1
#        
#        box_paste = np.copy(box)
#        y_paste = y
#        test_box = np.copy(box)
#      
#        if direction == 'lower':
#            test_box[name][i+1] = test_box[name][i]
#            test_box[name][i] = box_init[name][i]
#            logical = in_box(x_init, test_box)
#            data = x_init[logical][name]
#            if data.shape[0] > 0:
#                a = paste_alpha * y.shape[0]
#                b = (data.shape[0]-a)/data.shape[0]
#                paste_value = mquantiles(data, [b], alphap=1/3, betap=1/3)[0]
#                paste_value = int(round(paste_value))
#                box_paste[name][i] = paste_value
#                logical = in_box(x_init, box_paste)
#                y_paste = y_init[logical]
#        
#        if direction == 'upper':
#            test_box[name][i-1] = test_box[name][i]
#            test_box[name][i] = box_init[name][i]
#            logical = in_box(x_init, test_box)
#            data = x_init[logical][name]
#            if data.shape[0] > 0:
#                a = paste_alpha * y.shape[0]
#                b = a/data.shape[0]
#                paste_value = mquantiles(data, [b], alphap=1/3, betap=1/3)[0]
#                paste_value = int(round(paste_value))
#                box_paste[name][i] = paste_value
#                logical = in_box(x_init, box_paste)
#                y_paste = y_init[logical]
#    
#        # y means of pasted boxes
#        obj = obj_func(y,  y_paste)
#        
#        # mass of pasted boxes
#        mass_paste = y_init[logical].shape[0]/n
#    
#        return (obj, mass_paste, box_paste)
    
    def __categorical_paste(self, x_init,y_init, y, name, box,n, obj_func):
        pass
#        c_in_b = box[name][0]
#        c_t = set(x_init[name])
#        
#        if len(c_in_b) < len(c_t):
#            pastes = []
#            possible_cs = c_t - c_in_b
#            for entry in possible_cs:
#                temp_box = np.copy(box)
#                paste = copy.deepcopy(c_in_b)
#                paste.add(entry)
#                temp_box[name][:] = paste
#                logical = in_box(x_init, box)
#                obj = obj_func(y,  y_init[logical])
#                mass_paste = y_init[logical].shape[0]/n
#                pastes.append((obj, mass_paste, temp_box))
#            return pastes
#        else:
#            # no pastes possible, return empty list
#            return []
    
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
               'int32': __discrete_peel,
               'float64': __real_peel}

    __pastes = {'object': __categorical_paste,
               'int32': __discrete_paste,
               'float64': __real_paste}

    # dict with the various objective functions available
    __obj_functions = {DEFAULT : __def_obj_func}    