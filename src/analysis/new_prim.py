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


def _write_boxes_to_stdout(box_lims, uncertainties):
    '''
    
    write the lims for the uncertainties for each box lim to stdout
    
    :param box_lims: list of box_lims
    :param uncertainties: list of of uncertainties
    
    '''

    # fill the limits in for each uncertainty and each box
    # determine the length of the uncertainty names to align these properly
    #
    length = max([len(u) for u in uncertainties])
    length = max((length, len('uncertainty')))

    # determine size of values in box_lims
    # this should be based on the integers and floats only
    
    box = box_lims[-1]
    size = 0
    for u in uncertainties:
        data_type =  box[u].dtype
        if data_type == np.float64:
            size = max(size, 
                       len("{:>.2f}".format(box[u][0])), 
                       len("{:>.2f}".format(box[u][1])))
        elif data_type == np.int32:
            size = max(size, 
                       len("{:>.2}".format(box[u][0])), 
                       len("{:>.2}".format(box[u][1])))   
        elif data_type == np.object:
            s = len("{}".format(box[u][0]))
            s = int(s/2)-4
            size = max(size,
                       s)
    size = size+4

    # make the headers of the limits table
    # first header is box names
    # second header is min and max
    elements_1 = ["{0:<{1}}".format("uncertainty", length)]
    elements_2 = ["{0:<{1}}".format("", length)]
    for i in range(len(box_lims)):
        if i < len(box_lims)-1:
            box_name = 'box {}'.format(i+1)
        else:
            box_name = 'rest box'        
        
        elements_1.append("{0:>{2}}{1:>{3}}".format("{}".format(box_name),"", size+4, size-2))
        elements_2.append("{0:>{2}}{1:>{3}}".format("min", "max",size,size+2))
    line = "".join(elements_1)
    print line
    line = "".join(elements_2)
    print line
    
    for u in uncertainties:
        elements = ["{0:<{1}}".format(u, length)]
    
        for box in box_lims:
            data_type =  box[u].dtype
            if data_type == np.float64:
                data = list(box[u])
                data.append(size)
                data.append(size)
                
                elements.append("{0:>{2}.2f} -{1:>{3}.2f}".format(*data))
            elif data_type == np.int32:
                data = list(box[u])
                data.append(size)
                data.append(size)                
                
                elements.append("{0:>{2}} -{1:>{3}}".format(*data))            
            else:
                elements.append("{0:>{1}}".format(box[u][0], size*2+2))
        line = "".join(elements)
        print line
    print "\n\n"


class PrimBox(object):

    stats_format = "{0:<5}{mean:>10.2g}{mass:>10.2g}{coverage:>10.2g}{density:>10.2g}{restricted_dim:>10.2g}"
    stats_header = "{0:<5}{1:>10}{2:>10}{3:>10}{4:>10}{5:>10}".format('box', 
                              'mean', 'mass', 'coverage', 'density', 'res dim')
    
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
        

        self.yi = self.prim.in_box(self.box_lims[i])
        
        i = i+1 
        self.box_lims = self.box_lims[0:i]
        self.mean = self.mean[0:i]
        self.mass = self.mass[0:i]
        self.coverage = self.coverage[0:i]
        self.density = self.density[0:i]
        self.res_dim = self.res_dim[0:i]
        
        # after select, try to paste
        self.prim._paste(self)
       

    def update(self, box_lims, indices):
        '''
        
        update the box to the provided box limits.
        
        
        :param box_lims: the new box_lims
        :param indices: the indices of y that are inside the box
      
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
        self.res_dim.append(self.prim.determine_nr_restricted_dims(self.box_lims[-1]))
        
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

        print self.stats_header
        for i in range(len(self.box_lims)):
            stats = {'mean': self.mean[i], 
                     'mass': self.mass[i], 
                     'coverage': self.coverage[i], 
                     'density': self.density[i], 
                     'restricted_dim': self.res_dim[i]}            
            
            row = self._format_stats(i, stats)
            print row

    def _format_stats(self, nr, stats):
        row = self.stats_format.format(nr,**stats)
        return row


class PrimException(Exception):
    pass


class Prim(object):

    # parameters that control the mquantile calculation used
    # in peeling and pasting
    alpha = 1/3
    beta = 1/3
    
    message = "{0} point remaining, containing {1} cases of interest"
    
    def __init__(self, 
                 results,
                 classify, 
                 obj_function=DEFAULT, 
                 peel_alpha = 0.05, 
                 paste_alpha = 0.05,
                 mass_min = 0.05, 
                 threshold = None, 
                 threshold_type=ABOVE):
        '''
        
        :param results: the return from :meth:`perform_experiments`.
        :param classify: either a string denoting the outcome of interest to 
                         use or a function. 
        :param peel_alpha: parameter controlling the peeling stage (default = 0.05). 
        :param paste_alpha: parameter controlling the pasting stage (default = 0.05).
        :param mass_min: minimum mass of a box (default = 0.05). 
        :param threshold: the threshold of the output space that boxes should meet. 
        :param threshold_type: If 1, the boxes should go above the threshold, if -1
                               the boxes should go below the threshold, if 0, the 
                               algorithm looks for both +1 and -1.
        :param obj_func: The objective function to use. Default is 
                         :func:`def_obj_func`
        :raises: PrimException if data resulting from classify is not a 
                 1-d array. 
        :raises: TypeError if classify is not a string or a callable.
                     
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
        self.threshold_type = threshold_type
        self.obj_func = self._obj_functions[obj_function]
       
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
        
        self._update_yi_remaining()
    
    def perform_pca(self):
        '''
        
        Pre-process the data by performing a pca based rotation on it. 
        This effectively turns the algorithm into PCA-PRIM as described
        in the envsoft paper
        
        '''
        
        raise NotImplementedError()
    
    def find_box(self):
        '''
        
        Execute one iteration of the PRIM algorithm. That is, find one
        box, starting from the current state of Prim. 
        
        
        '''
        # set the indices
        self._update_yi_remaining()
        
        if self.yi_remaining.shape[0] == 0:
            info("no data remaining")
            return
        
        # log how much data and how many coi are remaining
        info(self.message.format(self.yi_remaining.shape[0],
                                 self.determine_coi(self.yi_remaining)))
        
        # make a new box that contains all the remaining data points
        box = PrimBox(self, self.box_init, self.yi_remaining[:])
        
        #  perform peeling phase
        box = self._peel(box)
        debug("peeling completed")

        # perform pasting phase        
        box = self._paste(box)
        debug("pasting completed")
        
        message = "mean: {0}, mass: {1}, coverage: {2}, density: {3} restricted_dimensions: {4}"
        message = message.format(box.mean[-1],
                                 box.mass[-1],
                                 box.coverage[-1],
                                 box.density[-1],
                                 box.res_dim[-1])

        if (self.threshold_type==ABOVE) &\
           (box.mean[-1] >= self.threshold):
            info(message)
            self.boxes.append(box)
            return box
        elif (self.threshold_type==BELOW) &\
           (box.mean[-1] <= self.threshold):
            info(message)
            self.boxes.append(box)
            return box
        else:
            # make a dump box
            info('box does not meet threshold criteria, returning dump box')
            box = PrimBox(self, self.box_init, self.yi_remaining[:])
            self.boxes.append(box)
            return box

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
        res_dim = self.determine_restricted_dims(box)
    
        for name in res_dim:
            value = x.dtype.fields.get(name)[0]
            
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
    
    def determine_nr_restricted_dims(self, box_lims):
        '''
        
        determine the number of restriced dimensions of a box given
        compared to the inital box that contains all the data
        
        :param box_lims: 
        
        '''
    
        return self.determine_restricted_dims(box_lims).shape[0]
    
    def determine_restricted_dims(self, box_lims):
        '''
        
        determine which dimensions of the given box are restricted compared 
        to compared to the initial box that contains all the data
        
        :param box_lims: 
        
        '''
    
        logical = self.compare(self.box_init, box_lims)
        u = np.asarray([entry[0] for entry in self.x.dtype.descr], 
                       dtype=object)
        dims = u[logical==False]
        return dims
    
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
    
#    def _getattr_(self, name):
#        # TODO intercept gets on self.yi_remaining, call an update prior
#        # to returning the value
   
    def write_boxes_to_stdout(self):
        '''
        
        Write the stats and box limits of the identified boxes to standard 
        out. It will  write all the box lims and the inital box as rest box. 
        The uncertainties will be sorted based on how restricted they are
        in the first box. 
        
        '''
      
        print self.boxes[0].stats_header
        i = -1
        
        boxes = self.boxes[:]
        if not np.all(self.compare(boxes[-1].box_lims[-1], self.box_init)):
            self._update_yi_remaining()
            box = PrimBox(self, self.box_init, self.yi_remaining[:])
            boxes.append(box)
        
        for nr, box in enumerate(boxes):
            nr +=1
            if nr == len(boxes):
                nr = 'rest'
            
            stats = {'mean': box.mean[i], 
                    'mass': box.mass[i], 
                    'coverage': box.coverage[i], 
                    'density': box.density[i], 
                    'restricted_dim': box.res_dim[i]}
            print box._format_stats(nr, stats)   

        print "\n"
        _write_boxes_to_stdout(*self._get_sorted_box_lims())

        
    
    def show_boxes(self, together=True):
        '''
        
        visualize the boxes.
        
        :param together: if true, all boxes will be shown in a single figure,
                         if false boxes will be shown in individual figures
        
        '''
        
        box_lims, uncs = self._get_sorted_box_lims()

        
        # normalize box_lim in het geval van een non obj unc
        # anders punten gebruiken, via een sort op een lijst versie van de
        # set van waarden
                
        if together:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            for u in uncs:
                dtype = self.box_init[u].dtype
                if dtype == object:
                    pass
                else:
                    pass
            
        else:
            pass
        
        raise NotImplementedError
   
   
    def _get_sorted_box_lims(self):

        # determine the uncertainties that are being restricted
        # in one or more boxes
        unc = set()
        for box in self.boxes:
            us  = self.determine_restricted_dims(box.box_lims[-1]).tolist()
            unc = unc.union(us)
        unc = np.asarray(list(unc))

        # normalize the range for the first box
        box_lim = self.boxes[0].box_lims[-1]
        nbl = self._normalize(box_lim, unc)
        box_size = nbl[:,1]-nbl[:,0]
        
        # sort the uncertainties based on the normalized size of the 
        # restricted dimensions
        unc = unc[np.argsort(box_size)]
        box_lims = [box.box_lims[-1] for box in self.boxes]

        if not np.all(self.compare(box_lims[-1], self.box_init)):
            box_lims.append(self.box_init)
        
        return box_lims, unc

    def _normalize(self, box_lim, unc):
        
        # normalize the range for the first box
        box_lim = self.boxes[0].box_lims[-1]
        norm_box_lim = np.zeros((len(unc), box_lim.shape[0]))
        for i, u in enumerate(unc):
            dtype = box_lim.dtype.fields[u][0]
            if dtype ==np.dtype(object):
                nu = len(box_lim[u][0])/ len(self.box_init[u][0])
                nl = 0
            else:
                lower, upper = box_lim[u]
                dif = (self.box_init[u][1]-self.box_init[u][0])
                a = 1/dif
                b = -1 * self.box_init[u][0] / dif
                nl = a * lower + b
                nu = a * upper + b
            norm_box_lim[i, :] = nl, nu
        return norm_box_lim
   
    def _update_yi_remaining(self):
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
    
    def _peel(self, box):
        '''
        
        Executes the peeling phase of the PRIM algorithm. Delegates peeling
        to data type specific helper methods.
        
        '''
        
        '''
        Peeling stage of PRIM 

        :param box: box limits
        
        
        '''
    
        mass_old = box.yi.shape[0]/self.n

        x = self.x[box.yi]
        y = self.y[box.yi]
       
        #identify all possible peels
        possible_peels = []
        for entry in x.dtype.descr:
            u = entry[0]
            dtype = x.dtype.fields.get(u)[0].name
            peels = self._peels[dtype](self, box, u, x)
            [possible_peels.append(entry) for entry in peels] 

        # determine the scores for each peel in order
        # to identify the next candidate box
        scores = []
        for entry in possible_peels:
            i, box_lim = entry
            obj = self.obj_func(self, y,  self.y[i])
            non_res_dim = len(x.dtype.descr)-\
                          self.determine_nr_restricted_dims(box_lim)
            score = (obj, non_res_dim, box_lim, i)
            scores.append(score)

        scores.sort(key=itemgetter(0,1), reverse=True)
        entry = scores[0]
        box_new, indices = entry[2:]
        
        mass_new = self.y[indices].shape[0]/self.n
       
        if (mass_new >= self.mass_min) &\
           (mass_new < mass_old):
            box.update(box_new, indices)
            return self._peel(box)
        else:
            #else return received box
            return box
    
    
    def _real_peel(self, box, u, x):
        '''
        
        returns two candidate new boxes, peel along upper and lower dimension
        
        :param box: a PrimBox instance
        :param u: the uncertainty for which to peel
        :returns: two box lims and the associated indices
        
        '''

        peels = []
        for direction in ['upper', 'lower']:
            
            if not np.any(np.isnan(x[u])):
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
            else:
                return []
    
        return peels
    
    def _discrete_peel(self, box, u, x):
        '''
        
        returns two candidate new boxes, peel along upper and lower dimension
        
        :param box: a PrimBox instance
        :param u: the uncertainty for which to peel
        :returns: two box lims and the associated indices

        
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
    
    def _categorical_peel(self, box, u, x):
        '''
        
        returns candidate new boxes for each possible removal of a single 
        category. So. if the box[u] is a categorical variable with 4 
        categories, this method will return 4 boxes. 
        
        :param box: a PrimBox instance
        :param u: the uncertainty for which to peel
        :returns: box lims and the associated indices
        
        
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


    def _paste(self, box):
        '''
        
        Executes the pasting phase of the PRIM. Delegates pasting to data 
        type specific helper methods.
        
     
        '''
        
        x = self.x[self.yi_remaining]
        y = self.y[self.yi_remaining]
        
        mass_old = box.yi.shape[0]/self.n
        
        res_dim = self.determine_restricted_dims(box.box_lims[-1])
        
        possible_pastes = []
        for u in res_dim:
            dtype = self.x.dtype.fields.get(u)[0].name
            pastes = self._pastes[dtype](self, box, u)
            [possible_pastes.append(entry) for entry in pastes] 
        
        # determine the scores for each peel in order
        # to identify the next candidate box
        scores = []
        for entry in possible_pastes:
            i, box_lim = entry
            obj = self.obj_func(self, y,  self.y[i])
            non_res_dim = len(x.dtype.descr)-\
                          self.determine_nr_restricted_dims(box_lim)
            score = (obj, non_res_dim, box_lim, i)
            scores.append(score)

        scores.sort(key=itemgetter(0,1), reverse=True)
        entry = scores[0]
        box_new, indices = entry[2:]
        
        mass_new = self.y[indices].shape[0]/self.n
        
        if (mass_new >= self.mass_min) &\
           (mass_new > mass_old):
            box.update(box_new, indices)
            return self._paste(box)
        else:
            #else return received box
            return box

    def _real_paste(self, box, u):
        '''
        
        returns two candidate new boxes, pasted along upper and lower 
        dimension
        
        :param box: a PrimBox instance
        :param u: the uncertainty for which to paste
        :returns: two box lims and the associated indices
       
        '''

        box_diff = self.box_init[u][1]-self.box_init[u][0]
        pa = self.paste_alpha * box.yi.shape[0]
    
        pastes = []
        for direction in ['upper', 'lower']:
            box_paste = np.copy(box.box_lims[-1])
            test_box = np.copy(box.box_lims[-1])
            
            if direction == 'lower':
                i = 0
                box_diff = -1*box_diff
                test_box[u][1] = test_box[u][i]
                test_box[u][i] = self.box_init[u][i]
                indices = self.in_box(test_box)
                data = self.x[indices][u]
                
                paste_value = self.box_init[u][i]
                if data.shape[0] > 0:
                    b = (data.shape[0]-pa)/data.shape[0]
                    paste_value = mquantiles(data, [b], alphap=self.alpha, 
                                             betap=self.beta)[0]
                
                    
            elif direction == 'upper':
                i = 1
                test_box[u][0] = test_box[u][i]
                test_box[u][i] = self.box_init[u][i]
                indices = self.in_box(test_box)
                data = self.x[indices][u]
                
                paste_value = self.box_init[u][i]
                if data.shape[0] > 0:
                    b = (pa)/data.shape[0]
                    paste_value = mquantiles(data, [b], alphap=self.alpha, 
                                             betap=self.beta)[0]
           
            box_paste[u][i] = paste_value
            indices = self.in_box(box_paste)
            
            pastes.append((indices, box_paste))
    
        return pastes        
    
    def _discrete_paste(self, box, u):
        '''
        
        returns two candidate new boxes, pasted along upper and lower 
        dimension
        
        :param box: a PrimBox instance
        :param u: the uncertainty for which to paste
        :returns: two box lims and the associated indices
       
        '''        
        box_diff = self.box_init[u][1]-self.box_init[u][0]
        pa = self.paste_alpha * box.yi.shape[0]
    
        pastes = []
        for direction in ['upper', 'lower']:
            box_paste = np.copy(box.box_lims[-1])
            test_box = np.copy(box.box_lims[-1])
            
            if direction == 'lower':
                i = 0
                box_diff = -1*box_diff
                test_box[u][1] = test_box[u][i]
                test_box[u][i] = self.box_init[u][i]
                indices = self.in_box(test_box)
                data = self.x[indices][u]
                
                paste_value = self.box_init[u][i]
                if data.shape[0] > 0:
                    b = (data.shape[0]-pa)/data.shape[0]
                    paste_value = mquantiles(data, [b], alphap=self.alpha, 
                                             betap=self.beta)[0]
                
                    
            elif direction == 'upper':
                i = 1
                test_box[u][0] = test_box[u][i]
                test_box[u][i] = self.box_init[u][i]
                indices = self.in_box(test_box)
                data = self.x[indices][u]
                
                paste_value = self.box_init[u][i]
                if data.shape[0] > 0:
                    b = (pa)/data.shape[0]
                    paste_value = mquantiles(data, [b], alphap=self.alpha, 
                                             betap=self.beta)[0]
           
            box_paste[u][i] = int(paste_value)
            indices = self.in_box(box_paste)
            
            pastes.append((indices, box_paste))
    
        return pastes    
        
    
    def _categorical_paste(self, box, u):
        '''
        
        Return a list of pastes, equal to the number of classes currently
        not on the box lim. 
        
        :param box: a PrimBox instance
        :param u: the uncertainty for which to paste
        :returns: a list of indices and box lims 
        
        
        '''
        box_lim = box.box_lims[-1]
        
        c_in_b = box_lim[u][0]
        c_t = self.box_init[u][0]
        
        if len(c_in_b) < len(c_t):
            pastes = []
            possible_cs = c_t - c_in_b
            for entry in possible_cs:
                box_paste = np.copy(box_lim)
                paste = copy.deepcopy(c_in_b)
                paste.add(entry)
                box_paste[u][:] = paste
                indices = self.in_box(box_paste)
                pastes.append((indices, box_paste))
            return pastes
        else:
            # no pastes possible, return empty list
            return []
    
    def _def_obj_func(self, y_old, y_new):
        r'''
        the default objective function used by prim, instead of the original
        objective function, this function can cope with continuous, integer, 
        and categorical uncertainties.      
        
        .. math::
            
            obj = \frac
                 {\text{ave} [y_{i}\mid x_{i}\in{B-b}] - \text{ave} [y\mid x\in{B}]}
                 {|n(y_{i})-n(y)|}
        
        where :math:`B-b` is the set of candidate new boxes, :math:`B` 
        the old box and :math:`y` are the y values belonging to the old 
        box. :math:`n(y_{i})` and :math:`n(y)` are the cardinality of 
        :math:`y_{i}` and :math:`y` respectively. So, this objective 
        function looks for the difference between  the mean of the old 
        box and the new box, divided by the change in the  number of 
        data points in the box. This objective function offsets a problem 
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



    _peels = {'object': _categorical_peel,
               'int32': _discrete_peel,
               'float64': _real_peel}

    _pastes = {'object': _categorical_paste,
               'int32': _discrete_paste,
               'float64': _real_paste}

    # dict with the various objective functions available
    _obj_functions = {DEFAULT : _def_obj_func}    