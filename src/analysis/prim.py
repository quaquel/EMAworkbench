'''
Created on 22 feb. 2013

@author: j.h.kwakkel

'''
from __future__ import division
from types import StringType, FloatType, IntType
from operator import itemgetter
import copy
import math

import numpy as np
import numpy.lib.recfunctions as recfunctions
from scipy.stats import binom

from mpl_toolkits.axes_grid1 import host_subplot  # @UnresolvedImport
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec 

import pandas as pd

from analysis.plotting_util import make_legend, COLOR_LIST
from expWorkbench import info, debug, EMAError, ema_logging
from analysis import pairs_plotting

DEFAULT = 'default'
ORIGINAL = 'original'

ABOVE = 1
BELOW = -1
PRECISION = '.2f'

def get_quantile(data, quantile):
    '''
    quantile calculation modeled on the implementation used in sdtoolkit
    
    this replaces the scipy.stats.mquantile that has been used before
    
    :param data: dataset for which quantile is needed
    :param quantile: the desired quantile
    
    '''
    assert quantile>0
    assert quantile<1
 
    data = data.compressed()
    data = list(data) #TODO do we really need a list? I doubt it
    data.sort()    
    
    i = (len(data)-1)*quantile
    index_lower =  int(math.floor(i))
    index_higher = int(math.ceil(i))
    
    value = 0

    if quantile > 0.5:
        # upper
        while (data[index_lower] == data[index_higher]) & (index_lower>0):
            index_lower -= 1
        value = (data[index_lower]+data[index_higher])/2
    else:
        #lower
        while (data[index_lower] == data[index_higher]) & (index_higher<len(data)-1):
            index_higher += 1
        value = (data[index_lower]+data[index_higher])/2


    return value

def _determine_size(box, uncertainties):
    '''helper function for determining spacing when writing boxlims to stdout
    
    :param box: a box definition, used only to acquire datatype
    :param uncertainties: a list of uncertainties to be printed
    
    fill the limits in for each uncertainty and each box
    determine the length of the uncertainty names to align these properly
    determine size of values in box_lims, this should be based on the integers 
    and floats only
    
    '''
    
    length = max([len(u) for u in uncertainties])
    length = max((length, len('uncertainty')))
    
    size = 0
    for u in uncertainties:
        data_type =  box[u].dtype
        if data_type == np.float64:
            size = max(size, 
                       len("{:>{}}".format(box[u][0], PRECISION)), 
                       len("{:>{}}".format(box[u][1], PRECISION)))
        elif data_type == np.int32:
            size = max(size, 
                       len("{:>}".format(box[u][0])), 
                       len("{:>}".format(box[u][1])))   
        elif data_type == np.object:
            s = len("{}".format(box[u][0]))
            s = int(s/2)-4
            size = max(size,
                       s)
    size = size+4
    return length, size


def _write_boxes_to_stdout(box_lims, uncertainties):
    '''
    
    write the lims for the uncertainties for each box lim to stdout
    
    :param box_lims: list of box_lims
    :param uncertainties: list of uncertainties
    
    '''

    box = box_lims[-1]
    length, size = _determine_size(box, uncertainties)

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
                data.append(PRECISION)
                
                elements.append("{0:>{2}{4}} -{1:>{3}{4}}".format(*data))
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

def _setup_figure(uncs):
    '''
    
    helper function for creating the basic layout for the figures that
    show the box lims.
    
    '''
    nr_unc = len(uncs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # create the shaded grey background
    rect = mpl.patches.Rectangle((0, -0.5), 1, nr_unc+1.5,
                                 alpha=0.25,  
                                 facecolor="#C0C0C0",
                                 edgecolor="#C0C0C0")
    ax.add_patch(rect)
    ax.set_xlim(xmin=-0.2, xmax=1.2)
    ax.set_ylim(ymin= -0.5, ymax=nr_unc-0.5)
    ax.yaxis.set_ticks([y for y in range(nr_unc)])
    ax.xaxis.set_ticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(uncs[::-1]) 
    return fig, ax

def _pair_wise_scatter(x,y, box_lim, restricted_dims):
    ''' helper function for pair wise scatter plotting
    
    :param x: the experiments
    :param y: the outcome of interest
    :param box_lim: a boxlim
    :param restricted_dims: list of uncertainties that define the boxlims
    
    '''

    restricted_dims = list(restricted_dims)
    combis = [(field1, field2) for field1 in restricted_dims\
                               for field2 in restricted_dims]

    grid = gridspec.GridSpec(len(restricted_dims), len(restricted_dims))                             
    grid.update(wspace = 0.1,
                hspace = 0.1)    
    figure = plt.figure()
    
    for field1, field2 in combis:
        i = restricted_dims.index(field1)
        j = restricted_dims.index(field2)
        ax = figure.add_subplot(grid[i,j])        
        ec='b'
        fc='b'
        
        if field1==field2:
            ec='white'
            fc='white'
        
        # scatter points
        for n in [0,1]:
            x_n = x[y==n]        
            x_1 = x_n[field2]
            x_2 = x_n[field1]
            
            if (n==0) :
                fc='white'
            elif ec=='b':
                fc='b'
            
            ax.scatter(x_1, x_2, facecolor=fc, edgecolor=ec, s=10)

        # draw boxlim
        if field1 != field2:
            x_1 = box_lim[field2]
            x_2 = box_lim[field1]

            for n in [0,1]:
                ax.plot(x_1,
                    [x_2[n], x_2[n]], c='r', linewidth=3)
                ax.plot([x_1[n], x_1[n]],
                    x_2, c='r', linewidth=3)
            
        #reuse labeling function from pairs_plotting
        pairs_plotting.do_text_ticks_labels(ax, i, j, field1, field2, None, restricted_dims)
            
    return figure
        
def _in_box(x, boxlim):
    '''
     
    returns the indices of the data points that are within the 
    box_lims.
    
    '''
    logical = np.ones(x.shape[0], dtype=np.bool)
    
    dims = recfunctions.get_names(boxlim.dtype)

    for name in dims:
        value = x.dtype.fields.get(name)[0]
        
        if value == 'object':
            entries = boxlim[name][0]
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
            logical = logical & (boxlim[name][0] <= x[name] )&\
                                        (x[name] <= boxlim[name][1])                
    
    indices = np.where(logical==True)
    
    assert len(indices)==1
    indices = indices[0]
    
    return indices


class CurEntry(object):
    '''a descriptor for the current entry on the peeling and pasting trajectory'''
    
    def __init__(self, name):
        self.name = name
        
    def __get__(self, instance, owner):
        return instance.peeling_trajectory[self.name][instance._cur_box]
    
    def __set__(self, instance, value):
        raise PrimException("this property cannot be assigned to")
                

class PrimBox(object):
    stats_format = "{0:<5}{mean:>10.2g}{mass:>10.2g}{coverage:>10.2g}{density:>10.2g}{restricted_dim:>10.2g}"
    stats_header = "{0:<5}{1:>10}{2:>10}{3:>10}{4:>10}{5:>10}".format('box', 
                              'mean', 'mass', 'coverage', 'density', 'res dim')
    
    coverage = CurEntry('coverage')
    density = CurEntry('density')
    mean = CurEntry('mean')
    res_dim = CurEntry('res dim')
    mass = CurEntry('mass')
    
    _frozen=False
    
    def __init__(self, prim, box_lims, indices):
        self.prim = prim
        
        # peeling and pasting trajectory
        colums = ['coverage', 'density', 'mean', 'res dim', 'mass']
        self.peeling_trajectory = pd.DataFrame(columns=colums)
        
        self.box_lims = []
        self._cur_box = -1
        
        # indices van data in box
        self.update(box_lims, indices)

    def __getattr__(self, name):
        '''
        used here to give box_lim same behaviour as coverage, density, mean
        res_dim, and mass. That is, it will return the box lim associated with
        the currently selected box. 
        '''
        
        if name=='box_lim':
            return self.box_lims[self._cur_box]
        else:
            raise AttributeError

    def inspect(self, i=None):
        '''
        
        Write the stats and box limits of the user specified box to standard 
        out. if i is not provided, the last box will be printed
        
        '''
      
        print self.stats_header
        
        if i == None:
            i = len(self.box_lims)-1
        
        stats = self.peeling_trajectory.iloc[i].to_dict()
        stats['restricted_dim'] = stats['res dim']
        
        print self._format_stats(i, stats)   
        print ""
                
        qp_values = self._calculate_quasi_p(i)
        uncs = [(key, value) for key, value in qp_values.iteritems()]
        uncs.sort(key=itemgetter(1))
        uncs = [uncs[0] for uncs in uncs]

        qp_col_size = len("qp values")+4
        box = self.box_lims[i]
        unc_col_size, value_col_size = _determine_size(box, uncs)
        
        # make the headers of the limits table
        # first header is box names
        # second header is min and max
        elements_1 = ["{0:<{1}}".format("uncertainty", unc_col_size)]
        elements_2 = ["{0:<{1}}".format("", unc_col_size)]
        box_name = 'box {}'.format(i)
        elements_1.append("{0:>{2}}{1:>{3}}".format("{}".format(box_name),"", value_col_size+4, value_col_size-2))
        elements_2.append("{0:>{3}}{1:>{4}}{2:>{5}}".format("min", "max","qp values", value_col_size, value_col_size+2, qp_col_size))
        line = "".join(elements_1)
        print line
        line = "".join(elements_2)
        print line
        
        for u in uncs:
            elements = ["{0:<{1}}".format(u, unc_col_size)]

            data_type =  box[u].dtype
            if data_type == np.float64:
                data = list(box[u])
                data.append(value_col_size)
                data.append(value_col_size)
                data.append(PRECISION)
                
                elements.append("{0:>{2}{4}} -{1:>{3}{4}}".format(*data))
            elif data_type == np.int32:
                data = list(box[u])
                data.append(value_col_size)
                data.append(value_col_size)                
                
                elements.append("{0:>{2}} -{1:>{3}}".format(*data))            
            else:
                elements.append("{0:>{1}}".format(box[u][0], value_col_size*2+2))
            
            elements.append("{0:>{1}{2}}".format(qp_values[u], qp_col_size, '.2e'))
            
            line = "".join(elements)
            print line
        print "\n"        
        
    def select(self, i):
        '''        
        select an entry from the peeling and pasting trajectory and update
        the prim box to this selected box.
        
        :param i: the index of the box to select
        
        '''
        if self._frozen:
            raise PrimException("""box has been frozen because PRIM has found 
                                at least one more recent box""")
        
        indices = _in_box(self.prim.x[self.prim.yi_remaining], self.box_lims[i])
        self.yi = self.prim.yi_remaining[indices]
        self._cur_box = i

    def drop_restriction(self, uncertainty):
        '''
        drop the restriction on the specified dimension. That is, replace
        the limits in the chosen box with a new box where for the specified 
        uncertainty the limits of the initial box are being used. The resulting
        box is added to the peeling trajectory.
        
        :param uncertainty:
        
        '''
        
        new_box_lim = copy.deepcopy(self.box_lim)
        new_box_lim[uncertainty][:] = self.box_lims[0][uncertainty][:]
        indices = _in_box(self.prim.x[self.prim.yi_remaining], new_box_lim)
        indices = self.prim.yi_remaining[indices]
        self.update(new_box_lim, indices)
        
    def update(self, box_lims, indices):
        '''
        
        update the box to the provided box limits.
        
        
        :param box_lims: the new box_lims
        :param indices: the indices of y that are inside the box
      
        '''
        self.yi = indices
        
        y = self.prim.y[self.yi]

        self.box_lims.append(box_lims)

        coi = self.prim.determine_coi(self.yi)

        data = {'coverage':coi/self.prim.t_coi, 
                'density':coi/y.shape[0],  
                'mean':np.mean(y),
                'res dim':self.prim.determine_nr_restricted_dims(self.box_lims[-1]),
                'mass':y.shape[0]/self.prim.n}
        new_row = pd.DataFrame([data])
        self.peeling_trajectory = self.peeling_trajectory.append(new_row, ignore_index=True)
        
        self._cur_box = len(self.peeling_trajectory)-1
        
        
    def show_ppt(self):
        '''show the peeling and pasting trajectory in a figure'''
        
        ax = host_subplot(111)
        ax.set_xlabel("peeling and pasting trajectory")
        
        par = ax.twinx()
        par.set_ylabel("nr. restricted dimensions")
            
        ax.plot(self.peeling_trajectory['mean'], label="mean")
        ax.plot(self.peeling_trajectory['mass'], label="mass")
        ax.plot(self.peeling_trajectory['coverage'], label="coverage")
        ax.plot(self.peeling_trajectory['density'], label="density")
        par.plot(self.peeling_trajectory['res dim'], label="restricted dims")
        ax.grid(True, which='both')
        ax.set_ylim(ymin=0,ymax=1)
        
        fig = plt.gcf()
        
        make_legend(['mean', 'mass', 'coverage', 'density', 'restricted_dim'],
                    fig, ncol=5, alpha=1)
        return fig
    
    def show_tradeoff(self):
        '''Visualise the trade off between coverage and density. Color is used
        to denote the number of restricted dimensions.'''
       
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        cmap = mpl.cm.jet #@UndefinedVariable
        boundaries = np.arange(-0.5, 
                               max(self.peeling_trajectory['res dim'])+1.5, 
                               step=1)
        ncolors = cmap.N
        norm = mpl.colors.BoundaryNorm(boundaries, ncolors)
        
        p = ax.scatter(self.peeling_trajectory['coverage'], 
                       self.peeling_trajectory['density'], 
                       c=self.peeling_trajectory['res dim'], norm=norm)
        ax.set_ylabel('density')
        ax.set_xlabel('coverage')
        ax.set_ylim(ymin=0, ymax=1.2)
        ax.set_xlim(xmin=0, xmax=1.2)
        
        ticklocs = np.arange(0, 
                             max(self.peeling_trajectory['res dim'])+1, 
                             step=1)
        cb = fig.colorbar(p, spacing='uniform', ticks=ticklocs, drawedges=True)
        cb.set_label("nr. of restricted dimensions")
        return fig
    
    def show_pairs_scatter(self):
        '''
        
        make a pair wise scatter plot of all the restricted dimensions
        with colour denoting whether a given point is of interest or not
        and the boxlims superimposed on top.
        
        '''   
        return _pair_wise_scatter(self.prim.x, self.prim.y, self.box_lim, 
                           self.prim.determine_restricted_dims(self.box_lim))
    
    def write_ppt_to_stdout(self):
        '''write the peeling and pasting trajectory to stdout'''
        print self.peeling_trajectory
        print "\n"

    def _calculate_quasi_p(self, i):
        '''helper function for calculating quasi-p values as discussed in 
        Bryant and Lempert (2010). This is in essence a one sided 
        binominal test. 
        
        :param i: the specific box in the peeling trajectory for which the 
                  quasi-p values are to be calculated
        
        '''
        
        box_lim = self.box_lims[i]
        restricted_dims = list(self.prim.determine_restricted_dims(box_lim))
        
        # total nr. of cases in box
        Tbox = self.peeling_trajectory['mass'][i] * self.prim.n 
        
        # total nr. of cases of interest in box
        Hbox = self.peeling_trajectory['coverage'][i] * self.prim.t_coi  
        
        qp_values = {}
        
        for u in restricted_dims:
            temp_box = copy.deepcopy(box_lim)
            temp_box[u] = self.box_lims[0][u]
        
            indices = _in_box(self.prim.x[self.prim.yi_remaining], temp_box)
            indices = self.prim.yi_remaining[indices]
            
            # total nr. of cases in box with one restriction removed
            Tj = indices.shape[0]  
            
            # total nr. of cases of interest in box with one restriction 
            # removed
            Hj = np.sum(self.prim.y[indices])
            
            p = Hj/Tj
            
            Hbox = int(Hbox)
            Tbox = int(Tbox)
            
            qp = binom.sf(Hbox-1, Tbox, p)
            qp_values[u] = qp
            
        return qp_values

    def _format_stats(self, nr, stats):
        '''helper function for formating box stats'''
        row = self.stats_format.format(nr,**stats)
        return row

class PrimException(Exception):
    pass

class Prim(object):
    message = "{0} points remaining, containing {1} cases of interest"
    
    def __init__(self, 
                 results,
                 classify, 
                 obj_function=DEFAULT, 
                 peel_alpha = 0.05, 
                 paste_alpha = 0.05,
                 mass_min = 0.05, 
                 threshold = None, 
                 threshold_type=ABOVE,
                 incl_unc=[]):
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
        :param incl_unc: optional argument, should be a list of uncertainties
                         that are to be included in the prim analysis. 
        :raises: PrimException if data resulting from classify is not a 
                 1-d array. 
        :raises: TypeError if classify is not a string or a callable.
                     
        '''
        assert threshold!=None
        if not incl_unc:
            self.x = np.ma.array(results[0])
        else:
            drop_names = set(recfunctions.get_names(results[0].dtype))-set(incl_unc)
            self.x = recfunctions.drop_fields(results[0], drop_names, asrecarray = True)
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
    
    def perform_pca(self, subsets=None, exclude=set()):
        '''
        
        WARNING:: code still needs to be tested!!!
        
        Pre-process the data by performing a pca based rotation on it. 
        This effectively turns the algorithm into PCA-PRIM as described
        in the envsoft paper
        
        :param subsets: optional kwarg, expects a dictionary with group name 
                        as key and a list of uncertainty names as values. 
                        If this is used, a constrained PCA-PRIM is executed
                        **note:** the list of uncertainties should not 
                        contain categorical uncertainties. 
        :param exclude: optional kwarg, the uncertainties that should be 
                        excluded. TODO: from what?
        
        '''
        
        #transform experiments to numpy array
        dtypes = self.x.dtype.fields
        object_dtypes = [key for key, value in dtypes.items() if value[0]==np.dtype(object)]
        
        #get experiments of interest
        # TODO this assumes binary classification!!!!!!!
        logical = self.y>=self.threshold
        
        # if no subsets are provided all uncertainties with non dtype object are
        # in the same subset, the name of this is r, for rotation
        if not subsets:
            subsets = {"r":[key for key, value in dtypes.items() if value[0].name!=np.dtype(object)]}
        
        # remove uncertainties that are in exclude and check whether 
        # uncertainties occur in more then one subset
        seen = set()
        for key, value in subsets.items():
            value = set(value) - set(exclude)

            subsets[key] = list(value)
            if (seen & value):
                raise EMAError("uncertainty occurs in more then one subset")
            else:
                seen = seen | set(value)
        
        #prepare the dtypes for the new rotated experiments recarray
        new_dtypes = []
        for key, value in subsets.items():
            self._assert_dtypes(value, dtypes)
            
            # the names of the rotated columns are based on the group name 
            # and an index
            [new_dtypes.append(("%s_%s" % (key, i), float)) for i in range(len(value))]
        
        #add the uncertainties with object dtypes to the end
        included_object_dtypes = set(object_dtypes)-set(exclude)
        [new_dtypes.append((name, object)) for name in included_object_dtypes ]
        
        #make a new empty recarray
        rotated_experiments = np.recarray((self.x.shape[0],),dtype=new_dtypes)
        
        #put the uncertainties with object dtypes already into the new recarray 
        for name in included_object_dtypes :
            rotated_experiments[name] = self.x[name]
        
        #iterate over the subsets, rotate them, and put them into the new recarray
        shape = 0
        for key, value in subsets.items():
            shape += len(value) 
        rotation_matrix = np.zeros((shape,shape))
        column_names = []
        row_names = []
        
        j = 0
        for key, value in subsets.items():
            data = self._rotate_subset(value, self.x, logical)
            subset_rotation_matrix, subset_experiments = data 
            rotation_matrix[j:j+len(value), j:j+len(value)] = subset_rotation_matrix
            [row_names.append(entry) for entry in value]
            j += len(value)
            
            for i in range(len(value)):
                name = "%s_%s" % (key, i)
                rotated_experiments[name] = subset_experiments[:,i]
                [column_names.append(name)]
        
        self.rotation_matrix = rotation_matrix
        self.column_names = column_names
        
        self.x = np.ma.array(rotated_experiments)
        self.box_init = self.make_box(self.x)
    
    def find_box(self):
        '''
        
        Execute one iteration of the PRIM algorithm. That is, find one
        box, starting from the current state of Prim. 
        
        
        '''
        # set the indices
        self._update_yi_remaining()
        
        # make boxes already found immutable 
        for box in self.boxes:
            box._frozen = True
        
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
        message = message.format(box.mean,
                                 box.mass,
                                 box.coverage,
                                 box.density,
                                 box.res_dim)

        if (self.threshold_type==ABOVE) &\
           (box.mean >= self.threshold):
            info(message)
            self.boxes.append(box)
            return box
        elif (self.threshold_type==BELOW) &\
           (box.mean <= self.threshold):
            info(message)
            self.boxes.append(box)
            return box
        else:
            # make a dump box
            info('box does not meet threshold criteria, value is {}, returning dump box'.format(box.mean[-1]))
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
        u = np.asarray(recfunctions.get_names(box_lims.dtype), 
                       dtype=object)
        dims = u[logical==False]
        return dims
    
    def make_box(self, x):
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
 
    def write_boxes_to_stdout(self):
        '''
        
        Write the stats and box limits of the identified boxes to standard 
        out. It will  write all the box lims and the inital box as rest box. 
        The uncertainties will be sorted based on how restricted they are
        in the first box. 
        
        '''
      
        print self.boxes[0].stats_header
        
        boxes = self.boxes[:]
        if not np.all(self.compare(boxes[-1].box_lims[-1], self.box_init)):
            self._update_yi_remaining()
            box = PrimBox(self, self.box_init, self.yi_remaining[:])
            boxes.append(box)
        
        for nr, box in enumerate(boxes):
            nr +=1
            if nr == len(boxes):
                nr = 'rest'
            
            stats = {'mean': box.mean, 
                    'mass': box.mass, 
                    'coverage': box.coverage, 
                    'density': box.density, 
                    'restricted_dim': box.res_dim}
            print box._format_stats(nr, stats)   

        print "\n"
        _write_boxes_to_stdout(*self._get_sorted_box_lims())

        
    def show_boxes(self, together=True):
        '''
        
        visualize the boxes.
        
        :param together: if true, all boxes will be shown in a single figure,
                         if false boxes will be shown in individual figures
        :returns: a single figure instance when plotting all figures
                  together, a list of figures otherwise. 
        
        '''
        
        # get the sorted box lims
        box_lims, uncs = self._get_sorted_box_lims()

        # normalize the box lims
        # we don't need to show the last box, for this is the 
        # box_init, which is visualized by a grey area in this
        # plot.
        norm_box_lims =  [self._normalize(box_lim, uncs) for 
                                        box_lim in box_lims[0:-1]]
                        
        if together:
            fig, ax = _setup_figure(uncs)
            
            for i, u in enumerate(uncs):
                # we want to have the most restricted dimension
                # at the top of the figure
                xi = len(uncs) - i - 1
                
                for j, norm_box_lim in enumerate(norm_box_lims):
                    self._plot_unc(xi, i, j, norm_box_lim, box_lims[j], u, ax)
           
            plt.tight_layout()
            return fig
        else:
            figs = []
            for j, norm_box_lim in enumerate(norm_box_lims):
                fig, ax = _setup_figure(uncs)
                figs.append(fig)
                for i, u in enumerate(uncs):
                    xi = len(uncs) - i - 1
                    self._plot_unc(xi, i, j, norm_box_lim, box_lims[j], u, ax)
        
                plt.tight_layout()
            return figs
   
    def _plot_unc(self, xi, i, j, norm_box_lim, box_lim, u, ax):
        '''
        
        :param xi: the row at which to plot
        :param i: the uncertainty being plotted
        :param j: the box being plotted
        :param u: the uncertainty being plotted:
        :param ax: the ax on which to plot
        
        '''

        dtype = self.box_init[u].dtype
            
        y = xi-j*0.1
        
        if dtype == object:
            elements = sorted(list(self.box_init[u][0]))
            max_value = (len(elements)-1)
            box_lim = box_lim[u][0]
            x = [elements.index(entry) for entry in 
                 box_lim]
            x = [entry/max_value for entry in x]
            y = [y] * len(x)
            
            ax.scatter(x,y,  edgecolor=COLOR_LIST[j],
                       facecolor=COLOR_LIST[j])
            
        else:
            ax.plot(norm_box_lim[i], (y, y),
                    COLOR_LIST[j])
   
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
        logical = np.ones(self.yi.shape[0],dtype=np.bool )
        for box in self.boxes:
            logical[box.yi] = False
        self.yi_remaining = self.yi[logical]
    
    def _peel(self, box):
        '''
        
        Executes the peeling phase of the PRIM algorithm. Delegates peeling
        to data type specific helper methods.

        :param box: box limits
        
        '''
    
        mass_old = box.yi.shape[0]/self.n

        x = self.x[box.yi]
       
        #identify all possible peels
        possible_peels = []
        for entry in x.dtype.descr:
            u = entry[0]
            dtype = x.dtype.fields.get(u)[0].name
            peels = self._peels[dtype](self, box, u, x)
            [possible_peels.append(entry) for entry in peels] 
        if not possible_peels:
            # there is no peel identified, so return box
            return box

        # determine the scores for each peel in order
        # to identify the next candidate box
        scores = []
        for entry in possible_peels:
            i, box_lim = entry
            obj = self.obj_func(self, self.y[box.yi],  self.y[i])
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
                
                box_peel = get_quantile(x[u], peel_alpha)
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
            
            box_peel = get_quantile(x[u], peel_alpha)
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
                if direction == 'upper':
                    new_limit = np.max(x[u])
                else:
                    new_limit = np.min(x[u])
            else:
                if direction =='upper':
                    new_limit = np.max(x[u][logical])
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
        
        mass_old = box.yi.shape[0]/self.n
        
        res_dim = self.determine_restricted_dims(box.box_lims[-1])
        
        possible_pastes = []
        for u in res_dim:
            debug("pasting "+u)
            dtype = self.x.dtype.fields.get(u)[0].name
            pastes = self._pastes[dtype](self, box, u)
            [possible_pastes.append(entry) for entry in pastes] 
        if not possible_pastes:
            # there is no peel identified, so return box
            return box
    
        # determine the scores for each peel in order
        # to identify the next candidate box
        scores = []
        for entry in possible_pastes:
            i, box_lim = entry
            obj = self.obj_func(self, self.y[box.yi],  self.y[i])
            non_res_dim = len(x.dtype.descr)-\
                          self.determine_nr_restricted_dims(box_lim)
            score = (obj, non_res_dim, box_lim, i)
            scores.append(score)

        scores.sort(key=itemgetter(0,1), reverse=True)
        entry = scores[0]
        box_new, indices = entry[2:]
        mass_new = self.y[indices].shape[0]/self.n
        
        mean_old = np.mean(self.y[box.yi])
        mean_new = np.mean(self.y[indices])
        
        if (mass_new >= self.mass_min) &\
           (mass_new > mass_old) &\
           (mean_old <= mean_new):
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

        pastes = []
        for i, direction in enumerate(['lower', 'upper']):
            box_paste = np.copy(box.box_lims[-1])
            paste_box = np.copy(box.box_lims[-1]) # box containing data candidate for pasting
            
            if direction == 'upper':
                paste_box[u][0] = paste_box[u][1]
                paste_box[u][1] = self.box_init[u][1]
                indices = _in_box(self.x[self.yi_remaining], paste_box)
                data = self.x[self.yi_remaining][indices][u]
                
                paste_value = self.box_init[u][i]
                if data.shape[0] > 0:
                    paste_value = get_quantile(data, self.paste_alpha)
                    
                assert paste_value >= box.box_lims[-1][u][i]
                    
            elif direction == 'lower':
                paste_box[u][0] = self.box_init[u][0]
                paste_box[u][1] = box_paste[u][0]
                
                indices = _in_box(self.x[self.yi_remaining], paste_box)
                data = self.x[self.yi_remaining][indices][u]
                
                paste_value = self.box_init[u][i]
                if data.shape[0] > 0:
                    paste_value = get_quantile(data, 1-self.paste_alpha)
           
                if not paste_value <= box.box_lims[-1][u][i]:
                    print paste_value, box.box_lims[-1][u][i]
            
            
            dtype = box_paste.dtype.fields[u][0]
            if dtype==np.int32:
                paste_value = np.int(paste_value)
            
            box_paste[u][i] = paste_value
            indices = _in_box(self.x[self.yi_remaining], box_paste)
            indices = self.yi_remaining[indices]
            
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
                indices = _in_box(self.x[self.yi_remaining], box_paste)
                pastes.append((indices, box_paste))
            return pastes
        else:
            # no pastes possible, return empty list
            return []
    
    def _default_obj_func(self, y_old, y_new):
        r'''
        the default objective function used by prim, instead of the original
        objective function, This function can cope with continuous, integer, 
        and categorical uncertainties. The basic idea is that the gain in mean
        is divided by the loss in mass. 
        
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
        
        if y_new.shape[0]>0:
            mean_new = np.mean(y_new)
        else:
            mean_new = 0
            
        obj = 0
        if mean_old != mean_new:
            if y_old.shape[0] > y_new.shape[0]:
                obj = (mean_new-mean_old)/(y_old.shape[0]-y_new.shape[0])
            elif y_old.shape[0] < y_new.shape[0]:
                obj = (mean_new-mean_old)/(y_new.shape[0]-y_old.shape[0])
            else:
                raise PrimException("mean is different, while shape is same, cannot be")
        return obj
    
    def _original_obj_fund(self, y_old, y_new):
        ''' The original objective function: the mean of the data inside the box'''
        
        if y_new.shape[0]>0:
            return np.mean(y_new)
        else:
            return -1    

    def _assert_dtypes(self, keys, dtypes):
        '''
        helper fucntion that checks whether none of the provided keys has
        a dtype object as value.
        '''
        
        for key in keys:
            if dtypes[key][0] == np.dtype(object):
                raise EMAError("%s has dtype object and can thus not be rotated" %key)
        return True

    def _rotate_subset(self, value, orig_experiments, logical): 
        '''
        rotate a subset
        
        :param value:
        :param orig_experiment:
        :param logical:
        
        '''
        list_dtypes = [(name, "<f8") for name in value]
        
        #cast everything to float
        drop_names = set(recfunctions.get_names(orig_experiments.dtype)) -set(value)
        orig_subset = recfunctions.drop_fields(orig_experiments, drop_names, asrecarray=True)
        subset_experiments = orig_subset.astype(list_dtypes).view('<f8').reshape(orig_experiments.shape[0], len(value))
 
        #normalize the data
        mean = np.mean(subset_experiments,axis=0)
        std = np.std(subset_experiments, axis=0)
        std[std==0] = 1 #in order to avoid a devision by zero
        subset_experiments = (subset_experiments - mean)/std
        
        #get the experiments of interest
        experiments_of_interest = subset_experiments[logical]
        
        #determine the rotation
        rotation_matrix =  self._determine_rotation(experiments_of_interest)
        
        #apply the rotation
        subset_experiments = np.dot(subset_experiments,rotation_matrix)
        return rotation_matrix, subset_experiments

    def _determine_rotation(self, experiments):
        '''
        Determine the rotation for the specified experiments
        
        :param experiments:
        
        '''
        covariance = np.cov(experiments.T)
        
        eigen_vals, eigen_vectors = np.linalg.eig(covariance)
    
        indices = np.argsort(eigen_vals)
        indices = indices[::-1]
        eigen_vectors = eigen_vectors[:,indices]
        eigen_vals = eigen_vals[indices]
        
        #make the eigen vectors unit length
        for i in range(eigen_vectors.shape[1]):
            eigen_vectors[:,i] / np.linalg.norm(eigen_vectors[:,i]) * np.sqrt(eigen_vals[i])
            
        return eigen_vectors

    _peels = {'object': _categorical_peel,
               'int32': _discrete_peel,
               'float64': _real_peel}

    _pastes = {'object': _categorical_paste,
               'int32': _real_paste,
               'float64': _real_paste}

    # dict with the various objective functions available
    _obj_functions = {DEFAULT : _default_obj_func,
                      ORIGINAL: _original_obj_fund}    