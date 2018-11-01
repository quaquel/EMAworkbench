'''

A scenario discovery oriented implementation of PRIM.

The implementation of prim provided here is datatype aware, so
categorical variables will be handled appropriately. It also uses a 
non-standard objective function in the peeling and pasting phase of the
algorithm. This algorithm looks at the increase in the mean divided 
by the amount of data removed. So essentially, it uses something akin
to the first order derivative of the original objective function. 

The implementation is designed for interactive use in combination with the
ipython notebook. 

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import copy
import math
from operator import itemgetter
import warnings

try:
    import altair as alt
except ImportError:
    alt = None
    warnings.warn(("altair based interactive "
                   "inspection not available"), ImportWarning)
    
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot  # @UnresolvedImport

import numpy as np
import pandas as pd
import seaborn as sns

from .plotting_util import make_legend
from ..util import info, debug, EMAError
from . import scenario_discovery_util as sdutil

# Created on 22 feb. 2013
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


__all__ = ['ABOVE', 'BELOW', 'setup_prim', 'Prim', 'PrimBox',
           'PrimException', 'MultiBoxesPrim']

LENIENT2 = 'lenient2'
LENIENT1 = 'lenient1'
ORIGINAL = 'original'

ABOVE = 1
BELOW = -1
PRECISION = '.2f'


def get_quantile(data, quantile):
    '''
    quantile calculation modeled on the implementation used in sdtoolkit

    Parameters
    ----------
    data : nd array like 
           dataset for which quantile is needed
    quantile : float
               the desired quantile

    '''
    assert quantile > 0
    assert quantile < 1

    data = np.sort(data)

    i = (len(data)-1)*quantile
    index_lower = int(math.floor(i))
    index_higher = int(math.ceil(i))

    value = 0

    if quantile > 0.5:
        # upper
        while (data[index_lower] == data[index_higher]) & (index_lower > 0):
            index_lower -= 1
        value = (data[index_lower]+data[index_higher])/2
    else:
        # lower
        while (data[index_lower] == data[index_higher]) & \
              (index_higher < len(data)-1):
            index_higher += 1
        value = (data[index_lower]+data[index_higher])/2

    return value


def _pair_wise_scatter(x, y, boxlim, box_init, restricted_dims):
    ''' helper function for pair wise scatter plotting

    #TODO the cases of interest should be in red rather than in blue
    # this will give a nice visual insight into the quality of the box
    # currently it is done through the face color being white or blue
    # this is not very clear


    Parameters
    ----------
    x : DataFrame
        the experiments
    y : numpy array
        the outcome of interest
    box_lim : DataFrame
              a boxlim
    box_init : DataFrame
    restricted_dims : collection of strings
                      list of uncertainties that define the boxlims

    '''
    

    x = x[restricted_dims]
    data = x.copy()
    
    # TODO:: have option to change 
    # diag to CDF, gives you effectively the 
    # regional sensitivity analysis results
    categorical_columns = data.select_dtypes('category').columns.values
    categorical_mappings = {}
    for column in categorical_columns:

        # reorder categorical data so we
        # can capture them in a single column
        categories_inbox = boxlim.loc[0, column]
        categories_all = box_init.loc[0, column]
        missing = categories_all - categories_inbox
        categories = list(categories_inbox) + list(missing)
        data[column] = data[column].cat.set_categories(categories)

        # keep the mapping for updating ticklabels
        categorical_mappings[column] = dict(enumerate(data[column].cat.categories))

        # replace column with codes
        data[column] = data[column].cat.codes

    data['y'] = y # for testing 
    grid = sns.pairplot(data=data, hue='y', vars=x.columns.values)

    cats = set(categorical_columns)
    for row, ylabel in zip(grid.axes, grid.y_vars):
        ylim = boxlim[ylabel]

        if ylabel in cats:
            y = -0.2
            height = len(ylim[0])-0.6 # 2 * 0.2
        else:
            y = ylim[0]
            height = ylim[1] - ylim[0]

        for ax, xlabel in zip(row, grid.x_vars):
            if ylabel == xlabel: continue

            if xlabel in cats:
                xlim = boxlim.loc[0, xlabel]
                x = -0.2
                width = len(xlim)-0.6 # 2 * 0.2
            else:
                xlim = boxlim[xlabel]
                x = xlim[0]
                width = xlim[1] - xlim[0]

            xy = x, y
            box = patches.Rectangle(xy, width, height, edgecolor='red',
                                    facecolor='none', lw=3)
            ax.add_patch(box)

    # do the yticklabeling for categorical rows
    for row, ylabel in zip(grid.axes, grid.y_vars):
        if ylabel in cats:
            ax = row[0]
            labels = []
            for entry in ax.get_yticklabels():
                _, value = entry.get_position()
                try:
                    label = categorical_mappings[ylabel][value]
                except KeyError:
                    label = ''
                labels.append(label)
            ax.set_yticklabels(labels)

    # do the xticklabeling for categorical columns
    for ax, xlabel in zip(grid.axes[-1], grid.x_vars):
        if xlabel in cats:
            labels = []
            locs = []
            mapping = categorical_mappings[ylabel]
            for i in range(-1, len(mapping)+1):
                locs.append(i)
                try:
                    label = categorical_mappings[xlabel][i]
                except KeyError:
                    label = ''
                labels.append(label)
            ax.set_xticks(locs)
            ax.set_xticklabels(labels, rotation=90)
    return grid

def setup_prim(results, classify, threshold, incl_unc=[], **kwargs):
    """Helper function for setting up the prim algorithm
    Parameters
    ----------
    results : tuple
              tuple of DataFrame and dict with numpy arrays
              the return from :meth:`perform_experiments`.
    classify : str or callable
               either a string denoting the outcome of interest to 
               use or a function. 
    threshold : double
                the minimum score on the density of the last box
                on the peeling trajectory. In case of a binary
                classification, this should be between 0 and 1. 
    incl_unc : list of str, optional
               list of uncertainties to include in prim analysis
    kwargs : dict
             valid keyword arguments for prim.Prim
    Returns
    -------
    a Prim instance
    Raises
    ------
    PrimException 
        if data resulting from classify is not a 1-d array. 
    TypeError 
        if classify is not a string or a callable.
    """

    x, y, mode = sdutil._setup(results, classify, incl_unc)

    return Prim(x, y, threshold=threshold, mode=mode, **kwargs)


def calculate_qp(data, x, y, Hbox, Tbox, box_lim, initial_boxlim):
    '''Helper function for calculating quasi p-values'''
    if data.size==0:
        return [-1, -1]
    
    u = data.name
    dtype = data.dtype
    
    unlimited = initial_boxlim[u]
    
    if dtype == object:
        temp_box = box_lim.copy()
        temp_box.loc[u] = unlimited
        qp = sdutil._calculate_quasip(x, y, temp_box,
                                      Hbox, Tbox)
        qp_values = [qp, -1]             
    else:
        qp_values = []
        for direction, (limit, unlimit) in enumerate(zip(data,
                                                         unlimited)):
            if unlimit != limit:
                temp_box = box_lim.copy()
                temp_box.loc[direction, u] = unlimit
                qp = sdutil._calculate_quasip(x, y, temp_box,
                                              Hbox, Tbox)
            else:
                qp = -1
            qp_values.append(qp)
    
    return qp_values


class CurEntry(object):
    '''a descriptor for the current entry on the peeling and pasting 
    trajectory'''

    def __init__(self, name):
        self.name = name

    def __get__(self, instance, _):
        return instance.peeling_trajectory[self.name][instance._cur_box]

    def __set__(self, instance, value):
        raise PrimException("this property cannot be assigned to")


class PrimBox(object):
    '''A class that holds information over a specific box 

    Attributes
    ----------
    coverage : float
               coverage of currently selected box
    density : float
               density of currently selected box
    mean : float
           mean of currently selected box
    res_dim : int
              number of restricted dimensions of currently selected box
    mass : float
           mass of currently selected box 
    peeling_trajectory : DataFrame
                         stats for each box in peeling trajectory
    box_lims : list
               list of box lims for each box in peeling trajectory

    by default, the currently selected box is the last box on the
    peeling trajectory, unless this is changed via
    :meth:`PrimBox.select`.

    '''

    coverage = CurEntry('coverage')
    density = CurEntry('density')
    mean = CurEntry('mean')
    res_dim = CurEntry('res_dim')
    mass = CurEntry('mass')

    _frozen = False

    def __init__(self, prim, box_lims, indices):
        '''init 

        Parameters
        ----------
        prim : Prim instance
        box_lims : DataFrame
        indices : ndarray


        '''

        self.prim = prim

        # peeling and pasting trajectory
        colums = ['coverage', 'density', 'mean', 'res_dim',
                  'mass', 'id']
        self.peeling_trajectory = pd.DataFrame(columns=colums)

        self.box_lims = []
        self.qp = []
        
        columns = ['name', 'lower', 'upper', 'minimum', 'maximum',
                   'qp_lower', 'qp_upper', 'id']
        self.boxes_quantitative = pd.DataFrame(columns=columns)
        
        columns = ['item', 'name', 'n_items', 'x', 'id']
        self.boxes_nominal = pd.DataFrame(columns=columns)
        
        self._cur_box = -1

        # indices van data in box
        self.update(box_lims, indices)

    def __getattr__(self, name):
        '''
        used here to give box_lim same behaviour as coverage, density,
        mean, res_dim, and mass. That is, it will return the box lim
        associated with the currently selected box. 
        '''

        if name == 'box_lim':
            return self.box_lims[self._cur_box]
        else:
            raise AttributeError

    def inspect(self, i=None, style='table', **kwargs):
        '''Write the stats and box limits of the user specified box to
        standard out. if i is not provided, the last box will be printed

        Parameters
        ----------
        i : int, optional
            the index of the box, defaults to currently selected box
        style : {'table', 'graph'}
                the style of the visualization

        additional kwargs are passed to the helper function that
        generates the table or graph

        '''
        if i == None:
            i = self._cur_box

        stats = self.peeling_trajectory.iloc[i].to_dict()
        stats['restricted_dim'] = stats['res_dim']
        
        qp_values = self.qp[i]
            
        uncs = [(key, value) for key, value in qp_values.items()]
        uncs.sort(key=itemgetter(1))
        uncs = [uncs[0] for uncs in uncs]

        if style == 'table':
            return self._inspect_table(i, uncs, qp_values, **kwargs)
        elif style == 'graph':
            return self._inspect_graph(i, uncs, qp_values, **kwargs)
        else:
            raise ValueError("style must be one of graph or table")

    def _inspect_table(self, i, uncs, qp_values):
        '''Helper function for visualizing box statistics in 
        table form'''
        # make the descriptive statistics for the box
        print(self.peeling_trajectory.iloc[i])
        print()

        # make the box definition
        columns = pd.MultiIndex.from_product([['box {}'.format(i)],
                                              ['min', 'max', 'qp values']])
        box_lim = pd.DataFrame(np.zeros((len(uncs), 3)),
                               index=uncs,
                               columns=columns)

        for unc in uncs:
            values = self.box_lims[i][unc][:]
            box_lim.loc[unc] = [values[0], values[1], str(qp_values[unc])]

        print(box_lim)
        print()

    def _inspect_graph(self,  i, uncs, qp_values,
                       ticklabel_formatter="{} ({})",
                       boxlim_formatter="{: .2g}",
                       table_formatter='{:.3g}'):
        '''Helper function for visualizing box statistics in 
        graph form'''

        return sdutil.plot_box(self.box_lims[i], qp_values,
                               self.prim.box_init, uncs, 
                               self.peeling_trajectory.loc[i, 'coverage'], 
                               self.peeling_trajectory.loc[i, "density"],
                               ticklabel_formatter=ticklabel_formatter,
                               boxlim_formatter=boxlim_formatter,
                               table_formatter=table_formatter)

    def inspect_tradeoff(self):
        boxes = []
        nominal_vars = []
        quantitative_dims = set(self.prim.x_float_colums.tolist() +\
                                self.prim.x_int_columns.tolist())
        nominal_dims = set(self.prim.x_nominal_columns)
        
        box_zero = self.box_lims[0]
        
        for i, (entry, qp) in enumerate(zip(self.box_lims, self.qp)):
            qp = pd.DataFrame(qp, index=['qp_lower', 'qp_upper'])
            dims = qp.columns.tolist()
            quantitative_res_dim = [e for e in dims if e in quantitative_dims]
            nominal_res_dims = [e for e in dims if e in nominal_dims]
            
            # handle quantitative
            df = entry
            
            box = df[quantitative_res_dim]
            box.index = ['x1', 'x2']
            box = box.T
            box['name'] = box.index
            box['id'] = int(i)
            box['minimum'] = box_zero[quantitative_res_dim].T.iloc[:, 0]
            box['maximum'] = box_zero[quantitative_res_dim].T.iloc[:, 1]
            box = box.join(qp.T)
            boxes.append(box)
            
            # handle nominal
            for dim in nominal_res_dims:
#                 TODO:: qp values
                items = df[nominal_res_dims].loc[0,:].values[0]
                for j, item in enumerate(items):
                    entry = dict(name=dim, n_items=len(items)+1,
                                 item=item, id=int(i),
                                 x=j/len(items))
                    nominal_vars.append(entry)
            
        boxes = pd.concat(boxes)
        nominal_vars = pd.DataFrame(nominal_vars)

        width = 400
        height = width
        
        point_selector = alt.selection_single(fields=['id'])

        peeling = self.peeling_trajectory.copy()
        peeling['id'] = peeling.index

        chart = alt.Chart(peeling).mark_circle(size=75).encode(
            x='coverage:Q',
            y='density:Q',
            color=alt.Color('res_dim:O',
                            scale=alt.Scale(range=sns.color_palette('YlGnBu', n_colors=8).as_hex())),
            opacity=alt.condition(point_selector, alt.value(1),
                                  alt.value(0.4))
        ).properties(
            selection=point_selector
        ).properties(
            width=width,
            height=height
        )
        
        
        # conda update -c conda-forge altair to 2.1
        # move this to encoding tooltip=[<list of items>]
        chart.encoding.tooltip = [
            {"type": "ordinal", 
             "field": "id"},
            {"type": "quantitative",
             "field": "coverage", "format" : ".2"},
            {"type": "quantitative",
             "field": "density", "format" : ".2"},
            {"type": "ordinal", "field": "res_dim", }
        ]
        
        base = alt.Chart(boxes).encode(
            x=alt.X('x_lower:Q', axis=alt.Axis(grid=False,
                                               title='box limits', labels=False),
                    scale=alt.Scale(domain=(0, 1), padding=0.1)),
            x2='x_upper:Q',
            y=alt.Y('name:N', scale=alt.Scale(padding=1.0))
        ).transform_calculate(
            x_lower='(datum.x1-datum.minimum)/(datum.maximum-datum.minimum)',
            x_upper='(datum.x2-datum.minimum)/(datum.maximum-datum.minimum)'
        ).transform_filter(
            point_selector
        ).properties(
            width=width,
        )
        
        lines = base.mark_rule()
        
        texts1 = base.mark_text(baseline='top', dy=5, align='left').encode(
            text=alt.Text('text:O') 
        ).transform_calculate(
            text=('datum.qp_lower>0?'
                  'format(datum.x1, ".2")+" ("+format(datum.qp_lower, ".1~g")+")" :'
                  'format(datum.x1, ".2")')
        )
        
        texts2 = base.mark_text(baseline='top', dy=5, align='right').encode(
            text=alt.Text('text:O'),
            x='x_upper:Q'
        ).transform_calculate(
            text=('datum.qp_upper>0?'
                  'format(datum.x2, ".2")+" ("+format(datum.qp_upper, ".1")+")" :'
                  'format(datum.x2, ".2")')
        )
        
        data = pd.DataFrame([dict(start=0, end=1)])
        rect = alt.Chart(data).mark_rect(opacity=0.05).encode(
            x='start:Q',
            x2='end:Q',
        )
        
        nominal = alt.Chart(nominal_vars).mark_point().encode(
            x='x:Q',
            y='name:N', 
        ).transform_filter(
            point_selector
        ).properties(
            width=width,
        )   
        
        texts3 = nominal.mark_text(baseline='top', 
                                   dy=5, align='center').encode(
            text='item:N'
        )
        
        layered = alt.layer(lines, texts1, texts2, rect, nominal,
                            texts3)
        
        return chart & layered
    
    def select(self, i):
        '''        
        select an entry from the peeling and pasting trajectory and
        update the prim box to this selected box.

        Parameters
        ----------
        i : int
            the index of the box to select.

        '''
        if self._frozen:
            raise PrimException(("box has been frozen because PRIM "
                                 "has found at least one more recent "
                                 "box"))

        res_dim = sdutil._determine_restricted_dims(self.box_lims[i],
                                                    self.prim.box_init)

        indices = sdutil._in_box(self.prim.x.loc[self.prim.yi_remaining,
                                                 res_dim],
                                 self.box_lims[i][res_dim])
        self.yi = self.prim.yi_remaining[indices]
        self._cur_box = i

    def drop_restriction(self, uncertainty):
        '''
        drop the restriction on the specified dimension. That is,
        replace the limits in the chosen box with a new box where for
        the specified uncertainty the limits of the initial box are
        being used. The resulting box is added to the peeling
        trajectory.

        Parameters
        ----------
        uncertainty : str

        '''

        new_box_lim = copy.deepcopy(self.box_lim)
        new_box_lim.loc[:, uncertainty]= self.box_lims[0].loc[:, uncertainty]
        indices = sdutil._in_box(self.prim.x.loc[self.prim.yi_remaining, :],
                                 new_box_lim)
        indices = self.prim.yi_remaining[indices]
        self.update(new_box_lim, indices)

    def update(self, box_lims, indices):
        '''

        update the box to the provided box limits.

        Parameters
        ----------
        box_lims: DataFrame
                  the new box_lims
        indices: ndarray
                 the indices of y that are inside the box

        '''
        self.yi = indices
        self.box_lims.append(box_lims)

        # peeling trajectory
        i = self.peeling_trajectory.shape[0]
        y = self.prim.y[self.yi]
        coi = self.prim.determine_coi(self.yi)
        
        restricted_dims = sdutil._determine_restricted_dims(self.box_lims[-1],
                                                            self.prim.box_init)

        data = {'coverage': coi/self.prim.t_coi,
                'density': coi/y.shape[0],
                'mean': np.mean(y),
                'res_dim': restricted_dims.shape[0],
                'mass': y.shape[0]/self.prim.n,
                'id': i}
        new_row = pd.DataFrame([data])
        self.peeling_trajectory = self.peeling_trajectory.append(new_row,
                                            ignore_index=True,sort=True)

        # boxlims
        qp = self._calculate_quasi_p(i, restricted_dims)
        self.qp.append(qp)
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
        par.plot(self.peeling_trajectory['res_dim'], label="restricted dims")
        ax.grid(True, which='both')
        ax.set_ylim(bottom=0, top=1)

        fig = plt.gcf()

        make_legend(['mean', 'mass', 'coverage', 'density',
                     'restricted_dim'],
                    ax, ncol=5, alpha=1)
        return fig

    def show_tradeoff(self, cmap=mpl.cm.viridis):  # @UndefinedVariable
        '''Visualize the trade off between coverage and density. Color
        is used to denote the number of restricted dimensions. 

        Parameters
        ----------
        cmap : valid matplotlib colormap

        Returns
        -------
        a Figure instance

        '''

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        boundaries = np.arange(-0.5,
                               max(self.peeling_trajectory['res_dim'])+1.5,
                               step=1)
        ncolors = cmap.N
        norm = mpl.colors.BoundaryNorm(boundaries, ncolors)

        p = ax.scatter(self.peeling_trajectory['coverage'],
                       self.peeling_trajectory['density'],
                       c=self.peeling_trajectory['res_dim'],
                       norm=norm,
                       cmap=cmap)
        ax.set_ylabel('density')
        ax.set_xlabel('coverage')
        ax.set_ylim(bottom=0, top=1.2)
        ax.set_xlim(left=0, right=1.2)

        ticklocs = np.arange(0,
                             max(self.peeling_trajectory['res_dim'])+1,
                             step=1)
        cb = fig.colorbar(p, spacing='uniform', ticks=ticklocs,
                          drawedges=True)
        cb.set_label("nr. of restricted dimensions")

        return fig

    def show_pairs_scatter(self, i=None):
        ''' Make a pair wise scatter plot of all the restricted
        dimensions with color denoting whether a given point is of
        interest or not and the boxlims superimposed on top.
        
        Parameters
        ----------
        i : int, optional
        
        Returns
        -------
        seaborn PairGrid

        '''
        if i == None:
            i = self._cur_box
        resdim = sdutil._determine_restricted_dims(self.box_lims[i],
                                                    self.prim.box_init)
        
        return _pair_wise_scatter(self.prim.x, self.prim.y,
                                  self.box_lims[i], self.prim.box_init,
                                  resdim)
                  
    def write_ppt_to_stdout(self):
        '''write the peeling and pasting trajectory to stdout'''
        print(self.peeling_trajectory)
        print("\n")

    def _calculate_quasi_p(self, i, restricted_dims):
        '''helper function for calculating quasi-p values as discussed
        in Bryant and Lempert (2010). This is a one sided  binomial
        test. 

        Parameters
        ----------
        i : int
            the specific box in the peeling trajectory for which the
            quasi-p values are to be calculated.

        Returns
        -------
        dict

        '''

        box_lim = self.box_lims[i]
        box_lim = box_lim[restricted_dims]

        # total nr. of cases in box
        Tbox = self.peeling_trajectory['mass'][i] * self.prim.n

        # total nr. of cases of interest in box
        Hbox = self.peeling_trajectory['coverage'][i] * self.prim.t_coi

        x = self.prim.x.loc[self.prim.yi_remaining, restricted_dims]
        y = self.prim.y[self.prim.yi_remaining]

        ## TODO use apply on df?
        
        qp_values = box_lim.apply(calculate_qp, axis=0, result_type='expand', 
                                  args=[x, y, Hbox, Tbox, box_lim,
                                        self.box_lims[0]])
        qp_values = qp_values.to_dict(orient='list')
        return qp_values

#     def _format_stats(self, nr, stats):
#         '''helper function for formating box stats'''
#         row = self.stats_format.format(nr, **stats)
#         return row


class PrimException(Exception):
    '''Base exception class for prim related exceptions'''
    pass


class Prim(sdutil.OutputFormatterMixin):
    '''Patient rule induction algorithm

    The implementation of Prim is tailored to interactive use in the
    context of scenario discovery

    Parameters
    ----------
    x : DataFrame
        the independent variables
    y : 1d ndarray
        the dependent variable
    threshold : float
                the density threshold that a box has to meet
    obj_function : {LENIENT1, LENIENT2, ORIGINAL}
                   the objective function used by PRIM. Defaults to a
                   lenient objective function based on the gain of mean
                   divided by the loss of mass. 
    peel_alpha : float, optional 
                 parameter controlling the peeling stage (default = 0.05). 
    paste_alpha : float, optional
                  parameter controlling the pasting stage (default = 0.05).
    mass_min : float, optional
               minimum mass of a box (default = 0.05). 
    threshold_type : {ABOVE, BELOW}
                     whether to look above or below the threshold value
    mode : {'binary', 'classification'}, optional
            indicated whether PRIM is used for regression, or for scenario
            classification in which case y should be a binary vector
    update_function = {'default', 'guivarch'}, optional
                      controls behavior of PRIM after having found a
                      first box. use either the default behavior were
                      all points are removed, or the procedure
                      suggested by guivarch et al (2016)
                      doi:10.1016/j.envsoft.2016.03.006 to simply set 
                      all points to be no longer of interest (only
                      valid in binary mode). 

    See also
    --------
    :mod:`cart`


    '''

    message = "{0} points remaining, containing {1} cases of interest"

    def __init__(self, x, y, threshold, obj_function=LENIENT1,
                 peel_alpha=0.05, paste_alpha=0.05, mass_min=0.05,
                 threshold_type=ABOVE, mode=sdutil.BINARY,
                 update_function='default'):
        assert mode in {sdutil.BINARY, sdutil.REGRESSION}
        assert self._assert_mode(y, mode, update_function)

        # preprocess x
        try:
            x.drop(columns='scenario', inplace=True)
        except KeyError:
            pass
        
        x_float = x.select_dtypes(np.float)
        self.x_float = x_float.values
        self.x_float_colums = x_float.columns.values
        
        x_int = x.select_dtypes(np.int)
        self.x_int = x_int.values
        self.x_int_columns = x_int.columns.values
        
        self.x_numeric_columns = np.concatenate([self.x_float_colums,
                                                 self.x_int_columns])
        
        x_nominal = x.select_dtypes(exclude=np.number)
        self.x_nominal = x_nominal.values
        self.x_nominal_columns = x_nominal.columns.values 

        # TODO::filter out dimensions with only single value

        self.n_cols = x.columns.shape[0]
        
        for column in self.x_nominal_columns:
            x[column] = x[column].astype('category')

        self.x = x
        self.y = y
        self.mode = mode

        self._update_yi_remaining = self._update_functions[update_function]

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
        self.yi = x.index.values

        # how many data points do we have
        self.n = self.y.shape[0]

        # how many cases of interest do we have?
        self.t_coi = self.determine_coi(self.yi)

        # initial box that contains all data
        self.box_init = sdutil._make_box(self.x)

        # make a list in which the identified boxes can be put
        self._boxes = []

        self._update_yi_remaining(self)

    @property
    def boxes(self):
        boxes = [box.box_lim for box in self._boxes]

        if not boxes:
            return [self.box_init]
        return boxes

    @property
    def stats(self):
        stats = []
        items = ['coverage', 'density', 'mass', 'res_dim']
        for box in self._boxes:
            stats.append({key: getattr(box, key) for key in items})
        return stats

    def perform_pca(self, subsets=None, exclude=set()):
        '''

        WARNING:: code still needs to be tested!!!

        Pre-process the data by performing a pca based rotation on it. 
        This effectively turns the algorithm into PCA-PRIM as described
        in `Dalal et al (2013) <http://www.sciencedirect.com/science/article/pii/S1364815213001345>`_

        Parameters
        ----------
        subsets: dict, optional 
                 expects a dictionary with group name as key and a list of 
                 uncertainty names as values. If this is used, a constrained 
                 PCA-PRIM is executed 

                ..note:: the list of uncertainties should not contain 
                         categorical uncertainties. 
        exclude : list of str, optional 
                  the uncertainties that should be excluded from the rotation

        '''

        # transform experiments to numpy array
        dtypes = self.x.dtype.fields
        object_dtypes = [key for key, value in dtypes.items()
                         if value[0] == np.dtype(object)]

        # get experiments of interest
        # TODO this assumes binary classification!!!!!!!
        logical = self.y >= self.threshold

        # if no subsets are provided all uncertainties with non dtype object
        # are in the same subset, the name of this is r, for rotation
        if not subsets:
            subsets = {"r": [key for key, value in dtypes.items()
                             if value[0].name != np.dtype(object)]}
        else:
            # remove uncertainties that are in exclude and check whether
            # uncertainties occur in more then one subset
            seen = set()
            for key, value in subsets.items():
                value = set(value) - set(exclude)

                subsets[key] = list(value)
                if (seen & value):
                    raise EMAError(
                        "uncertainty occurs in more then one subset")
                else:
                    seen = seen | set(value)

        # prepare the dtypes for the new rotated experiments recarray
        new_dtypes = []
        for key, value in subsets.items():
            self._assert_dtypes(value, dtypes)

            # the names of the rotated columns are based on the group name
            # and an index
            [new_dtypes.append((str("{}_{}".format(key, i)), float)) for i
             in range(len(value))]

        # add the uncertainties with object dtypes to the end
        included_object_dtypes = set(object_dtypes)-set(exclude)
        [new_dtypes.append((name, object)) for name in included_object_dtypes]

        # make a new empty recarray
        rotated_experiments = np.empty((self.x.shape[0],), dtype=new_dtypes)

        # put the uncertainties with object dtypes already into the new recarray
        for name in included_object_dtypes:
            rotated_experiments[name] = self.x[name]

        # iterate over the subsets, rotate them, and put them into the new
        # recarray
        shape = 0
        for key, value in subsets.items():
            shape += len(value)
        rotation_matrix = np.zeros((shape, shape))
        column_names = []
        row_names = []

        j = 0
        for key, value in subsets.items():
            data = self._rotate_subset(value, self.x, logical)
            subset_rotation_matrix, subset_experiments = data
            rotation_matrix[j:j+len(value), j:j+len(value)
                            ] = subset_rotation_matrix
            [row_names.append(entry) for entry in value]
            j += len(value)

            for i in range(len(value)):
                name = "%s_%s" % (key, i)
                rotated_experiments[name] = subset_experiments[:, i]
                [column_names.append(name)]

        self.rotation_matrix = rotation_matrix
        self.column_names = column_names
        self.row_names = row_names

        self.x = np.ma.array(rotated_experiments)
        self.box_init = sdutil._make_box(self.x)

    def find_box(self):
        '''Execute one iteration of the PRIM algorithm. That is, find one
        box, starting from the current state of Prim.'''
        # set the indices
        self._update_yi_remaining(self)

        # make boxes already found immutable
        for box in self._boxes:
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

        message = ("mean: {0}, mass: {1}, coverage: {2}, "
                   "density: {3} restricted_dimensions: {4}")
        message = message.format(box.mean,
                                 box.mass,
                                 box.coverage,
                                 box.density,
                                 box.res_dim)

        if (self.threshold_type == ABOVE) &\
           (box.mean >= self.threshold):
            info(message)
            self._boxes.append(box)
            return box
        elif (self.threshold_type == BELOW) &\
                (box.mean <= self.threshold):
            info(message)
            self._boxes.append(box)
            return box
        else:
            # make a dump box
            info('box does not meet threshold criteria, value is {}, returning dump box'.format(
                box.mean))
            box = PrimBox(self, self.box_init, self.yi_remaining[:])
            self._boxes.append(box)
            return box

    def determine_coi(self, indices):
        '''        
        Given a set of indices on y, how many cases of interest are
        there in this set.

        Parameters
        ----------
        indices: ndarray
                 a valid index for y

        Returns
        ------- 
        int
            the number of cases of interest.

        Raises
        ------
        ValueError 
            if threshold_type is not either ABOVE or BELOW

        '''

        y = self.y[indices]

        if self.threshold_type == ABOVE:
            coi = y[y >= self.threshold].shape[0]
        elif self.threshold_type == BELOW:
            coi = y[y <= self.threshold].shape[0]
        else:
            raise ValueError("threshold type is not one of ABOVE or BELOW")

        return coi

    def _update_yi_remaining_default(self):
        '''

        Update yi_remaining in light of the state of the boxes
        associated with this prim instance.

        '''

        # set the indices
        logical = np.ones(self.yi.shape[0], dtype=np.bool)
        for box in self._boxes:
            logical[box.yi] = False
        self.yi_remaining = self.yi[logical]

    def _update_yi_remaining_guivarch(self):
        '''

        Update yi_remaining in light of the state of the boxes
        associated with this prim instance using the modified version 
        from  Guivarch et al (2016) doi:10.1016/j.envsoft.2016.03.006

        '''
        # set the indices
        for box in self._boxes:
            self.y[box.yi] = 0

        self.yi_remaining = self.yi

    def _peel(self, box):
        '''

        Executes the peeling phase of the PRIM algorithm. Delegates
        peeling to data type specific helper methods.

        '''

        mass_old = box.yi.shape[0]/self.n

        x_float = self.x_float[box.yi]
        x_int = self.x_int[box.yi]
        x_nominal = self.x_nominal[box.yi]

        # identify all possible peels
        possible_peels = []
        
        for x, columns, dtype,  in [(x_float, self.x_float_colums,
                                     'float'),
                                    (x_int, self.x_int_columns,
                                     'int'),
                                    (x_nominal, self.x_nominal_columns,
                                     'object')]:
            for j, u in enumerate(columns):
                peels = self._peels[dtype](self, box, u, j, x)
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
            non_res_dim = self.n_cols -\
                sdutil._determine_nr_restricted_dims(box_lim,
                                                     self.box_init)
            score = (obj, non_res_dim, box_lim, i)
            scores.append(score)

        scores.sort(key=itemgetter(0, 1), reverse=True)
        entry = scores[0]

        obj_score = entry[0]
        box_new, indices = entry[2:]

        mass_new = self.y[indices].shape[0]/self.n

        if (mass_new >= self.mass_min) &\
           (mass_new < mass_old) &\
           (obj_score > 0):
            box.update(box_new, indices)
            return self._peel(box)
        else:
            # else return received box
            return box

    def _real_peel(self, box, u, j, x):
        '''

        returns two candidate new boxes, peel along upper and lower
        dimension

        Parameters
        ----------
        box : a PrimBox instance
        u : str
            the uncertainty for which to peel
        j : int
            column for which to peel
        x : ndarray

        Returns
        -------
        tuple
            two box lims and the associated indices

        '''

        peels = []
        for direction in ['upper', 'lower']:
            xj = x[:, j]

            peel_alpha = self.peel_alpha

            i = 0
            if direction == 'upper':
                peel_alpha = 1-self.peel_alpha
                i = 1

            box_peel = get_quantile(xj, peel_alpha)
            if direction == 'lower':
                logical = xj >= box_peel
                indices = box.yi[logical]
            if direction == 'upper':
                logical = xj <= box_peel
                indices = box.yi[logical]
            temp_box = copy.deepcopy(box.box_lims[-1])
            temp_box.loc[i, u] = box_peel
            peels.append((indices, temp_box))


        return peels

    def _discrete_peel(self, box, u, j,  x):
        '''

        returns two candidate new boxes, peel along upper and lower
        dimension

        Parameters
        ----------
        box : a PrimBox instance
        u : str
            the uncertainty for which to peel
        j : int
            column for which to peel
        x : ndarray

        Returns
        -------
        tuple
            two box lims and the associated indices

        '''
        peels = []
        for direction in ['upper', 'lower']:
            peel_alpha = self.peel_alpha
            xj = x[:, j]
            box_lim = box.box_lims[-1]

            i = 0
            if direction == 'upper':
                peel_alpha = 1-self.peel_alpha
                i = 1

            box_peel = get_quantile(xj, peel_alpha)
            box_peel = int(box_peel)

            # determine logical associated with peel value
            if direction == 'lower':
                if box_peel == box_lim.loc[i, u]:
                    logical = (xj > box_lim.loc[i, u]) &\
                              (xj <= box_lim.loc[i+1, u])
                else:
                    logical = (xj >= box_peel) &\
                              (xj <= box_lim.loc[i+1, u])
            if direction == 'upper':
                if box_peel == box_lim.loc[i, u]:
                    logical = (xj < box_lim.loc[i, u]) &\
                              (xj >= box_lim[i-1, u])
                else:
                    logical = (xj <= box_peel) &\
                              (xj >= box_lim.loc[i-1, u])

            # determine value of new limit given logical
            if xj[logical].shape[0] == 0:
                if direction == 'upper':
                    new_limit = np.max(xj)
                else:
                    new_limit = np.min(xj)
            else:
                if direction == 'upper':
                    new_limit = np.max(xj[logical])
                else:
                    new_limit = np.min(xj[logical])

            indices = box.yi[logical]
            temp_box = copy.deepcopy(box_lim)
            temp_box.loc[i, u] = new_limit
            peels.append((indices, temp_box))

        return peels

    def _categorical_peel(self, box, u, j, x):
        '''

        returns candidate new boxes for each possible removal of a
        single  category. So. if the box[u] is a categorical variable
        with 4 categories, this method will return 4 boxes. 

        Parameters
        ----------
        box : a PrimBox instance
        u : str
            the uncertainty for which to peel
        j : int
            column for which to peel
        x : ndarray

        Returns
        -------
        tuple
            a list of box lims and the associated indices

        '''
        entries = box.box_lims[-1].loc[0, u]

        if len(entries) > 1:
            peels = []
            for entry in entries:
                bools = []
                
                temp_box = box.box_lims[-1].copy()
                peel = copy.deepcopy(entries)
                peel.discard(entry)
                temp_box[u] = [peel, peel]

                if type(list(entries)[0]) not in (str, float,
                                                  int, bool):
                    for element in x[u]:
                        if element != entry:
                            bools.append(True)
                        else:
                            bools.append(False)
                    logical = np.asarray(bools, dtype=bool)
                else:
                    logical = x[:, j] != entry
                indices = box.yi[logical]
                peels.append((indices,  temp_box))
            return peels
        else:
            # no peels possible, return empty list
            return []

    def _paste(self, box):
        ''' Executes the pasting phase of the PRIM. Delegates pasting
        to data type specific helper methods.'''

        x = self.x.loc[self.yi_remaining, :]

        mass_old = box.yi.shape[0]/self.n

        # need to break this down by dtype
        restricted_dims = sdutil._determine_restricted_dims(box.box_lims[-1],
                                                    self.box_init)
        res_dim = set(restricted_dims)

        possible_pastes = []
        for columns, dtype,  in [(self.x_float_colums, 'float'),
                                    (self.x_int_columns, 'int'),
                                    (self.x_nominal_columns, 'object')]:
            for u in enumerate(columns):
                if u not in res_dim: continue
                debug("pasting "+u)
#                 welke X hier te gebruiken? eigenlijk kan niet 
#                 op basis van yi_remaining toch zijn?
#                 voor peel zou je toch gebruik moeten maken van de 
#                 status van de box? (dus yi van primBox)
#                 
#                 voor paste is het nog ingewikkelder, je moet namelijk
#                 een nieuwe logical maken voor de candidate paste dimensie

                pastes = self._pastes[dtype](self, box, u, x, restricted_dims)
                [possible_pastes.append(entry) for entry in pastes]
            if not possible_pastes:
                # there is no peel identified, so return box
                return box

#         possible_pastes = []
#         for u in res_dim:
#             debug("pasting "+u)
#             dtype = self.x.dtype.fields.get(u)[0].name
#             pastes = self._pastes[dtype](self, box, u)
#             [possible_pastes.append(entry) for entry in pastes]


        # determine the scores for each peel in order
        # to identify the next candidate box
        scores = []
        for entry in possible_pastes:
            i, box_lim = entry
            obj = self.obj_func(self, self.y[box.yi],  self.y[i])
            non_res_dim = len(x.dtype.descr) -\
                sdutil._determine_nr_restricted_dims(box_lim,
                                                     self.box_init)
            score = (obj, non_res_dim, box_lim, i)
            scores.append(score)

        scores.sort(key=itemgetter(0, 1), reverse=True)
        entry = scores[0]
        obj, _, box_new, indices = entry
        mass_new = self.y[indices].shape[0]/self.n

        mean_old = np.mean(self.y[box.yi])
        mean_new = np.mean(self.y[indices])

        if (mass_new >= self.mass_min) &\
           (mass_new > mass_old) &\
           (obj > 0) &\
           (mean_new > mean_old):
            box.update(box_new, indices)
            return self._paste(box)
        else:
            # else return received box
            return box

    def _real_paste(self, box, u, x, restricted_dims):
        ''' returns two candidate new boxes, pasted along upper and
        lower dimension

        Parameters
        ----------
        box : a PrimBox instance
        u : str
            the uncertainty for which to peel
        x : ndarray

        
        Returns
        -------
        tuple
            two box lims and the associated indices

        '''

        pastes = []
        for i, direction in enumerate(['lower', 'upper']):
            box_paste = box.box_lims[-1][restricted_dims].copy()
            # box containing data candidate for pasting
            paste_box = box_paste.copy()

            minimum, maximum = self.box_init[u].values

            if direction == 'lower':
                paste_box.loc[:, u] = minimum, box_paste.loc[0, u]

                indices = sdutil._in_box(x, paste_box)
                data = x.loc[indices, u]

                paste_value = minimum
                if data.size > 0:
                    paste_value = get_quantile(data, 1-self.paste_alpha)

                if not paste_value <= box.box_lims[-1][u][i]:
                    print("{}, {}".format(paste_value, box.box_lims[-1].loc[i, u]))
            else:  #direction == 'upper':
                paste_box.loc[0, u] = paste_box.loc[1, u], maximum
                
                indices = sdutil._in_box(x, paste_box)
                data = x.loc[indices, u]

                paste_value = maximum
                if data.size > 0:
                    paste_value = get_quantile(data, self.paste_alpha)

                assert paste_value >= box.box_lims[-1].loc[i, u]

            dtype = box_paste.dtype.fields[u][0]
            if dtype == np.int32:
                paste_value = np.int(paste_value)

            box_paste.loc[i, u] = paste_value
            logical = sdutil._in_box(x, box_paste)
            indices = self.yi_remaining[logical]

            pastes.append((indices, box_paste))

        return pastes

    def _categorical_paste(self, box, u, x, restricted_dims):
        '''

        Return a list of pastes, equal to the number of classes currently
        not on the box lim. 

        Parameters
        ----------
        box : a PrimBox instance
        u : str
            the uncertainty for which to peel
        x : ndarray


        Returns
        -------
        tuple
            a list of box lims and the associated indices


        '''
        box_lim = box.box_lims[-1][restricted_dims]

        c_in_b = box_lim.loc[0, u]
        c_t = self.box_init.loc[0, u]

        if len(c_in_b) < len(c_t):
            pastes = []
            possible_cs = c_t - c_in_b
            for entry in possible_cs:
                paste = copy.deepcopy(c_in_b)
                paste.add(entry)

                box_paste = box_lim.copy()
                box_paste.loc[:, u] = [paste, paste]

                indices = sdutil._in_box(x, 
                                         box_paste)
                indices = self.yi_remaining[indices]
                pastes.append((indices, box_paste))
            return pastes
        else:
            # no pastes possible, return empty list
            return []

    def _lenient1_obj_func(self, y_old, y_new):
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

        if y_new.shape[0] > 0:
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
                raise PrimException('''mean is different {} vs {}, while shape is the same,
                                       this cannot be the case'''.format(mean_old, mean_new))
        return obj

    def _lenient2_obj_func(self, y_old, y_new):
        '''

        friedman and fisher 14.6


        '''
        mean_old = np.mean(y_old)

        if y_new.shape[0] > 0:
            mean_new = np.mean(y_new)
        else:
            mean_new = 0

        obj = 0
        if mean_old != mean_new:
            if y_old.shape == y_new.shape:
                raise PrimException('''mean is different {} vs {}, while shape is the same,
                                       this cannot be the case'''.format(mean_old, mean_new))

            change_mean = mean_new - mean_old
            change_mass = abs(y_old.shape[0]-y_new.shape[0])
            mass_new = y_new.shape[0]

            obj = mass_new * change_mean / change_mass

        return obj

    def _original_obj_func(self, y_old, y_new):
        ''' The original objective function: the mean of the data inside the 
        box'''

        if y_new.shape[0] > 0:
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
                raise EMAError(
                    "%s has dtype object and can thus not be rotated" % key)
        return True

    def _assert_mode(self, y, mode, update_function):
        if mode == sdutil.BINARY:
            return set(np.unique(y)) == {0, 1}
        if update_function == 'guivarch':
            return False
        return True

    def _rotate_subset(self, value, orig_experiments, logical):
        '''
        rotate a subset

        Parameters
        ----------
        value : list of str
        orig_experiment : numpy structured array
        logical : boolean array

        '''
        list_dtypes = [(name, "<f8") for name in value]

        # cast everything to float
        drop_names = set(rf.get_names(orig_experiments.dtype)) - set(value)
        orig_subset = rf.drop_fields(orig_experiments, drop_names,
                                     asrecarray=True)
        subset_experiments = orig_subset.astype(list_dtypes).view(
            '<f8').reshape(orig_experiments.shape[0], len(value))

        # normalize the data
        mean = np.mean(subset_experiments, axis=0)
        std = np.std(subset_experiments, axis=0)
        std[std == 0] = 1  # in order to avoid a devision by zero
        subset_experiments = (subset_experiments - mean)/std

        # get the experiments of interest
        experiments_of_interest = subset_experiments[logical]

        # determine the rotation
        rotation_matrix = self._determine_rotation(experiments_of_interest)

        # apply the rotation
        subset_experiments = np.dot(subset_experiments, rotation_matrix)
        return rotation_matrix, subset_experiments

    def _determine_rotation(self, experiments):
        '''
        Determine the rotation for the specified experiments

        '''
        covariance = np.cov(experiments.T)

        eigen_vals, eigen_vectors = np.linalg.eig(covariance)

        indices = np.argsort(eigen_vals)
        indices = indices[::-1]
        eigen_vectors = eigen_vectors[:, indices]
        eigen_vals = eigen_vals[indices]

        # make the eigen vectors unit length
        for i in range(eigen_vectors.shape[1]):
            eigen_vectors[:, i] / \
                np.linalg.norm(eigen_vectors[:, i]) * np.sqrt(eigen_vals[i])

        return eigen_vectors

    _peels = {'object': _categorical_peel,
              'int': _discrete_peel,
              'float': _real_peel}

    _pastes = {'object': _categorical_paste,
               'int': _real_paste,
               'float': _real_paste}

    # dict with the various objective functions available
    _obj_functions = {LENIENT2: _lenient2_obj_func,
                      LENIENT1: _lenient1_obj_func,
                      ORIGINAL: _original_obj_func}

    _update_functions = {'default': _update_yi_remaining_default,
                         'guivarch': _update_yi_remaining_guivarch}
