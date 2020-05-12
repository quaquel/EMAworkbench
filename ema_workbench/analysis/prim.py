'''

A scenario discovery oriented implementation of PRIM.

The implementation of prim provided here is data type aware, so
categorical variables will be handled appropriately. It also uses a
non-standard objective function in the peeling and pasting phase of the
algorithm. This algorithm looks at the increase in the mean divided
by the amount of data removed. So essentially, it uses something akin
to the first order derivative of the original objective function.

The implementation is designed for interactive use in combination with
the jupyter notebook.

'''
import copy
import itertools
import matplotlib as mpl
from operator import itemgetter
import warnings

import numpy as np
import pandas as pd
import seaborn as sns

try:
    import altair as alt
except ImportError:
    alt = None
    warnings.warn(("altair based interactive "
                   "inspection not available"), ImportWarning)

from ..util import (EMAError, temporary_filter, INFO, get_module_logger)
from . import scenario_discovery_util as sdutil
from .prim_util import (PrimException, CurEntry, PRIMObjectiveFunctions,
                        NotSeen, is_pareto_efficient,
                        get_quantile, rotate_subset, calculate_qp,
                        determine_dimres, is_significant)

# Created on 22 feb. 2013
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


__all__ = ['ABOVE', 'BELOW', 'setup_prim', 'Prim', 'PrimBox',
           "pca_preprocess", "run_constrained_prim", 'PRIMObjectiveFunctions']
_logger = get_module_logger(__name__)


ABOVE = 1
BELOW = -1
PRECISION = '.2f'


def pca_preprocess(experiments, y, subsets=None, exclude=set()):
    '''perform PCA to preprocess experiments before running PRIM

    Pre-process the data by performing a pca based rotation on it.
    This effectively turns the algorithm into PCA-PRIM as described
    in `Dalal et al (2013) <http://www.sciencedirect.com/science/article/pii/S1364815213001345>`_

    Parameters
    ----------
    experiments : DataFrame
    y : ndarray
        one dimensional binary array
    subsets : dict, optional
              expects a dictionary with group name as key and a list of
              uncertainty names as values. If this is used, a constrained
              PCA-PRIM is executed
    exclude : list of str, optional
              the uncertainties that should be excluded from the rotation

    Returns
    -------
    rotated_experiments
        DataFrame
    rotation_matrix
        DataFrame

    Raises
    ------
    RuntimeError
        if mode is not binary (i.e. y is not a binary classification).
        if X contains non numeric columns


    '''
    # experiments to rotate
    x = experiments.drop(exclude, axis=1)

    #
    if not x.select_dtypes(exclude=np.number).empty:
        raise RuntimeError("X includes non numeric columns")
    if not set(np.unique(y)) == {0, 1}:
        raise RuntimeError('y should only contain 0s and 1s')

    # if no subsets are provided all uncertainties with non dtype object
    # are in the same subset, the name of this is r, for rotation
    if not subsets:
        subsets = {"r": x.columns.values.tolist()}
    else:
        # TODO:: should we check on double counting in subsets?
        #        should we check all uncertainties are in x?
        pass

    # prepare the dtypes for the new rotated experiments dataframe
    new_columns = []
    new_dtypes = []
    for key, value in subsets.items():
        # the names of the rotated columns are based on the group name
        # and an index
        subset_cols = ["{}_{}".format(key, i) for i in range(len(value))]
        new_columns.extend(subset_cols)
        new_dtypes.extend((float,) * len(value))

    # make a new empty experiments dataframe
    rotated_experiments = pd.DataFrame(index=experiments.index.values)

    for name, dtype in zip(new_columns, new_dtypes):
        rotated_experiments[name] = pd.Series(dtype=dtype)

    # put the uncertainties with object dtypes already into the new
    for entry in exclude:
        rotated_experiments[name] = experiments[entry]

    # iterate over the subsets, rotate them, and put them into the new
    # experiments dataframe
    rotation_matrix = np.zeros((x.shape[1], ) * 2)
    column_names = []
    row_names = []

    j = 0
    for key, value in subsets.items():
        x_subset = x[value]
        subset_rotmat, subset_experiments = rotate_subset(x_subset, y)
        rotation_matrix[j:j + len(value), j:j + len(value)] = subset_rotmat
        row_names.extend(value)
        j += len(value)

        for i in range(len(value)):
            name = "%s_%s" % (key, i)
            rotated_experiments[name] = subset_experiments[:, i]
            [column_names.append(name)]

    rotation_matrix = pd.DataFrame(rotation_matrix, index=row_names,
                                   columns=column_names)

    return rotated_experiments, rotation_matrix


def run_constrained_prim(experiments, y, issignificant=True,
                         **kwargs):
    ''' Run PRIM repeatedly while constraining the maximum number of dimensions
    available in x

    Improved usage of PRIM as described in `Kwakkel (2019) <https://onlinelibrary.wiley.com/doi/full/10.1002/ffo2.8>`_.

    Parameters
    ----------
    x : numpy structured array
    y : numpy array
    issignificant : bool, optional
                    if True, run prim only on subsets of dimensions
                    that are significant for the initial PRIM on the
                    entire dataset.
    **kwargs : any additional keyword arguments are passed on to PRIM


    Returns
    -------
    PrimBox instance

    '''
    frontier = []
    merged_lims = []
    merged_qp = []

    alg = Prim(experiments, y, threshold=0.1, **kwargs)
    boxn = alg.find_box()

    dims = determine_dimres(boxn, issignificant=issignificant)

    # run prim for all possible combinations of dims
    subsets = []
    for n in range(1, len(dims) + 1):
        for subset in itertools.combinations(dims, n):
            subsets.append(subset)
    _logger.info("going to run PRIM {} times".format(len(subsets)))

    boxes = [boxn]
    for subset in subsets:
        with temporary_filter(__name__, INFO):
            x = experiments.loc[:, subset].copy()
            alg = Prim(x, y, threshold=0.1, **kwargs)
            box = alg.find_box()
            boxes.append(box)

    box_init = boxn.prim.box_init
    not_seen = NotSeen()

    for box in boxes:
        peeling = box.peeling_trajectory
        lims = box.box_lims

        logical = np.ones(box.peeling_trajectory.shape[0], dtype=np.bool)

        for i in range(box.peeling_trajectory.shape[0]):
            lim = lims[i]

            boxlim = box_init.copy()
            for column in lim:
                boxlim[column] = lim[column]

            boolean = is_significant(box, i) & not_seen(boxlim)

            logical[i] = boolean
            if boolean:
                merged_lims.append(boxlim)
                merged_qp.append(box.qp[i])
        frontier.append(peeling[logical])

    frontier = pd.concat(frontier)

    # remove dominated boxes
    peeling_trajectory = frontier.reset_index(drop=True)
    data = peeling_trajectory.iloc[:, [0, 1, -1]].copy().values
    data[:, 2] *= -1
    logical = is_pareto_efficient(data)

    # resort to ensure sensible ordering
    pt = peeling_trajectory[logical]
    pt = pt.reset_index(drop=True)
    pt = pt.sort_values(['res_dim', 'coverage'],
                        ascending=[True, False])

    box_lims = [lim for lim, entry in zip(merged_lims, logical) if entry]
    qps = [qp for qp, entry in zip(merged_qp, logical) if entry]
    sorted_lims = []
    sorted_qps = []
    for entry in pt.index:
        sorted_lims.append(box_lims[int(entry)])
        sorted_qps.append(qps[int(entry)])

    # ensuring index has normal order starting from 0
    pt = pt.reset_index(drop=True)
    pt.id = pt.index

    # create PrimBox
    box = PrimBox(boxn.prim, boxn.prim.box_init, boxn.prim.yi)
    box.peeling_trajectory = pt
    box.box_lims = sorted_lims
    box.qp = sorted_qps
    return box


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


class PrimBox(object):
    '''A class that holds information for a specific box

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
        self._resampled = []
        self.yi_initial = indices[:]

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
        standard out. if i is not provided, the last box will be
        printed

        Parameters
        ----------
        i : int, optional
            the index of the box, defaults to currently selected box
        style : {'table', 'graph'}
                the style of the visualization

        additional kwargs are passed to the helper function that
        generates the table or graph

        '''
        if i is None:
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
            values = self.box_lims[i][unc]
            box_lim.loc[unc] = [values[0], values[1],
                                str(qp_values[unc])]

        print(box_lim)
        print()

    def _inspect_graph(self, i, uncs, qp_values,
                       ticklabel_formatter="{} ({})",
                       boxlim_formatter="{: .2g}",
                       table_formatter='{:.3g}'):
        '''Helper function for visualizing box statistics in
        graph form'''

        return sdutil.plot_box(self.box_lims[i], qp_values,
                               self.prim.box_init, uncs,
                               self.peeling_trajectory.at[i, 'coverage'],
                               self.peeling_trajectory.at[i, "density"],
                               ticklabel_formatter=ticklabel_formatter,
                               boxlim_formatter=boxlim_formatter,
                               table_formatter=table_formatter)

    def inspect_tradeoff(self):
        # TODO::
        # make legend with res_dim color code a selector as well?
        # https://medium.com/dataexplorations/focus-generating-an-interactive-legend-in-altair-9a92b5714c55
        
        boxes = []
        nominal_vars = []
        quantitative_dims = set(self.prim.x_float_colums.tolist() +
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
                # TODO:: qp values
                items = df[dim].values[0]
                for j, item in enumerate(items):
#                     we need to have tick labeling to be dynamic?
#                     adding it to the dict wont work, creates horrible figure
#                     unless we can force a selection?
                    name = f"{dim}, {qp.loc[qp.index[0], dim]: .2g}"
                    entry = dict(name=name, n_items=len(items) + 1,
                                 item=item, id=int(i),
                                 x=j / len(items))
                    nominal_vars.append(entry)

        boxes = pd.concat(boxes)
        nominal_vars = pd.DataFrame(nominal_vars)

        width = 400
        height = width

        point_selector = alt.selection_single(fields=['id'])

        peeling = self.peeling_trajectory.copy()
        peeling['id'] = peeling.index

        chart = alt.Chart(peeling).mark_circle(
            size=75).encode(
            x='coverage:Q',
            y='density:Q',
            color=alt.Color(
                'res_dim:O',
                scale=alt.Scale(
                    range=sns.color_palette(
                        'YlGnBu',
                        n_colors=8).as_hex())),
            opacity=alt.condition(
                point_selector,
                alt.value(1),
                alt.value(0.4))).properties(
            selection=point_selector).properties(
            width=width,
            height=height)

        # conda update -c conda-forge altair to 2.1
        # move this to encoding tooltip=[<list of items>]
        chart.encoding.tooltip = [
            {"type": "ordinal",
             "field": "id"},
            {"type": "quantitative",
             "field": "coverage", "format": ".2"},
            {"type": "quantitative",
             "field": "density", "format": ".2"},
            {"type": "ordinal", "field": "res_dim", }
        ]

        base = alt.Chart(boxes).encode(
            x=alt.X('x_lower:Q', axis=alt.Axis(grid=False,
                                               title='box limits',
                                               labels=False),
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

        texts1 = base.mark_text(
            baseline='top', dy=5, align='left').encode(
            text=alt.Text('text:O')).transform_calculate(
            text=(
                'datum.qp_lower>0?'
                'format(datum.x1, ".2")+" ("+format(datum.qp_lower, ".1~g")+")" :'
                'format(datum.x1, ".2")'))

        texts2 = base.mark_text(
            baseline='top', dy=5, align='right').encode(
            text=alt.Text('text:O'),x='x_upper:Q').transform_calculate(
            text=(
                'datum.qp_upper>0?'
                'format(datum.x2, ".2")+" ("+format(datum.qp_upper, ".1")+")" :'
                'format(datum.x2, ".2")'))

        data = pd.DataFrame([dict(start=0, end=1)])
        rect = alt.Chart(data).mark_rect(opacity=0.05).encode(
            x='start:Q',
            x2='end:Q',
        )

        # TODO:: for qp can we do something with the y encoding here and
        # connecting this to a selection?
        # seems tricky, no clear way to control the actual labels
        # or can we use the text channel identical to the above?
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

    def resample(self, i=None, iterations=10, p=1 / 2):
        '''Calculate resample statistics for candidate box i

        Parameters
        ----------
        i : int, optional
        iterations : int, optional
        p : float, optional


        Returns
        -------
        DataFrame

        '''
        if i is None:
            i = self._cur_box

        x = self.prim.x.loc[self.yi_initial, :]
        y = self.prim.y[self.yi_initial]

        if len(self._resampled) < iterations:
            with temporary_filter(__name__, INFO, 'find_box'):
                for j in range(len(self._resampled), iterations):
                    _logger.info('resample {}'.format(j))
                    index = np.random.choice(x.index, size=int(x.shape[0] * p),
                                             replace=False)
                    x_temp = x.loc[index, :].reset_index(drop=True)
                    y_temp = y[index]

                    box = Prim(x_temp, y_temp, threshold=0.1,
                               peel_alpha=self.prim.peel_alpha,
                               paste_alpha=self.prim.paste_alpha).find_box()
                    self._resampled.append(box)

        counters = []
        for _ in range(2):
            counter = {column: 0.0 for column in x.columns}
            counters.append(counter)

        coverage = self.peeling_trajectory.coverage[i]
        density = self.peeling_trajectory.density[i]

        for box in self._resampled:
            coverage_index = (
                box.peeling_trajectory.coverage -
                coverage).abs().idxmin()
            density_index = (
                box.peeling_trajectory.density -
                density).abs().idxmin()
            for counter, index in zip(counters, [coverage_index,
                                                 density_index]):
                for unc in box.qp[index].keys():
                    counter[unc] += 1 / iterations

        scores = pd.DataFrame(counters,
                              index=['reproduce coverage',
                                     'reproduce density'],
                              columns=box.box_lim.columns).T * 100
        return scores.sort_values(by=['reproduce coverage',
                                      'reproduce density'],
                                  ascending=False)

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

    def drop_restriction(self, uncertainty='', i=-1):
        '''Drop the restriction on the specified dimension for box i

        Parameters
        ----------
        i : int, optional
            defaults to the currently selected box, which
            defaults to the latest box on the trajectory
        uncertainty : str


        Replace the limits in box i with a new box where
        for the specified uncertainty the limits of the initial box are
        being used. The resulting box is added to the peeling trajectory.

        '''
        if i == -1:
            i = self._cur_box

        new_box_lim = self.box_lims[i].copy()
        new_box_lim.loc[:, uncertainty] = self.box_lims[0].loc[:, uncertainty]
        indices = sdutil._in_box(self.prim.x.loc[self.prim.yi_remaining, :],
                                 new_box_lim)
        indices = self.prim.yi_remaining[indices]
        self.update(new_box_lim, indices)

    def update(self, box_lims, indices):
        '''update the box to the provided box limits.

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

        data = {'coverage': coi / self.prim.t_coi,
                'density': coi / y.shape[0],
                'mean': np.mean(y),
                'res_dim': restricted_dims.shape[0],
                'mass': y.shape[0] / self.prim.n,
                'id': i}
        new_row = pd.DataFrame([data])
        self.peeling_trajectory = self.peeling_trajectory.append(
            new_row, ignore_index=True, sort=True)

        # boxlims
        qp = self._calculate_quasi_p(i, restricted_dims)
        self.qp.append(qp)
        self._cur_box = len(self.peeling_trajectory) - 1

    def show_ppt(self):
        '''show the peeling and pasting trajectory in a figure'''
        return sdutil.plot_ppt(self.peeling_trajectory)

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
        return sdutil.plot_tradeoff(self.peeling_trajectory, cmap=cmap)

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
        if i is None:
            i = self._cur_box

        resdim = sdutil._determine_restricted_dims(self.box_lims[i],
                                                   self.prim.box_init)

        return sdutil.plot_pair_wise_scatter(self.prim.x.iloc[self.yi_initial,:],
                                             self.prim.y[self.yi_initial],
                                             self.box_lims[i],
                                             self.prim.box_init,
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

        # TODO use apply on df?

        qp_values = box_lim.apply(calculate_qp, axis=0, result_type='expand',
                                  args=[x, y, Hbox, Tbox, box_lim,
                                        self.box_lims[0]])
        qp_values = qp_values.to_dict(orient='list')
        return qp_values


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
    mode : {RuleInductionType.BINARY, RuleInductionType.REGRESSION}, optional
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

    def __init__(self, x, y, threshold,
                 obj_function=PRIMObjectiveFunctions.LENIENT1,
                 peel_alpha=0.05, paste_alpha=0.05, mass_min=0.05,
                 threshold_type=ABOVE, mode=sdutil.RuleInductionType.BINARY,
                 update_function='default'):
        assert mode in {sdutil.RuleInductionType.BINARY,
                        sdutil.RuleInductionType.REGRESSION}
        assert self._assert_mode(y, mode, update_function)
        # preprocess x
        try:
            x.drop(columns='scenario', inplace=True)
        except KeyError:
            pass
        x = x.reset_index(drop=True)

        x_float = x.select_dtypes([np.float, np.float32, np.float64, float])
        self.x_float = x_float.values
        self.x_float_colums = x_float.columns.values

        x_int = x.select_dtypes([np.int, np.int32, np.int64, int])
        self.x_int = x_int.values
        self.x_int_columns = x_int.columns.values

        self.x_numeric_columns = np.concatenate([self.x_float_colums,
                                                 self.x_int_columns])

        x_nominal = x.select_dtypes(exclude=np.number)

        # filter out dimensions with only single value
        for column in x_nominal.columns.values:
            if np.unique(x[column]).shape == (1,):
                x = x.drop(column, axis=1)
                _logger.info(("{} dropped from analysis "
                              "because only a single category").format(column))

        x_nominal = x.select_dtypes(exclude=np.number)
        self.x_nominal = x_nominal.values
        self.x_nominal_columns = x_nominal.columns.values

        self.n_cols = x.columns.shape[0]

        for column in self.x_nominal_columns:
            x[column] = x[column].astype('category')

        self.x = x
        self.y = y
        self.mode = mode

        self._update_yi_remaining = self._update_functions[update_function]

        if len(self.y.shape) > 1:
            raise PrimException("y is not a 1-d array")
        if self.y.shape[0] != len(self.x):
            raise PrimException("len(y) != len(x)")

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

    def find_box(self):
        '''Execute one iteration of the PRIM algorithm. That is, find one
        box, starting from the current state of Prim.'''
        # set the indices
        self._update_yi_remaining(self)

        # make boxes already found immutable
        for box in self._boxes:
            box._frozen = True

        if self.yi_remaining.shape[0] == 0:
            _logger.info("no data remaining")
            return

        # log how much data and how many coi are remaining
        _logger.info(
            self.message.format(
                self.yi_remaining.shape[0],
                self.determine_coi(
                    self.yi_remaining)))

        # make a new box that contains all the remaining data points
        box = PrimBox(self, self.box_init, self.yi_remaining[:])

        #  perform peeling phase
        box = self._peel(box)
        _logger.debug("peeling completed")

        # perform pasting phase
        box = self._paste(box)
        _logger.debug("pasting completed")

        message = ("mean: {0}, mass: {1}, coverage: {2}, "
                   "density: {3} restricted_dimensions: {4}")
        message = message.format(box.mean,
                                 box.mass,
                                 box.coverage,
                                 box.density,
                                 box.res_dim)

        if (self.threshold_type == ABOVE) &\
           (box.mean >= self.threshold):
            _logger.info(message)
            self._boxes.append(box)
            return box
        elif (self.threshold_type == BELOW) &\
                (box.mean <= self.threshold):
            _logger.info(message)
            self._boxes.append(box)
            return box
        else:
            # make a dump box
            _logger.info(('box does not meet threshold criteria, '
                          'value is {}, returning dump box').format(
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

        mass_old = box.yi.shape[0] / self.n

        x_float = self.x_float[box.yi]
        x_int = self.x_int[box.yi]
        x_nominal = self.x_nominal[box.yi]

        # identify all possible peels
        possible_peels = []

        for x, columns, dtype, in [(x_float, self.x_float_colums,
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
            obj = self.obj_func(self, self.y[box.yi], self.y[i])
            non_res_dim = self.n_cols -\
                sdutil._determine_nr_restricted_dims(box_lim,
                                                     self.box_init)
            score = (obj, non_res_dim, box_lim, i)
            scores.append(score)

        scores.sort(key=itemgetter(0, 1), reverse=True)
        entry = scores[0]

        obj_score = entry[0]
        box_new, indices = entry[2:]

        mass_new = self.y[indices].shape[0] / self.n

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
                peel_alpha = 1 - self.peel_alpha
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

    def _discrete_peel(self, box, u, j, x):
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
                peel_alpha = 1 - self.peel_alpha
                i = 1

            box_peel = get_quantile(xj, peel_alpha)
            box_peel = int(box_peel)

            # determine logical associated with peel value
            if direction == 'lower':
                if box_peel == box_lim.loc[i, u]:
                    logical = (xj > box_lim.loc[i, u]) &\
                              (xj <= box_lim.loc[i + 1, u])
                else:
                    logical = (xj >= box_peel) &\
                              (xj <= box_lim.loc[i + 1, u])
            if direction == 'upper':
                if box_peel == box_lim.loc[i, u]:
                    logical = (xj < box_lim.loc[i, u]) &\
                              (xj >= box_lim.loc[i - 1, u])
                else:
                    logical = (xj <= box_peel) &\
                              (xj >= box_lim.loc[i - 1, u])

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
                    for element in x[:, j]:
                        if element != entry:
                            bools.append(True)
                        else:
                            bools.append(False)
                    logical = np.asarray(bools, dtype=bool)
                else:
                    logical = x[:, j] != entry
                indices = box.yi[logical]
                peels.append((indices, temp_box))
            return peels
        else:
            # no peels possible, return empty list
            return []

    def _paste(self, box):
        ''' Executes the pasting phase of the PRIM. Delegates pasting
        to data type specific helper methods.'''

        mass_old = box.yi.shape[0] / self.n

        # need to break this down by dtype
        restricted_dims = sdutil._determine_restricted_dims(box.box_lims[-1],
                                                            self.box_init)
        res_dim = set(restricted_dims)

        x = self.x.loc[self.yi_remaining, :]

        # identify all possible pastes
        possible_pastes = []
        for columns, dtype, in [(self.x_float_colums,
                                 'float'),
                                (self.x_int_columns,
                                 'int'),
                                (self.x_nominal_columns,
                                 'object')]:
            for i, u in enumerate(columns):
                if u not in res_dim:
                    continue
                _logger.debug("pasting " + u)
                pastes = self._pastes[dtype](self, box, u, x,
                                             restricted_dims)
                [possible_pastes.append(entry) for entry in pastes]
            if not possible_pastes:
                # there is no peel identified, so return box
                return box

        # determine the scores for each peel in order
        # to identify the next candidate box
        scores = []
        for entry in possible_pastes:
            i, box_lim = entry
            obj = self.obj_func(self, self.y[box.yi], self.y[i])
            non_res_dim = len(x.columns) -\
                sdutil._determine_nr_restricted_dims(box_lim,
                                                     self.box_init)
            score = (obj, non_res_dim, box_lim, i)
            scores.append(score)

        scores.sort(key=itemgetter(0, 1), reverse=True)
        entry = scores[0]
        obj, _, box_new, indices = entry
        mass_new = self.y[indices].shape[0] / self.n

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

    def _real_paste(self, box, u, x, resdim):
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
        boxlim = box.box_lims[-1]

        for i, direction in enumerate(['lower', 'upper']):
            box_paste = boxlim.copy()
            # box containing data candidate for pasting
            paste_box = boxlim.copy()

            minimum, maximum = self.box_init[u].values

            if direction == 'lower':
                paste_box.loc[:, u] = minimum, box_paste.loc[0, u]

                indices = sdutil._in_box(x[resdim], paste_box[resdim])
                data = x.loc[indices, u]

                paste_value = minimum
                if data.size > 0:
                    paste_value = get_quantile(data, 1 - self.paste_alpha)

                assert paste_value <= boxlim.loc[i, u]
            else:  # direction == 'upper':
                paste_box.loc[:, u] = paste_box.loc[1, u], maximum

                indices = sdutil._in_box(x[resdim], paste_box[resdim])
                data = x.loc[indices, u]

                paste_value = maximum
                if data.size > 0:
                    paste_value = get_quantile(data, self.paste_alpha)

                assert paste_value >= box.box_lims[-1].loc[i, u]

            dtype = box_paste[u].dtype
            if dtype == np.int32:
                paste_value = np.int(paste_value)

            box_paste.loc[i, u] = paste_value
            logical = sdutil._in_box(x[resdim], box_paste[resdim])
            indices = self.yi_remaining[logical]

            pastes.append((indices, box_paste))

        return pastes

    def _categorical_paste(self, box, u, x, resdim):
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
        box_lim = box.box_lims[-1]

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

                indices = sdutil._in_box(x[resdim],
                                         box_paste[resdim])
                indices = self.yi_remaining[indices]
                pastes.append((indices, box_paste))
            return pastes
        else:
            # no pastes possible, return empty list
            return []

    def _lenient1_obj_func(self, y_old, y_new):
        r'''
        the default objective function used by prim, instead of the
        original objective function, This function can cope with
        continuous, integer, and categorical uncertainties. The basic
        idea is that the gain in mean is divided by the loss in mass.

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
        data points in the box. This objective function offsets a
        problem in case of categorical data where the normal objective
        function often results in boxes mainly based on the categorical
        data.

        '''
        mean_old = np.mean(y_old)

        if y_new.shape[0] > 0:
            mean_new = np.mean(y_new)
        else:
            mean_new = 0

        obj = 0
        if mean_old != mean_new:
            if y_old.shape[0] > y_new.shape[0]:
                obj = (mean_new - mean_old) / (y_old.shape[0] - y_new.shape[0])
            elif y_old.shape[0] < y_new.shape[0]:
                obj = (mean_new - mean_old) / (y_new.shape[0] - y_old.shape[0])
            else:
                raise PrimException(
                    '''mean is different {} vs {}, while shape is the same,
                                       this cannot be the case'''.format(
                        mean_old, mean_new))
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
                raise PrimException(
                    '''mean is different {} vs {}, while shape is the same,
                                       this cannot be the case'''.format(
                        mean_old, mean_new))

            change_mean = mean_new - mean_old
            change_mass = abs(y_old.shape[0] - y_new.shape[0])
            mass_new = y_new.shape[0]

            obj = mass_new * change_mean / change_mass

        return obj

    def _original_obj_func(self, y_old, y_new):
        ''' The original objective function: the mean of the data
        inside the box'''

        if y_new.shape[0] > 0:
            return np.mean(y_new)
        else:
            return -1

    def _assert_mode(self, y, mode, update_function):
        if mode == sdutil.RuleInductionType.BINARY:
            return set(np.unique(y)) == {0, 1}
        if update_function == 'guivarch':
            return False
        return True

    def _assert_dtypes(self, keys, dtypes):
        '''
        helper fucntion that checks whether none of the provided keys
        has a dtype object as value.
        '''

        for key in keys:
            if dtypes[key][0] == np.dtype(object):
                raise EMAError(
                    "%s has dtype object and can thus not be rotated" % key)
        return True

    _peels = {'object': _categorical_peel,
              'int': _discrete_peel,
              'float': _real_peel}

    _pastes = {'object': _categorical_paste,
               'int': _real_paste,
               'float': _real_paste}

    # dict with the various objective functions available
    # todo:: move functions themselves to ENUM?
    _obj_functions = {PRIMObjectiveFunctions.LENIENT2: _lenient2_obj_func,
                      PRIMObjectiveFunctions.LENIENT1: _lenient1_obj_func,
                      PRIMObjectiveFunctions.ORIGINAL: _original_obj_func}

    _update_functions = {'default': _update_yi_remaining_default,
                         'guivarch': _update_yi_remaining_guivarch}
