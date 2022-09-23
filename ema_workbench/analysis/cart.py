"""
A scenario discovery oriented implementation of CART. It essentially is
a wrapper around scikit-learn's version of CART.

"""
import io
import math
from io import StringIO

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree

from ema_workbench.util.ema_exceptions import EMAError
from . import scenario_discovery_util as sdutil
from ..util import get_module_logger

# Created on May 22, 2015
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


__all__ = ["setup_cart", "CART"]
_logger = get_module_logger(__name__)


def setup_cart(results, classify, incl_unc=None, mass_min=0.05):
    """helper function for performing cart in combination with data
    generated by the workbench.

    Parameters
    ----------
    results : tuple of DataFrame and dict with numpy arrays
              the return from :meth:`perform_experiments`.
    classify : string, function or callable
               either a string denoting the outcome of interest to
               use or a function.
    incl_unc : list of strings, optional
    mass_min : float, optional


    Raises
    ------
    TypeError
        if classify is not a string or a callable.

    """
    x, outcomes = results

    if incl_unc is not None:
        drop_names = set(x.columns.values.tolist()) - set(incl_unc)
        x = x.drop(drop_names, axis=1)

    if isinstance(classify, str):
        y = outcomes[classify]
        mode = sdutil.RuleInductionType.REGRESSION
    elif callable(classify):
        y = classify(outcomes)
        mode = sdutil.RuleInductionType.BINARY
    else:
        raise TypeError("unknown type for classify")

    return CART(x, y, mass_min, mode=mode)


class CART(sdutil.OutputFormatterMixin):
    """CART algorithm

    can be used in a manner similar to PRIM. It provides access
    to the underlying tree, but it can also show the boxes described by the
    tree in a table or graph form similar to prim.

    Parameters
    ----------
    x : DataFrame
    y : 1D ndarray
    mass_min : float, optional
               a value between 0 and 1 indicating the minimum fraction
               of data points in a terminal leaf. Defaults to 0.05,
               identical to prim.
    mode : {BINARY, CLASSIFICATION, REGRESSION}
           indicates the mode in which CART is used. Binary indicates
           binary classification, classification is multiclass, and regression
           is regression.

    Attributes
    ----------
    boxes : list
            list of DataFrame box lims
    stats : list
            list of dicts with stats

    Notes
    -----
    This class is a wrapper around scikit-learn's CART algorithm. It provides
    an interface to CART that is more oriented towards scenario discovery, and
    shared some methods with PRIM

    See also
    --------
    :mod:`prim`

    """

    sep = "!?!"

    def __init__(self, x, y, mass_min=0.05, mode=sdutil.RuleInductionType.BINARY):
        """init"""

        try:
            x = x.drop(["scenario"], axis=1)
        except KeyError:
            pass

        self.x = x
        self.y = y
        self.mass_min = mass_min
        self.mode = mode

        # we need to transform the DataFrame into a ndarray
        # we use dummy variables for each category in case of categorical
        # variables. Integers are treated as floats
        dummies = pd.get_dummies(self.x, prefix_sep=self.sep)

        self.dummiesmap = {}
        for column, values in x.select_dtypes(exclude=np.number).iteritems():
            mapping = {str(entry): entry for entry in values.unique()}
            self.dummiesmap[column] = mapping

        self.feature_names = dummies.columns.values.tolist()
        self._x = dummies.values
        self._boxes = None
        self._stats = None

    @property
    def boxes(self):
        """

        Returns
        -------
        list with boxlims for each terminal leaf

        """

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
        leafs = np.argwhere(left == -1)[:, 0]

        def recurse(left, right, child, lineage=None):
            if lineage is None:
                # lineage = [self.clf.tree_.value[child]]
                lineage = []

            if child in left:
                parent = np.where(left == child)[0].item()
                split = "l"
            else:
                parent = np.where(right == child)[0].item()
                split = "r"

            lineage.append((parent, split, threshold[parent], features[parent]))

            if parent == 0:
                lineage.reverse()
                return lineage
            else:
                return recurse(left, right, parent, lineage)

        box_init = sdutil._make_box(self.x)
        boxes = []
        for leaf in leafs:
            branch = recurse(left, right, leaf)
            box = box_init.copy()
            for node in branch:
                direction = node[1]
                value = node[2]
                unc = node[3]

                if direction == "l":
                    if unc in box_init.columns:
                        box.loc[1, unc] = value
                    else:
                        unc, cat = unc.split(self.sep)
                        cats = box.loc[0, unc]
                        # TODO:: cat is a str needs casting?
                        # what about a lookup table mapping
                        # each str cat to the associate actual cat
                        # object
                        # can be created when making the dummy variables

                        cats.discard(self.dummiesmap[unc][cat])
                        box.loc[:, unc] = [set(cats), set(cats)]
                else:
                    if unc in box_init.columns:
                        if box[unc].dtype == np.int32:
                            value = math.ceil(value)
                        box.loc[0, unc] = value

            boxes.append(box)
        self._boxes = boxes
        return self._boxes

    @property
    def stats(self):
        """

        Returns
        -------
        list with scenario discovery statistics for each terminal leaf

        """

        if self._stats:
            return self._stats

        boxes = self.boxes

        box_init = sdutil._make_box(self.x)

        self._stats = []
        for box in boxes:
            boxstats = self._boxstat_methods[self.mode](self, box, box_init)
            self._stats.append(boxstats)
        return self._stats

    def _binary_stats(self, box, box_init):
        indices = sdutil._in_box(self.x, box)

        y_in_box = self.y[indices]
        box_coi = np.sum(y_in_box)

        boxstats = {
            "coverage": box_coi / np.sum(self.y),
            "density": box_coi / y_in_box.shape[0],
            "res dim": sdutil._determine_nr_restricted_dims(box, box_init),
            "mass": y_in_box.shape[0] / self.y.shape[0],
        }
        return boxstats

    def _regression_stats(self, box, box_init):
        indices = sdutil._in_box(self.x, box)

        y_in_box = self.y[indices]

        boxstats = {
            "mean": np.mean(y_in_box),
            "mass": y_in_box.shape[0] / self.y.shape[0],
            "res dim": sdutil._determine_nr_restricted_dims(box, box_init),
        }
        return boxstats

    def _classification_stats(self, box, box_init):
        indices = sdutil._in_box(self.x, box)

        y_in_box = self.y[indices]
        classes = set(self.y)
        classes = sorted(classes)

        counts = [y_in_box[y_in_box == ci].shape[0] for ci in classes]

        total_gini = 0
        for count in counts:
            total_gini += (count / y_in_box.shape[0]) ** 2
        gini = 1 - total_gini

        boxstats = {
            "gini": gini,
            "mass": y_in_box.shape[0] / self.y.shape[0],
            "box_composition": counts,
            "res dim": sdutil._determine_nr_restricted_dims(box, box_init),
        }

        return boxstats

    _boxstat_methods = {
        sdutil.RuleInductionType.BINARY: _binary_stats,
        sdutil.RuleInductionType.REGRESSION: _regression_stats,
        sdutil.RuleInductionType.CLASSIFICATION: _classification_stats,
    }

    def build_tree(self):
        """train CART on the data"""
        min_samples = int(self.mass_min * self.x.shape[0])

        if self.mode == sdutil.RuleInductionType.REGRESSION:
            self.clf = tree.DecisionTreeRegressor(min_samples_leaf=min_samples)
        else:
            self.clf = tree.DecisionTreeClassifier(min_samples_leaf=min_samples)
        self.clf.fit(self._x, self.y)

    def show_tree(self, mplfig=True, format="png"):
        """return a png (defaults) or svg of the tree

        On Windows, graphviz needs to be installed with conda.

        Parameters
        ----------
        mplfig : bool, optional
                 if true (default) returns a matplotlib figure with the tree,
                 otherwise, it returns the output as bytes
        format : {'png', 'svg'}, default 'png'
                 Gives a format of the output.

        """
        assert self.clf
        import pydot  # dirty hack for read the docs

        dot_data = StringIO()
        tree.export_graphviz(self.clf, out_file=dot_data, feature_names=self.feature_names)
        dot_data = dot_data.getvalue()  # .encode('ascii') # @UndefinedVariable
        graphs = pydot.graph_from_dot_data(dot_data)

        # FIXME:: pydot now always returns a list, used to be either a
        # singleton or a list. This is a stopgap which might be sufficient
        # but just in case, we raise an error if assumption of len==1 does
        # not hold
        if len(graphs) > 1:
            raise EMAError("trying to visualize more than one tree")

        graph = graphs[0]

        if format == "png":
            img = graph.create_png()
            if mplfig:
                fig, ax = plt.subplots()
                ax.imshow(mpimg.imread(io.BytesIO(img)))
                ax.axis("off")
                return fig
        elif format == "svg":
            img = graph.create_svg()
        else:
            raise TypeError("""format must be in {'png', 'svg'}""")

        return img
