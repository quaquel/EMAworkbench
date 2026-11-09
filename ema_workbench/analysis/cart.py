"""A scenario discovery oriented implementation of CART.

It essentially is a wrapper around scikit-learn's version of CART, with some scneario
discovery specific functionality.

"""

from __future__ import annotations

import contextlib
import io
from io import StringIO

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import tree

from ema_workbench.util.ema_exceptions import EMAError

from ..util import get_module_logger
from . import scenario_discovery_util as sdutil

# Created on May 22, 2015
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


__all__ = ["CART"]
_logger = get_module_logger(__name__)


class CART(sdutil.OutputFormatterMixin):
    """CART algorithm.

    Can be used in a manner similar to PRIM. It provides access
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

    See Also
    --------
    :mod:`prim`

    """

    sep = "!?!"

    def __init__(
        self,
        x: pd.DataFrame,
        y: npt.NDArray,
        mass_min: float = 0.05,
        mode: sdutil.RuleInductionType = sdutil.RuleInductionType.BINARY,
    ) -> None:
        """Init."""
        with contextlib.suppress(KeyError):
            x = x.drop(["scenario"], axis=1)

        self.x = x
        self.y = y
        self.mass_min = mass_min
        self.mode = mode

        # we need to transform the DataFrame into a ndarray
        # we use dummy variables for each category in case of categorical
        # variables. Integers are treated as floats
        dummies = pd.get_dummies(self.x, prefix_sep=self.sep)

        self.dummiesmap = {}
        for column, values in x.select_dtypes(exclude=np.number).items():
            mapping = {str(entry): entry for entry in values.unique()}
            self.dummiesmap[column] = mapping

        self.feature_names = dummies.columns.values.tolist()
        self._x = dummies.values
        self._boxes = None
        self._stats = None

    @property
    def boxes(self) -> list[pd.DataFrame]:
        """Return a list with the box limits for each terminal leaf.

        Returns
        -------
        list with boxlims for each terminal leaf

        """
        if self._boxes:
            return self._boxes

        assert self.clf
        self._boxes = [
            box
            for _, box in sdutil._extract_tree_boxes(
                self.clf, self.x, self.feature_names, self.dummiesmap, self.sep
            )
        ]
        return self._boxes

    @property
    def stats(self) -> list[dict]:
        """Returns list with the scenario discovery statistics for each terminal leaf.

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

    def _binary_stats(self, box: pd.DataFrame, box_init: pd.DataFrame) -> dict:
        indices = sdutil._in_box(self.x, box)

        y_in_box = self.y[indices]
        box_coi = np.sum(y_in_box)

        boxstats = {
            "coverage": box_coi / np.sum(self.y),
            "density": box_coi / y_in_box.shape[0],
            "res_dim": sdutil._determine_nr_restricted_dims(box, box_init),
            "mass": y_in_box.shape[0] / self.y.shape[0],
        }
        return boxstats

    def _regression_stats(self, box: pd.DataFrame, box_init: pd.DataFrame) -> dict:
        indices = sdutil._in_box(self.x, box)

        y_in_box = self.y[indices]

        boxstats = {
            "mean": np.mean(y_in_box),
            "mass": y_in_box.shape[0] / self.y.shape[0],
            "res_dim": sdutil._determine_nr_restricted_dims(box, box_init),
        }
        return boxstats

    def _classification_stats(self, box: pd.DataFrame, box_init: pd.DataFrame) -> dict:
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
            "purity": sdutil._compute_purity(counts),
            "mass": y_in_box.shape[0] / self.y.shape[0],
            "box_composition": counts,
            "res_dim": sdutil._determine_nr_restricted_dims(box, box_init),
        }

        return boxstats

    _boxstat_methods = {
        sdutil.RuleInductionType.BINARY: _binary_stats,
        sdutil.RuleInductionType.REGRESSION: _regression_stats,
        sdutil.RuleInductionType.CLASSIFICATION: _classification_stats,
    }

    def extract_rules(self, tolerance: float = 0.0) -> dict:
        """Return, for each class, the region where the tree predicts it as one rule.

        Each rule is the union of the boxes of the leaves that predict the
        class, written with shared conditions factored out. With the default
        tolerance of 0, a rule selects exactly the points in those leaves.

        Parameters
        ----------
        tolerance : float, optional
                    drop near vacuous splits from a rule if that changes the
                    membership of at most this share of the points the rule
                    describes. Gives shorter rules that are no longer exactly
                    equivalent to the boxes.

        Returns
        -------
        dict
            class -> :class:`~ema_workbench.analysis.scenario_discovery_util.Rule`

        Raises
        ------
        EMAError
            in regression mode, where leaves do not predict classes

        """
        assert self.clf
        if self.mode == sdutil.RuleInductionType.REGRESSION:
            raise EMAError("rules are only available in (binary) classification mode")

        tree_ = self.clf.tree_
        leaves = np.flatnonzero(tree_.children_left == -1)
        predicted = {
            leaf: self.clf.classes_[np.argmax(tree_.value[leaf][0])] for leaf in leaves
        }
        return {
            klass: sdutil._build_tree_rule(
                self.clf,
                self.x,
                self._x,
                self.feature_names,
                {leaf for leaf in leaves if predicted[leaf] == klass},
                self.dummiesmap,
                self.sep,
                tolerance,
            )
            for klass in self.clf.classes_
        }

    def build_tree(self) -> None:
        """Train CART on the data."""
        min_samples = int(self.mass_min * self.x.shape[0])

        if self.mode == sdutil.RuleInductionType.REGRESSION:
            self.clf = tree.DecisionTreeRegressor(min_samples_leaf=min_samples)
        else:
            self.clf = tree.DecisionTreeClassifier(min_samples_leaf=min_samples)
        self.clf.fit(self._x, self.y)

    def show_tree(self, mplfig: bool = True, format: str = "png"):
        """Return a png (defaults) or svg of the tree.

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
        import pydot  # noqa: PLC0415 dirty hack for read the docs

        dot_data = StringIO()
        tree.export_graphviz(
            self.clf, out_file=dot_data, feature_names=self.feature_names
        )
        dot_data = dot_data.getvalue()  # .encode('ascii') # @UndefinedVariable
        graphs = pydot.graph_from_dot_data(dot_data)

        # FIXME:: pydot now always returns a list, used to be either a
        # singleton or a list. This is a stopgap which might be sufficient
        # but just in case, we raise an error if assumption of len==1 does
        # not hold
        if len(graphs) > 1:
            raise EMAError(
                f"Expected a single tree for visualization, but found {len(graphs)} trees."
            )

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
            raise TypeError(f"format must be 'png' or 'svg' (instead of {format}).")

        return img
