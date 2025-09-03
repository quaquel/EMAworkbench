"""Main namespace for analysis package."""

# importing anything from analysis segfaults java with netlogo on a mac
# for now no clue why
#

__all__ = [
    "CART",
    "Density",
    "DiagKind",
    "Logit",
    "PlotType",
    "Prim",
    "RuleInductionType",
    "envelopes",
    "get_ex_feature_scores",
    "get_feature_scores_all",
    "get_rf_feature_scores",
    "get_univariate_feature_scores",
    "kde_over_time",
    "lines",
    "multiple_densities",
    "pairs_plotting",
    "pca_preprocess",
    "run_constrained_prim",
    "set_fig_to_bw",
    "setup_cart",
    "setup_prim"
]

from . import pairs_plotting
from .b_and_w_plotting import set_fig_to_bw
from .cart import CART, setup_cart
from .feature_scoring import (
    get_ex_feature_scores,
    get_feature_scores_all,
    get_rf_feature_scores,
    get_univariate_feature_scores,
)
from .logistic_regression import Logit
from .plotting import envelopes, kde_over_time, lines, multiple_densities
from .plotting_util import Density, PlotType
from .prim import Prim, pca_preprocess, run_constrained_prim, setup_prim
from .prim_util import DiagKind
from .scenario_discovery_util import RuleInductionType
