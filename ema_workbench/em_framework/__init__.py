"""em_framework namespace."""

import warnings

__all__ = [
    "ArchiveLogger",
    "ArrayOutcome",
    "BooleanParameter",
    "CategoricalParameter",
    "Category",
    "Constant",
    "Constraint",
    "Convergence",
    "EpsilonIndicatorMetric",
    "EpsilonProgress",
    "ExperimentReplication",
    "FASTSampler",
    "FileModel",
    "FullFactorialSampler",
    "GenerationalDistanceMetric",
    "HypervolumeMetric",
    "IntegerParameter",
    "InvertedGenerationalDistanceMetric",
    "IpyparallelEvaluator",
    "LHSSampler",
    "MPIEvaluator",
    "Model",
    "MonteCarloSampler",
    "MorrisSampler",
    "MultiprocessingEvaluator",
    "OutputSpaceExploration",
    "Policy",
    "RealParameter",
    "ReplicatorModel",
    "ReplicatorModel",
    "Samplers",
    "ScalarOutcome",
    "Scenario",
    "SequentialEvaluator",
    "SobolSampler",
    "SpacingMetric",
    "TimeSeriesOutcome",
    "epsilon_nondominated",
    "get_SALib_problem",
    "model",
    "optimize",
    "outcomes",
    "parameters",
    "parameters_from_csv",
    "parameters_to_csv",
    "perform_experiments",
    "rebuild_platypus_population",
    "samplers",
    "to_problem",
    "to_robust_problem",
]

from .evaluators import (
    Samplers,
    SequentialEvaluator,
    optimize,
    perform_experiments,
)
from .model import FileModel, Model, ReplicatorModel
from .optimization import (
    ArchiveLogger,
    Convergence,
    EpsilonIndicatorMetric,
    EpsilonProgress,
    GenerationalDistanceMetric,
    HypervolumeMetric,
    InvertedGenerationalDistanceMetric,
    SpacingMetric,
    epsilon_nondominated,
    rebuild_platypus_population,
    to_problem,
    to_robust_problem,
)
from .outcomes import ArrayOutcome, Constraint, ScalarOutcome, TimeSeriesOutcome
from .parameters import (
    BooleanParameter,
    CategoricalParameter,
    Category,
    Constant,
    IntegerParameter,
    RealParameter,
    parameters_from_csv,
    parameters_to_csv,
)
from .points import ExperimentReplication, Policy, Scenario
from .salib_samplers import FASTSampler, MorrisSampler, SobolSampler, get_SALib_problem
from .samplers import (
    FullFactorialSampler,
    LHSSampler,
    MonteCarloSampler,
)

try:
    from .futures_ipyparallel import IpyparallelEvaluator
except ImportError:
    warnings.warn("ipyparallel not installed - IpyparalleEvaluator not available", stacklevel=2)
    IpyparallelEvaluator = None

from .futures_mpi import MPIEvaluator
from .futures_multiprocessing import MultiprocessingEvaluator
from .outputspace_exploration import OutputSpaceExploration

del warnings
