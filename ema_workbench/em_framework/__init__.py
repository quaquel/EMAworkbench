import warnings

__all__ = [
    "parameters",
    "model",
    "outcomes",
    "samplers",
    "Model",
    "FileModel",
    "ReplicatorModel",
    "ScalarOutcome",
    "TimeSeriesOutcome",
    "Constraint",
    "RealParameter",
    "IntegerParameter",
    "CategoricalParameter",
    "BooleanParameter",
    "Scenario",
    "Policy",
    "ExperimentReplication",
    "Constant",
    "parameters_from_csv",
    "parameters_to_csv",
    "Category",
    "SobolSampler",
    "MorrisSampler",
    "get_SALib_problem",
    "FASTSampler",
    "perform_experiments",
    "optimize",
    "IpyparallelEvaluator",
    "MPIEvaluator",
    "MultiprocessingEvaluator",
    "SequentialEvaluator",
    "ReplicatorModel",
    "ArrayOutcome",
    "Samplers",
    "OutputSpaceExploration",
    "EpsilonProgress",
    "ArchiveLogger",
    "HypervolumeMetric",
    "GenerationalDistanceMetric",
    "SpacingMetric",
    "InvertedGenerationalDistanceMetric",
    "EpsilonIndicatorMetric",
    "epsilon_nondominated",
    "rebuild_platypus_population",
    "to_problem",
    "to_robust_problem",
]

from .outcomes import ScalarOutcome, TimeSeriesOutcome, Constraint, ArrayOutcome
from .model import Model, FileModel, ReplicatorModel, Replicator, SingleReplication

from .parameters import (
    RealParameter,
    IntegerParameter,
    CategoricalParameter,
    BooleanParameter,
    Constant,
    parameters_from_csv,
    parameters_to_csv,
    Category,
)
from .samplers import (
    MonteCarloSampler,
    FullFactorialSampler,
    LHSSampler,
    sample_levers,
    sample_uncertainties,
    sample_parameters,
)
from .points import Scenario, Policy, Point, ExperimentReplication, experiment_generator

from .salib_samplers import SobolSampler, MorrisSampler, FASTSampler, get_SALib_problem
from .evaluators import (
    perform_experiments,
    optimize,
    SequentialEvaluator,
    Samplers,
)

from .ema_mpi import MPIEvaluator

from .optimization import (
    Convergence,
    EpsilonProgress,
    ArchiveLogger,
    HypervolumeMetric,
    EpsilonIndicatorMetric,
    GenerationalDistanceMetric,
    InvertedGenerationalDistanceMetric,
    SpacingMetric,
    epsilon_nondominated,
    rebuild_platypus_population,
    to_problem,
    to_robust_problem,
)
from .futures_ipyparallel import IpyparallelEvaluator
from .futures_multiprocessing import MultiprocessingEvaluator
from .futures_mpi import MPIEvaluator
from .outputspace_exploration import OutputSpaceExploration

try:
    from .evaluators import IpyparallelEvaluator
except ImportError:
    IpyparallelEvaluator = None
    warnings.warn("ipyparallel not available", ImportWarning)

del warnings
