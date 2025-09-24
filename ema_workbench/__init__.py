"""main namespace for workbench."""

__all__ = [
    "ArrayOutcome",
    "BooleanParameter",
    "CategoricalParameter",
    "Constant",
    "Constraint",
    "EMAError",
    "EpsilonIndicatorMetric",
    "GenerationalDistanceMetric",
    "HypervolumeMetric",
    "IntegerParameter",
    "InvertedGenerationalDistanceMetric",
    "IpyparallelEvaluator",
    "MPIEvaluator",
    "Model",
    "MultiprocessingEvaluator",
    "OutputSpaceExploration",
    "RealParameter",
    "ReplicatorModel",
    "Sample",
    "Samplers",
    "ScalarOutcome",
    "SequentialEvaluator",
    "SpacingMetric",
    "TimeSeriesOutcome",
    "em_framework",
    "ema_logging",
    "epsilon_nondominated",
    "load_results",
    "optimize",
    "perform_experiments",
    "process_replications",
    "save_results",
    "util",
    ]


from . import em_framework, util
from .em_framework import (
    ArrayOutcome,
    BooleanParameter,
    CategoricalParameter,
    Constant,
    Constraint,
    EpsilonIndicatorMetric,
    GenerationalDistanceMetric,
    HypervolumeMetric,
    IntegerParameter,
    InvertedGenerationalDistanceMetric,
    IpyparallelEvaluator,
    Model,
    MPIEvaluator,
    MultiprocessingEvaluator,
    OutputSpaceExploration,
    RealParameter,
    ReplicatorModel,
    Sample,
    Samplers,
    ScalarOutcome,
    SequentialEvaluator,
    SpacingMetric,
    TimeSeriesOutcome,
    epsilon_nondominated,
    optimize,
    perform_experiments,
)
from .util import (
    EMAError,
    ema_logging,
    load_results,
    process_replications,
    save_results,
)

# from . import analysis

__version__ = "3.0.0-dev"
