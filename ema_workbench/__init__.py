from . import em_framework
from .em_framework import (Model, RealParameter, CategoricalParameter,
                           BooleanParameter,
                           IntegerParameter, perform_experiments, optimize,
                           ScalarOutcome, TimeSeriesOutcome, Constant,
                           Scenario, Policy, MultiprocessingEvaluator,
                           IpyparallelEvaluator, SequentialEvaluator,
                           ReplicatorModel, Constraint, ArrayOutcome,
                           Samplers)

from . import util
from .util import (save_results, load_results, ema_logging, EMAError,
                   process_replications)

# from . import analysis

__version__ = '2.1'
