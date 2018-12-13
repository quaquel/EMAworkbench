from __future__ import (absolute_import)
from . import analysis
from . import em_framework
<<<<<<< HEAD

from .em_framework import (Model, RealParameter, CategoricalParameter, BooleanParameter,
=======
from .em_framework import (Model, RealParameter, CategoricalParameter, BinaryParameter,
>>>>>>> ab9dc14... introduce BinaryParameter
                           IntegerParameter, perform_experiments, optimize,
                           ScalarOutcome, TimeSeriesOutcome, Constant,
                           Scenario, Policy, MultiprocessingEvaluator,
                           IpyparallelEvaluator, SequentialEvaluator,
                           ReplicatorModel, Constraint)

from . import util
from .util import (save_results, load_results, ema_logging, EMAError,
                   experiments_to_scenarios)

__version__ = '1.3'
