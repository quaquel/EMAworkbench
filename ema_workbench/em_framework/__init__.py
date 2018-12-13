from __future__ import (absolute_import, unicode_literals, division,
                        print_function)
import warnings
__all__ = ["ema_parallel", "parameters"
           "model", "outcomes", "samplers",
           "Model", 'FileModel', "ModelEnsemble",
           "Outcome", "ScalarOutcome", "TimeSeriesOutcome", "Constraint",
<<<<<<< HEAD
           "RealParameter", "IntegerParameter", "CategoricalParameter", "BooleanParameter",
=======
           "RealParameter", "IntegerParameter", "CategoricalParameter", "BinaryParameter",
>>>>>>> ab9dc14... introduce BinaryParameter
           "Scenario", "Policy", "Experiment", "Constant", "create_parameters",
           "parameters_to_csv", "Category", "SobolSampler", "MorrisSampler",
           "get_SALib_problem", "FASTSampler",
           "peform_experiments", 'optimize', "IpyparallelEvaluator",
           "MultiprocessingEvaluator", "SequentialEvaluator"
           'ReplicatorModel', "EpsilonProgress", "HyperVolume",
           "Convergence", "ArchiveLogger"]

from .outcomes import ScalarOutcome, TimeSeriesOutcome, Outcome, Constraint
from .model import Model, FileModel, ReplicatorModel
<<<<<<< HEAD

from .parameters import (RealParameter, IntegerParameter, CategoricalParameter, BooleanParameter,
=======
from .parameters import (RealParameter, IntegerParameter, CategoricalParameter, BinaryParameter,
>>>>>>> ab9dc14... introduce BinaryParameter
                         Scenario, Policy, Constant, Experiment, create_parameters,
                         parameters_to_csv, Category, experiment_generator)
from .samplers import (MonteCarloSampler, FullFactorialSampler, LHSSampler,
                       PartialFactorialSampler, sample_levers,
                       sample_uncertainties)
from .salib_samplers import (SobolSampler, MorrisSampler, FASTSampler,
                             get_SALib_problem)
from .evaluators import (perform_experiments, optimize,
                         MultiprocessingEvaluator, SequentialEvaluator)
from .optimization import (Convergence, HyperVolume, EpsilonProgress,
                           ArchiveLogger)


try:
    from .evaluators import IpyparallelEvaluator
except ImportError:
    IpyparallelEvaluator = None
    warnings.warn("ipyparallel not available", ImportWarning)

del warnings