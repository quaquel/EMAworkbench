import atexit
import logging
import os
import shutil
import warnings

from .evaluators import BaseEvaluator, experiment_generator
from .futures_util import setup_working_directories, finalizer, determine_rootdir
from .util import NamedObjectMap
from .model import AbstractModel
from .experiment_runner import ExperimentRunner
from ..util import get_module_logger, get_rootlogger

from ..util import ema_logging

__all__ = ["MPIEvaluator"]

_logger = get_module_logger(__name__)


experiment_runner = None
logcom = None
rank = None


class RankFilter(logging.Filter):
    """Filter for adding mpi rank to log message"""

    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        record.rank = self.rank
        return True


def mpi_initializer(models, log_level, root_dir):
    global experiment_runner
    global rank
    from mpi4py.MPI import COMM_WORLD

    rank = COMM_WORLD.Get_rank()

    # setup the experiment runner
    msis = NamedObjectMap(AbstractModel)
    msis.extend(models)
    experiment_runner = ExperimentRunner(msis)

    # setup the logging
    rootlogger = get_rootlogger()
    rootlogger.setLevel(30)

    formatter = logging.Formatter("[%(rank)s/%(levelname)s] %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    rootlogger.addHandler(handler)

    # can't we handle this at the handler before sending the message?
    logfilter = RankFilter(rank)
    rootlogger.addFilter(logfilter)
    for _, mod_logger in ema_logging._module_loggers.items():
        mod_logger.addFilter(logfilter)

    # setup the working directories
    tmpdir = setup_working_directories(models, root_dir)
    if tmpdir:
        atexit.register(finalizer, os.path.abspath(tmpdir))


def run_experiment_mpi(experiment):
    _logger.debug(f"MPI Rank {rank}: starting {repr(experiment)}")

    outcomes = experiment_runner.run_experiment(experiment)

    _logger.debug(f"MPI Rank {rank}: completed {experiment}")

    return experiment, outcomes


class MPIEvaluator(BaseEvaluator):
    """Evaluator for experiments using MPI Pool Executor from mpi4py"""

    def __init__(self, msis, n_processes=None, **kwargs):
        super().__init__(msis, **kwargs)
        warnings.warn(
            "The MPIEvaluator is experimental. Its interface and functionality might change in future releases.\n"
            "We welcome your feedback at: https://github.com/quaquel/EMAworkbench/discussions/311",
            FutureWarning,
        )
        self._pool = None
        self.n_processes = n_processes
        self.root_dir = None

    def initialize(self):
        # Only import mpi4py if the MPIEvaluator is used, to avoid unnecessary dependencies.
        from mpi4py.futures import MPIPoolExecutor

        self.root_dir = determine_rootdir(self._msis)

        self._pool = MPIPoolExecutor(
            max_workers=self.n_processes,
            initializer=mpi_initializer,
            initargs=(self._msis, _logger.level, self.root_dir),
        )  # Removed initializer arguments
        _logger.info(f"MPI pool started with {self._pool._max_workers} workers")
        if self._pool._max_workers <= 10:
            _logger.warning(
                f"With only a few workers ({self._pool._max_workers}), the MPIEvaluator may be slower than the Sequential- or MultiprocessingEvaluator"
            )
        return self

    def finalize(self):
        self._pool.shutdown()
        _logger.info("MPI pool has been shut down")

        if self.root_dir:
            shutil.rmtree(self.root_dir)

    def evaluate_experiments(self, scenarios, policies, callback, combine="factorial"):
        ex_gen = experiment_generator(scenarios, self._msis, policies, combine=combine)
        experiments = list(ex_gen)

        results = self._pool.map(run_experiment_mpi, experiments)
        for experiment, outcomes in results:
            callback(experiment, outcomes)

        _logger.info(f"MPIEvaluator: Completed all {len(experiments)} experiments")
