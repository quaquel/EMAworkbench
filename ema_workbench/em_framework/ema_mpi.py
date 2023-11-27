import logging
import shutil

from .experiment_runner import ExperimentRunner
from .evaluators import BaseEvaluator, determine_rootdir
from .model import AbstractModel
from .points import experiment_generator
from .util import NamedObjectMap
from .ema_multiprocessing import setup_working_directories
from ..util import get_module_logger, ema_logging

_logger = get_module_logger(__name__)

__all__ = ["MPIEvaluator"]


class RankFilter(logging.Filter):
    """
    This is a filter which injects contextual information into the log.

    Rather than use actual contextual information, we just use random
    data in this demo.
    """

    def __init__(self, rank):
        self.rank = rank

    def filter(self, record):
        record.rank = self.rank
        return True


def mpi_initializer(models, log_level, root_dir):
    global experiment_runner

    from mpi4py.MPI import COMM_WORLD

    rank = COMM_WORLD.Get_rank()

    # setup the experiment runner
    msis = NamedObjectMap(AbstractModel)
    msis.extend(models)
    experiment_runner = ExperimentRunner(msis)

    # setup the logging
    logging.basicConfig(level=log_level, format="[%(rank)s/%(levelname)s] %(message)s")

    filter = RankFilter(rank)
    ema_logging.get_rootlogger().addFilter(filter)
    for _, mod_logger in ema_logging._module_loggers.items():
        mod_logger.addFilter(filter)

    # setup the working directories
    # make a root temp
    # copy each model directory
    tmpdir = setup_working_directories(models, root_dir)

    # register a cleanup finalizer function
    # remove the root temp
    # TODO this is tricky because of the muliple evaluator problem
    # if tmpdir:
    #     multiprocessing.util.Finalize(
    #         None, finalizer, args=(os.path.abspath(tmpdir),), exitpriority=10
    #     )


def run_experiment_mpi(experiment):
    from mpi4py.MPI import COMM_WORLD

    rank = COMM_WORLD.Get_rank()

    model_name = experiment.model_name
    _logger.debug(f"MPI Rank {rank}: starting {repr(experiment)}")

    outcomes = experiment_runner.run_experiment(experiment)

    _logger.debug(f"MPI Rank {rank}: completed {experiment}")

    return experiment, outcomes


class MPIEvaluator(BaseEvaluator):
    """Evaluator for experiments using MPI Pool Executor from mpi4py"""

    def __init__(self, msis, n_processes=None, **kwargs):
        super().__init__(msis, **kwargs)
        # warnings.warn(
        #     "The MPIEvaluator is experimental. Its interface and functionality might change in future releases.\n"
        #     "We welcome your feedback at: https://github.com/quaquel/EMAworkbench/discussions/311",
        #     FutureWarning,
        # )
        self._pool = None
        self.n_processes = n_processes
        self.rootdir = None

    def initialize(self):
        # Only import mpi4py if the MPIEvaluator is used, to avoid unnecessary dependencies.
        from mpi4py.futures import MPIPoolExecutor

        # fixme
        self.rootdir = determine_rootdir(self._msis)

        self._pool = MPIPoolExecutor(
            max_workers=self.n_processes,
            initializer=mpi_initializer,
            initargs=(self._msis, _logger.level, self.rootdir),
        )

        _logger.info(f"MPI pool started with {self._pool._max_workers} workers")
        # if self._pool._max_workers <= 10:
        #     _logger.warning(
        #         f"With only a few workers ({self._pool._max_workers}), the MPIEvaluator may be slower than the Sequential- or MultiprocessingEvaluator"
        #     )
        return self

    def finalize(self):
        if self._pool:
            self._pool.shutdown(wait=True)
            self._pool = None
            _logger.info("MPI pool has been shut down")

        # FIXME again this is a bit tricky because this means
        # FIXME you can use an evaluator only once if used through a context manager
        if self.rootdir:
            shutil.rmtree(self.rootdir)

    def evaluate_experiments(self, scenarios, policies, callback, combine="factorial"):
        ex_gen = experiment_generator(scenarios, self._msis, policies, combine=combine)
        experiments = list(ex_gen)

        # _logger.info(
        #     f"MPIEvaluator: Starting {len(packed)} experiments using MPI pool with {self._pool._max_workers} workers"
        # )
        results = self._pool.map(run_experiment_mpi, experiments)

        _logger.info(f"MPIEvaluator: Completed all {len(experiments)} experiments")
        for experiment, outcomes in results:
            callback(experiment, outcomes)
        _logger.info(f"MPIEvaluator: Callback completed for all {len(experiments)} experiments")
