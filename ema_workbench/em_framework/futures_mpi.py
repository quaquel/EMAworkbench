import warnings


from .evaluators import BaseEvaluator, experiment_generator
from ..util import get_module_logger


__all__ = ["MPIEvaluator"]

_logger = get_module_logger(__name__)


def run_experiment_mpi(packed_data):
    from mpi4py.MPI import COMM_WORLD

    rank = COMM_WORLD.Get_rank()

    experiment, model_name, msis = packed_data
    _logger.debug(f"MPI Rank {rank}: starting {repr(experiment)}")

    models = NamedObjectMap(AbstractModel)
    models.extend(msis)
    experiment_runner = ExperimentRunner(models)

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

    def initialize(self):
        # Only import mpi4py if the MPIEvaluator is used, to avoid unnecessary dependencies.
        from mpi4py.futures import MPIPoolExecutor

        self._pool = MPIPoolExecutor(max_workers=self.n_processes)  # Removed initializer arguments
        _logger.info(f"MPI pool started with {self._pool._max_workers} workers")
        if self._pool._max_workers <= 10:
            _logger.warning(
                f"With only a few workers ({self._pool._max_workers}), the MPIEvaluator may be slower than the Sequential- or MultiprocessingEvaluator"
            )
        return self

    def finalize(self):
        self._pool.shutdown()
        _logger.info("MPI pool has been shut down")

    def evaluate_experiments(self, scenarios, policies, callback, combine="factorial"):
        ex_gen = experiment_generator(scenarios, self._msis, policies, combine=combine)
        experiments = list(ex_gen)

        packed = [(experiment, experiment.model_name, self._msis) for experiment in experiments]

        _logger.info(
            f"MPIEvaluator: Starting {len(packed)} experiments using MPI pool with {self._pool._max_workers} workers"
        )
        results = self._pool.map(run_experiment_mpi, packed)

        _logger.info(f"MPIEvaluator: Completed all {len(packed)} experiments")
        for experiment, outcomes in results:
            callback(experiment, outcomes)
        _logger.info(f"MPIEvaluator: Callback completed for all {len(packed)} experiments")
