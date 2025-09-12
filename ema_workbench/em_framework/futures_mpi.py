"""Support for MPI evaluators."""

import atexit
import logging
import os
import shutil
import threading
import time
import warnings
from collections.abc import Callable, Iterable
from logging.handlers import QueueHandler

from ..util import get_module_logger, get_rootlogger, method_logger
from .evaluators import BaseEvaluator
from .experiment_runner import ExperimentRunner
from .futures_util import determine_rootdir, finalizer, setup_working_directories
from .model import AbstractModel
from .points import Experiment
from .util import NamedObjectMap

__all__ = ["MPIEvaluator"]

_logger = get_module_logger(__name__)

experiment_runner = None


class RankFilter(logging.Filter):
    """Filter for adding mpi rank to log message."""

    def __init__(self, rank):
        """Init."""
        super().__init__()
        self.rank = rank

    def filter(self, record):
        """Filter records."""
        record.rank = self.rank
        return True


def mpi_initializer(models, log_level, root_dir):
    """Initalizer."""
    global experiment_runner # noqa: PLW0603
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()

    # setup the experiment runner
    msis = NamedObjectMap(AbstractModel)
    msis.extend(models)
    experiment_runner = ExperimentRunner(msis)

    # setup the working directories
    tmpdir = setup_working_directories(models, root_dir)
    if tmpdir:
        atexit.register(finalizer(experiment_runner), os.path.abspath(tmpdir))

    # setup the logging
    service = "logwatcher"
    port = MPI.Lookup_name(service)
    logcomm = MPI.COMM_WORLD.Connect(port)

    root_logger = get_rootlogger()

    handler = MPIHandler(logcomm)
    handler.addFilter(RankFilter(rank))
    handler.setLevel(log_level)
    handler.setFormatter(
        logging.Formatter("[worker %(rank)s/%(levelname)s] %(message)s")
    )
    root_logger.addHandler(handler)
    _logger.info(f"worker {rank} initialized")


def logwatcher(start_event, stop_event):
    """Logwatcher function."""
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()

    info = MPI.INFO_NULL
    port = MPI.Open_port(info)
    _logger.debug(f"opened port: {port}")

    service = "logwatcher"
    MPI.Publish_name(service, port)
    _logger.info(f"published service: {service}")
    start_event.set()

    root = 0
    _logger.info("waiting for client connections...")
    comm = MPI.COMM_WORLD.Accept(port)
    _logger.info("clients connected...")

    while not stop_event.is_set():
        if rank == root:
            record = comm.recv(None, MPI.ANY_SOURCE, tag=0)
            try:
                logger = logging.getLogger(record.name)
            except Exception as e:
                if record.msg is None:
                    _logger.debug("received sentinel")
                    break
                else:
                    # AttributeError if record does not have a name attribute
                    # TypeError record.name is not a string
                    raise e
            else:
                logger.callHandlers(record)

    _logger.info("closing logwatcher")


def run_experiment_mpi(experiment):
    """Run a single experiment."""
    _logger.debug(f"starting {experiment.experiment_id}")

    outcomes = experiment_runner.run_experiment(experiment)

    _logger.debug(f"completed {experiment.experiment_id}")

    return experiment, outcomes


def send_sentinel():
    """Send sentinel to ensure logging is closed down."""
    record = logging.makeLogRecord({"level":logging.CRITICAL, "msg":None, "name":42})

    for handler in get_rootlogger().handlers:
        if isinstance(handler, MPIHandler):
            _logger.debug("sending sentinel")
            handler.communicator.send(record, 0, 0)


class MPIHandler(QueueHandler):
    """Handler that sends events from the worker process to the master process."""

    def __init__(self, communicator):
        """Initialise an instance, using the passed queue."""
        logging.Handler.__init__(self)
        self.communicator = communicator

    def emit(self, record):
        """Emit a record.

        Writes the LogRecord to the queue, preparing it for pickling first.
        """
        record = self.prepare(record)
        try:
            self.communicator.send(record, 0, 0)
        except Exception:
            self.handleError(record)


class MPIEvaluator(BaseEvaluator):
    """Evaluator for experiments using MPI Pool Executor from mpi4py."""

    def __init__(self, msis, n_processes=None, **kwargs):
        """Init."""
        super().__init__(msis, **kwargs)
        warnings.warn(
            "The MPIEvaluator is experimental. Its interface and functionality might change in future releases.\n"
            "We welcome your feedback at: https://github.com/quaquel/EMAworkbench/discussions/311",
            FutureWarning, stacklevel=2,
        )
        self._pool = None
        self.root_dir = None
        self.stop_event = None
        self.n_processes = n_processes

    @method_logger(__name__)
    def initialize(self):
        """Initialize the MPI pool."""
        # Only import mpi4py if the MPIEvaluator is used, to avoid unnecessary dependencies.
        from mpi4py.futures import MPIPoolExecutor

        start_event = threading.Event()
        self.stop_event = threading.Event()
        self.logwatcher_thread = threading.Thread(
            name="logwatcher",
            target=logwatcher,
            daemon=False,
            args=(
                start_event,
                self.stop_event,
            ),
        )
        self.logwatcher_thread.start()
        start_event.wait()
        _logger.info("logwatcher server started")

        self.root_dir = determine_rootdir(self._msis)
        self._pool = MPIPoolExecutor(
            max_workers=self.n_processes,
            initializer=mpi_initializer,
            initargs=(self._msis, _logger.level, self.root_dir),
        )

        _logger.info(f"MPI pool started with {self._pool._max_workers} workers")
        if self._pool._max_workers <= 10:
            _logger.warning(
                f"With only a few workers ({self._pool._max_workers}), the MPIEvaluator may be slower than the Sequential- or MultiprocessingEvaluator"
            )
        return self

    @method_logger(__name__)
    def finalize(self):
        """Finalize the MPIPoolExecutor."""
        # submit sentinel
        self.stop_event.set()
        self._pool.submit(send_sentinel)
        self._pool.shutdown()
        self.logwatcher_thread.join(timeout=60)

        if self.logwatcher_thread.is_alive():
            _logger.warning("houston we have a problem, logwatcher is still alive")

        if self.root_dir:
            shutil.rmtree(self.root_dir)

        time.sleep(0.1)
        _logger.info("MPI pool has been shut down")

    @method_logger(__name__)
    def evaluate_experiments(
        self, experiments:Iterable[Experiment], callback:Callable, **kwargs
    ):
        """Evaluate experiments using MPIPoolExecutor."""
        experiments = list(experiments)

        results = self._pool.map(run_experiment_mpi, experiments, **kwargs)
        for experiment, outcomes in results:
            callback(experiment, outcomes)

        _logger.info(f"MPIEvaluator: Completed all {len(experiments)} experiments")
