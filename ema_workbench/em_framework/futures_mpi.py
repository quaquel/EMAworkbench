import atexit
import copy
import logging
import os
import shutil
import threading
import time
import warnings

from logging.handlers import QueueHandler

from .evaluators import BaseEvaluator, experiment_generator
from .futures_util import setup_working_directories, finalizer, determine_rootdir
from .util import NamedObjectMap
from .model import AbstractModel
from .experiment_runner import ExperimentRunner
from ..util import get_module_logger, get_rootlogger, method_logger

from ..util import ema_logging

__all__ = ["MPIEvaluator"]

_logger = get_module_logger(__name__)

experiment_runner = None


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
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()

    # setup the experiment runner
    msis = NamedObjectMap(AbstractModel)
    msis.extend(models)
    experiment_runner = ExperimentRunner(msis)

    # setup the logging
    info = MPI.INFO_NULL
    service = "logwatcher"
    port = MPI.Lookup_name(service)
    logcomm = MPI.COMM_WORLD.Connect(port, info, 0)

    root_logger = get_rootlogger()

    handler = MPIHandler(logcomm)
    handler.addFilter(RankFilter(rank))
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter("[worker %(rank)s/%(levelname)s] %(message)s"))
    root_logger.addHandler(handler)

    # setup the working directories
    tmpdir = setup_working_directories(models, root_dir)
    if tmpdir:
        atexit.register(finalizer(experiment_runner), os.path.abspath(tmpdir))

    # _logger.info(f"worker {rank} initialized")
    root_logger.info(f"worker {rank} initialized")


def logwatcher(start_event, stop_event):
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()

    info = MPI.INFO_NULL
    port = MPI.Open_port(info)
    # print(f"client: {rank} {port}")
    _logger.debug(f"opened port: {port}")

    service = "logwatcher"
    MPI.Publish_name(service, info, port)
    _logger.debug(f"published service: {service}")
    start_event.set()

    root = 0
    _logger.debug("waiting for client connection...")
    comm = MPI.COMM_WORLD.Accept(port, info, root)
    _logger.debug("client connected...")

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
    _logger.debug(f"starting {experiment.experiment_id}")

    outcomes = experiment_runner.run_experiment(experiment)

    _logger.debug(f"completed {experiment.experiment_id}")

    return experiment, outcomes


def send_sentinel():
    record = logging.makeLogRecord(dict(level=logging.CRITICAL, msg=None, name=42))

    for handler in get_rootlogger().handlers:
        if isinstance(handler, MPIHandler):
            _logger.debug("sending sentinel")
            handler.communicator.send(record, 0, 0)


class MPIHandler(QueueHandler):
    """
    This handler sends events from the worker process to the master process

    """

    def __init__(self, communicator):
        """
        Initialise an instance, using the passed queue.
        """
        logging.Handler.__init__(self)
        self.communicator = communicator

    def emit(self, record):
        """
        Emit a record.

        Writes the LogRecord to the queue, preparing it for pickling first.
        """
        record = self.prepare(record)
        try:
            self.communicator.send(record, 0, 0)
        except Exception:
            self.handleError(record)


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
        self.root_dir = None
        self.stop_event = None
        self.n_processes = n_processes

    @method_logger(__name__)
    def initialize(self):
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
        # submit sentinel
        self.stop_event.set()
        self._pool.submit(send_sentinel)
        self._pool.shutdown()
        self.logwatcher_thread.join(timeout=60)

        if self.logwatcher_thread.is_alive():
            _logger.warning(f"houston we have a problem")

        if self.root_dir:
            shutil.rmtree(self.root_dir)

        time.sleep(0.1)
        _logger.info("MPI pool has been shut down")

    @method_logger(__name__)
    def evaluate_experiments(self, scenarios, policies, callback, combine="factorial", **kwargs):
        experiments = list(experiment_generator(scenarios, self._msis, policies, combine=combine))

        results = self._pool.map(run_experiment_mpi, experiments, **kwargs)
        for experiment, outcomes in results:
            callback(experiment, outcomes)

        _logger.info(f"MPIEvaluator: Completed all {len(experiments)} experiments")
