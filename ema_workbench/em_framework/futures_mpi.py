import atexit
import copy
import logging
import os
import shutil
import threading
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
    # global logcomm
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

    rootlogger = get_rootlogger()
    rootlogger.setLevel(log_level)

    formatter = logging.Formatter("[%(rank)s/%(levelname)s] %(message)s")
    handler = MPIHandler(logcomm)
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

    _logger.info(f"worker {rank} initialized")


def logwatcher():
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()

    info = MPI.INFO_NULL
    port = MPI.Open_port(info)
    _logger.debug(f"opened port: {port}")

    service = "logwatcher"
    MPI.Publish_name(service, info, port)
    _logger.debug(f"published service: {service}")

    root = 0
    _logger.debug("waiting for client connection...")
    comm = MPI.COMM_WORLD.Accept(port, info, root)
    _logger.debug("client connected...")

    while True:
        done = False
        if rank == root:
            message = comm.recv(None, MPI.ANY_SOURCE, tag=0)
            if message is None:
                done = True
            else:
                try:
                    print(f"{message.msg}")
                except Exception:
                    print("invalid expression: %s" % message)
        done = MPI.COMM_WORLD.bcast(done, root)
        if done:
            break


def run_experiment_mpi(experiment):
    _logger.debug(f"MPI Rank {rank}: starting {repr(experiment)}")

    outcomes = experiment_runner.run_experiment(experiment)

    _logger.debug(f"MPI Rank {rank}: completed {experiment}")

    return experiment, outcomes


class MPIHandler(logging.Handler):
    """
    This handler sends events from the worker process to the master process

    """

    def __init__(self, communicator):
        """
        Initialise an instance, using the passed queue.
        """
        logging.Handler.__init__(self)
        self.communicator = communicator

    def prepare(self, record):
        """

        Adapted from Queuehandler

        Prepares a record for queuing. The object returned by this method is
        enqueued.

        The base implementation formats the record to merge the message
        and arguments, and removes unpickleable items from the record
        in-place.

        You might want to override this method if you want to convert
        the record to a dict or JSON string, or send a modified copy
        of the record while leaving the original intact.
        """
        # The format operation gets traceback text into record.exc_text
        # (if there's exception data), and also returns the formatted
        # message. We can then use this to replace the original
        # msg + args, as these might be unpickleable. We also zap the
        # exc_info and exc_text attributes, as they are no longer
        # needed and, if not None, will typically not be pickleable.
        msg = self.format(record)
        record = copy.copy(record)
        record.message = msg
        record.msg = msg
        record.args = None
        record.exc_info = None
        record.exc_text = None
        return record

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
        self.n_processes = n_processes
        self.root_dir = None

    def initialize(self):
        # Only import mpi4py if the MPIEvaluator is used, to avoid unnecessary dependencies.
        from mpi4py.futures import MPIPoolExecutor

        self.logwatcher_thread = threading.Thread(name="logwatcher", target=logwatcher, daemon=True)
        self.logwatcher_thread.start()

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
