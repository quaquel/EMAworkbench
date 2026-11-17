"""Unit tests for the MPI evaluator."""

import logging
import platform
from unittest.mock import Mock

import pytest

import ema_workbench
from ema_workbench.em_framework import futures_mpi

# Check if mpi4py is installed and if we're on a Linux environment
try:
    # import the MPI submodule as well, because the tests patch mpi4py.MPI,
    # which only exists once it has been imported
    import mpi4py.MPI  # noqa: F401
except ImportError:
    MPI_AVAILABLE = False
else:
    MPI_AVAILABLE = True
CAN_TEST = (platform.system() == "Linux") or (platform.system() == "Darwin")


@pytest.mark.skipif(
    (not MPI_AVAILABLE) or (not CAN_TEST),
    reason="Test requires mpi4py installed and a Linux or Mac OS environment",
)
def test_mpi_evaluator(mocker):
    """Test mpi evaluator."""
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        pytest.fail(
            "mpi4py is not installed. It's required for this test. Install with: pip install mpi4py"
        )

    mocked_MPIPoolExecutor = mocker.patch(
        "mpi4py.futures.MPIPoolExecutor", autospec=True
    )
    mocker.patch(
        "ema_workbench.em_framework.futures_mpi.threading.Thread", autospec=True
    )
    mocker.patch(
        "ema_workbench.em_framework.futures_mpi.threading.Event", autospec=True
    )

    mocked_callback = mocker.patch(
        "ema_workbench.em_framework.evaluators.DefaultCallback",
    )
    mocked_generator = mocker.patch(
        "ema_workbench.em_framework.points.experiment_generator", autospec=True
    )
    mocked_generator.return_value = iter(
        [
            1,
        ]
    )

    model = Mock(spec=ema_workbench.Model)
    model.name = "test"

    # Create a mock experiment with the required attribute
    mock_experiment = Mock()
    mock_experiment.model_name = "test"
    mocked_generator.return_value = [
        mock_experiment,
    ]

    pool_mock = Mock()
    pool_mock.map.return_value = [(1, ({}, {}))]
    pool_mock._max_workers = 5  # Arbitrary number
    mocked_MPIPoolExecutor.return_value = pool_mock

    with futures_mpi.MPIEvaluator(model) as evaluator:
        evaluator.evaluate_experiments(mocked_generator([], [], []), mocked_callback)

        mocked_MPIPoolExecutor.assert_called()
        pool_mock.map.assert_called_once()

    # Check that pool shutdown was called
    pool_mock.shutdown.assert_called_once()


@pytest.mark.skipif(
    (not MPI_AVAILABLE) or (not CAN_TEST),
    reason="Test requires mpi4py installed and a Linux or Mac OS environment",
)
def test_logwatcher(mocker):
    """Test log watcher."""
    mocked_MPI = mocker.patch("mpi4py.MPI", autospec=True)
    mocked_MPI.COMM_WORLD = Mock()
    mocked_MPI.COMM_WORLD.Get_rank.return_value = 0

    # use a real logger instead of patching logging.getLogger, which is global
    # and is also used by pytest itself while the patch is active
    received = []

    class CollectingHandler(logging.Handler):
        def emit(self, record):
            received.append(record)

    worker_logger = logging.getLogger("EMA.worker_0")
    handler = CollectingHandler()
    worker_logger.addHandler(handler)

    mocked_MPI.INFO_NULL = None
    mocked_MPI.Open_port.return_value = "somestring"

    comm_mock = Mock()
    mocked_MPI.COMM_WORLD.Accept.return_value = comm_mock

    message = logging.LogRecord(
        "EMA.worker_0", logging.INFO, __file__, 0, "a message", None, None
    )
    comm_mock.recv.side_effect = [
        message,
    ]

    mocked_MPI.COMM_WORLD.bcast.return_value = True
    start_event = Mock()
    start_event.is_set.side_effect = [False, True]

    stop_event = Mock()
    stop_event.is_set.side_effect = [False, True]

    try:
        futures_mpi.logwatcher(start_event, stop_event)
    finally:
        worker_logger.removeHandler(handler)

    assert received == [message]


@pytest.mark.skipif(
    (not MPI_AVAILABLE) or (not CAN_TEST),
    reason="Test requires mpi4py installed and a Linux or Mac OS environment",
)
def test_mpi_initializer(mocker):
    """Test initializer of the MPI pool."""
    mocked_MPI = mocker.patch("mpi4py.MPI", autospec=True)

    mocker.patch(
        "ema_workbench.em_framework.points.experiment_generator", autospec=True
    )

    mocked_MPI.COMM_WORLD = Mock()
    mocked_MPI.COMM_WORLD.Get_rank.return_value = 0
    mocked_MPI.INFO_NULL = None
    mocked_MPI.Lookup_name.return_value = "somestring"

    comm_mock = Mock()
    mocked_MPI.COMM_WORLD.Connect.return_value = comm_mock

    handler_mock_instance = Mock(spec=ema_workbench.em_framework.futures_mpi.MPIHandler)
    handler_mock_instance.level = 0

    handler_mock = mocker.patch("ema_workbench.em_framework.futures_mpi.MPIHandler")
    handler_mock.return_value = handler_mock_instance

    model = Mock(spec=ema_workbench.Model)
    model.name = "test"
    models = [model]
    log_level = 10
    root_dir = None

    try:
        futures_mpi.mpi_initializer(models, log_level, root_dir)
    finally:
        # the initializer adds the handler to the root logger of the workbench
        ema_workbench.util.ema_logging.get_rootlogger().removeHandler(
            handler_mock_instance
        )

    # handler_mock.handle.assert_called()


@pytest.mark.skipif(
    (not MPI_AVAILABLE) or (not CAN_TEST),
    reason="Test requires mpi4py installed and a Linux or Mac OS environment",
)
def test_MPIHandler():
    """Test hander for logging in MPI."""
    communicator = Mock()

    handler = futures_mpi.MPIHandler(communicator)

    record = Mock(spec=logging.LogRecord)
    record.fmt = Mock(spec=logging.Formatter)
    record.fmt.format.return_value = "some text"
    record.exc_info = None
    record.exc_text = None
    record.stack_info = None

    handler.emit(record)
    communicator.send.assert_called_once()

    # communicator.send = Mock()
    # communicator.send.side_effect = Exception()
    # handler.emit(record)


@pytest.mark.skipif(
    (not MPI_AVAILABLE) or (not CAN_TEST),
    reason="Test requires mpi4py installed and a Linux or Mac OS environment",
)
def test_run_experiment_mpi(mocker):
    """Test running an experiment using MPI."""
    mocked_generator = mocker.patch(
        "ema_workbench.em_framework.futures_mpi.experiment_runner",
        autospec=True,
    )

    mock_experiment = Mock()
    mock_experiment.model_name = "test"

    futures_mpi.run_experiment_mpi(mock_experiment)

    mocked_generator.run_experiment.assert_called_once_with(mock_experiment)


@pytest.mark.skipif(
    (not MPI_AVAILABLE) or (not CAN_TEST),
    reason="Test requires mpi4py installed and a Linux or Mac OS environment",
)
def test_RankFilter():
    """Test RankFilter class."""
    rank = 1
    filter = futures_mpi.RankFilter(rank)

    record = Mock()
    filter.filter(record)

    assert record.rank == rank
