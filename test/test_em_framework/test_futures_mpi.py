import platform
import pytest

from unittest.mock import Mock

import ema_workbench
from ema_workbench.em_framework import futures_mpi

# Check if mpi4py is installed and if we're on a Linux environment
try:
    import mpi4py
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
    try:
        import mpi4py
    except ImportError:
        pytest.fail(
            "mpi4py is not installed. It's required for this test. Install with: pip install mpi4py"
        )

    mocked_MPIPoolExecutor = mocker.patch("mpi4py.futures.MPIPoolExecutor", autospec=True)
    mocker.patch("ema_workbench.em_framework.futures_mpi.threading.Thread", autospec=True)
    mocked_callback = mocker.patch(
        "ema_workbench.em_framework.evaluators.DefaultCallback",
    )
    mocked_generator = mocker.patch(
        "ema_workbench.em_framework.futures_mpi.experiment_generator",
        autospec=True,
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
        evaluator.evaluate_experiments(10, 10, mocked_callback)

        mocked_MPIPoolExecutor.assert_called_once()
        pool_mock.map.assert_called_once()

    # Check that pool shutdown was called
    pool_mock.shutdown.assert_called_once()


def test_logwatcher(mocker):
    mocked_MPI = mocker.patch("mpi4py.MPI", autospec=True)
    mocked_MPI.COMM_WORLD = Mock()
    mocked_MPI.COMM_WORLD.Get_rank.return_value = 0

    mocked_get_logger = mocker.patch("logging.getLogger", autospec=True)
    mocked_logger = Mock()
    mocked_get_logger.return_value = mocked_logger

    mocked_MPI.INFO_NULL = None
    mocked_MPI.Open_port.return_value = "somestring"

    comm_mock = Mock()
    mocked_MPI.COMM_WORLD.Accept.return_value = comm_mock

    message = Mock()
    message.name = "EMA.worker_0"
    comm_mock.recv.side_effect = [message, None]

    mocked_MPI.COMM_WORLD.bcast.return_value = True
    futures_mpi.logwatcher()

    mocked_get_logger.assert_called_once_with(message.name)
    mocked_logger.callHandlers.assert_called_once_with(message)


@pytest.mark.skipif(
    (not MPI_AVAILABLE) or (not CAN_TEST),
    reason="Test requires mpi4py installed and a Linux or Mac OS environment",
)
def test_mpi_initializer(mocker):
    mocked_MPI = mocker.patch("mpi4py.MPI", autospec=True)
    mocked_generator = mocker.patch(
        "ema_workbench.em_framework.futures_mpi.experiment_generator",
        autospec=True,
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

    futures_mpi.mpi_initializer(models, log_level, root_dir)

    # handler_mock.handle.assert_called()
