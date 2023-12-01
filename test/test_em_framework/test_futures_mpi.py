import unittest.mock as mock
import unittest
import platform

import ema_workbench
from ema_workbench.em_framework import futures_mpi


class TestMPIEvaluator(unittest.TestCase):
    # Check if mpi4py is installed and if we're on a Linux environment
    try:
        import mpi4py

        MPI_AVAILABLE = True
    except ImportError:
        MPI_AVAILABLE = False
    IS_LINUX = platform.system() == "Linux"

    @unittest.skipUnless(
        MPI_AVAILABLE and IS_LINUX, "Test requires mpi4py installed and a Linux environment"
    )
    @mock.patch("mpi4py.futures.MPIPoolExecutor")
    @mock.patch("ema_workbench.em_framework.evaluators.DefaultCallback")
    @mock.patch("ema_workbench.em_framework.evaluators.experiment_generator")
    def test_mpi_evaluator(self, mocked_generator, mocked_callback, mocked_MPIPoolExecutor):
        try:
            import mpi4py
        except ImportError:
            self.fail(
                "mpi4py is not installed. It's required for this test. Install with: pip install mpi4py"
            )

        model = mock.Mock(spec=ema_workbench.Model)
        model.name = "test"

        # Create a mock experiment with the required attribute
        mock_experiment = mock.Mock()
        mock_experiment.model_name = "test"
        mocked_generator.return_value = [mock_experiment]

        pool_mock = mock.Mock()
        pool_mock.map.return_value = [(1, ({}, {}))]
        pool_mock._max_workers = 5  # Arbitrary number
        mocked_MPIPoolExecutor.return_value = pool_mock

        with futures_mpi.MPIEvaluator(model) as evaluator:
            evaluator.evaluate_experiments(10, 10, mocked_callback)

            mocked_MPIPoolExecutor.assert_called_once()
            pool_mock.map.assert_called_once()

        # Check that pool shutdown was called
        pool_mock.shutdown.assert_called_once()
