"""


"""
import unittest.mock as mock
import unittest
import platform

import ema_workbench
from ema_workbench.em_framework import evaluators
import ipyparallel

# Created on 14 Mar 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


class TestEvaluators(unittest.TestCase):
    @mock.patch("ema_workbench.em_framework.evaluators.DefaultCallback")
    @mock.patch("ema_workbench.em_framework.evaluators.experiment_generator")
    @mock.patch("ema_workbench.em_framework.evaluators.ExperimentRunner")
    def test_sequential_evalutor(self, mocked_runner, mocked_generator, mocked_callback):
        model = mock.Mock(spec=ema_workbench.Model)
        model.name = "test"
        mocked_generator.return_value = [1]
        mocked_runner.return_value = mocked_runner  # return the mock upon initialization
        mocked_runner.run_experiment.return_value = {}, {}

        evaluator = evaluators.SequentialEvaluator(model)
        evaluator.evaluate_experiments(10, 10, mocked_callback)

        mocked_runner.assert_called_once()  # test initialization of runner
        mocked_runner.run_experiment.assert_called_once_with(1)

    #         searchover
    #         union

    @mock.patch("ema_workbench.em_framework.evaluators.multiprocessing")
    @mock.patch("ema_workbench.em_framework.evaluators.DefaultCallback")
    @mock.patch("ema_workbench.em_framework.evaluators.experiment_generator")
    @mock.patch("ema_workbench.em_framework.evaluators.add_tasks")
    def test_multiprocessing_evaluator(
        self, mocked_add_task, mocked_generator, mocked_callback, mocked_multiprocessing
    ):
        model = mock.Mock(spec=ema_workbench.Model)
        model.name = "test"
        mocked_generator.return_value = [1]
        mocked_multiprocessing.cpu_count.return_value = 4

        with evaluators.MultiprocessingEvaluator(model, 2) as evaluator:
            evaluator.evaluate_experiments(10, 10, mocked_callback)

            mocked_add_task.assert_called_once()

    @mock.patch("ema_workbench.em_framework.evaluators.set_engine_logger")
    @mock.patch("ema_workbench.em_framework.evaluators.initialize_engines")
    @mock.patch("ema_workbench.em_framework.evaluators.start_logwatcher")
    @mock.patch("ema_workbench.em_framework.evaluators.DefaultCallback")
    @mock.patch("ema_workbench.em_framework.evaluators.experiment_generator")
    def test_ipyparallel_evaluator(
        self, mocked_generator, mocked_callback, mocked_start, mocked_initialize, mocked_set
    ):
        model = mock.Mock(spec=ema_workbench.Model)
        model.name = "test"
        mocked_generator.return_value = [1]
        mocked_start.return_value = mocked_start, None

        client = mock.MagicMock(spec=ipyparallel.Client)
        lb_view = mock.Mock()
        lb_view.map.return_value = [(1, ({}, {}))]

        client.load_balanced_view.return_value = lb_view

        with evaluators.IpyparallelEvaluator(model, client) as evaluator:
            evaluator.evaluate_experiments(10, 10, mocked_callback)
            lb_view.map.called_once()

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

        with evaluators.MPIEvaluator(model) as evaluator:
            evaluator.evaluate_experiments(10, 10, mocked_callback)

            mocked_MPIPoolExecutor.assert_called_once()
            pool_mock.map.assert_called_once()

        # Check that pool shutdown was called
        pool_mock.shutdown.assert_called_once()

    def test_perform_experiments(self):
        pass


if __name__ == "__main__":
    unittest.main()
