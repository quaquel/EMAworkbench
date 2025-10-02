"""Unit tests for experiment runner."""

# Created on Aug 11, 2015
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


import unittest
import unittest.mock as mock

from ema_workbench.em_framework.experiment_runner import ExperimentRunner
from ema_workbench.em_framework.model import AbstractModel, Model
from ema_workbench.em_framework.parameters import RealParameter
from ema_workbench.em_framework.points import Experiment, Sample
from ema_workbench.em_framework.util import NamedObjectMap
from ema_workbench.util import ExperimentError, EMAError


class ExperimentRunnerTestCase(unittest.TestCase):
    def test_init(self):
        mockMSI = mock.Mock(spec=Model)
        mockMSI.name = "test"
        runner = ExperimentRunner([mockMSI])

        self.assertEqual(mockMSI, runner.msis[mockMSI.name])

    def test_run_experiment(self):
        mock_msi = mock.Mock(spec=Model)
        mock_msi.name = "test"
        mock_msi.uncertainties = [RealParameter("a", 0, 10), RealParameter("b", 0, 10)]
        mock_msi.constants = []

        runner = ExperimentRunner([mock_msi])

        experiment = Experiment(
            "test", mock_msi.name, Sample("none"), Sample(a=1, b=2), 0
        )

        runner.run_experiment(experiment)

        sc, p, c = mock_msi.run_model.call_args[0]
        self.assertEqual(sc.name, experiment.scenario.name)
        self.assertEqual(p, experiment.policy)

        mock_msi.reset_model.assert_called_once_with()

        # assert handling of case error
        mock_msi = mock.Mock(spec=Model)
        mock_msi.name = "test"
        mock_msi.run_model.side_effect = Exception("some exception")
        mock_msi.constants = []
        msis = NamedObjectMap(AbstractModel)
        msis["test"] = mock_msi

        runner = ExperimentRunner(msis)

        experiment = Experiment(
            "test", mock_msi.name, Sample("none"), Sample(a=1, b=2), 0
        )

        with self.assertRaises(EMAError):
            runner.run_experiment(experiment)

        # assert handling of case error
        mock_msi = mock.Mock(spec=Model)
        mock_msi.name = "test"
        mock_msi.run_model.side_effect = ExperimentError("message", {})
        msis = NamedObjectMap(AbstractModel)
        msis["test"] = mock_msi
        mock_msi.constants = []
        runner = ExperimentRunner(msis)

        experiment = Experiment(
            "test", mock_msi.name, Sample("none"), Sample(a=1, b=2), 0
        )
        runner.run_experiment(experiment)


if __name__ == "__main__":
    unittest.main()
