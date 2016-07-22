
import unittest
from ema_workbench.em_framework import ModelEnsemble


class TestPySDConnector(unittest.TestCase):

    def test_connector_basic(self):
        """
        Test that the connector can instantiate the pysd interface object
        Returns
        -------

        """
        from ema_workbench.connectors import PySDConnector
        model = PySDConnector('../models/Teacup.mdl')
        self.assertIsInstance(model, PySDConnector)

    def test_add_uncertainties(self):
        """
        Test actually running an experiment
        Returns
        -------

        """
        from ema_workbench.connectors import PySDConnector
        model = PySDConnector('../models/Teacup.mdl',
                              uncertainties_dict={'Room Temperature': (33, 120)})

        ensemble = ModelEnsemble()  # instantiate an ensemble
        ensemble.model_structure = model  # set the model on the ensemble
        ensemble.parallel = False

        nr_runs = 10
        experiments, outcomes = ensemble.perform_experiments(nr_runs)

        self.assertEqual(experiments.shape[0], nr_runs)
        self.assertIn('TIME', outcomes.keys())



    def test_add_outcomes(self):
        from ema_workbench.connectors import PySDConnector
        model = PySDConnector('../models/Teacup.mdl',
                              uncertainties_dict={'Room Temperature': (33, 120)},
                              outcomes_list=['Teacup Temperature'])

        ensemble = ModelEnsemble()  # instantiate an ensemble
        ensemble.model_structure = model  # set the model on the ensemble
        ensemble.parallel = False

        nr_runs = 10
        experiments, outcomes = ensemble.perform_experiments(nr_runs)

        self.assertEqual(experiments.shape[0], nr_runs)
        self.assertIn('TIME', outcomes.keys())
        self.assertIn('Teacup Temperature', outcomes.keys())


    def test_parallel_experiment(self):
        """
        Test running an experiment in parallel
        Returns
        -------

        """
        from ema_workbench.connectors import PySDConnector
        model = PySDConnector('../models/Teacup.mdl',
                              uncertainties_dict={'Room Temperature': (33, 120)},
                              outcomes_list=['Teacup Temperature'])

        ensemble = ModelEnsemble()  # instantiate an ensemble
        ensemble.model_structure = model  # set the model on the ensemble
        ensemble.parallel = True
        results = ensemble.perform_experiments(cases=20)

    def test_multiple_models(self):
        """
        Test running running with two different pysd models
        Returns
        -------

        """
        from ema_workbench.connectors import PySDConnector
        market_model = PySDConnector('../models/Sales_Agent_Market_Building_Dynamics.mdl',
                                     uncertainties_dict={'Startup Subsidy': (0, 3),
                                                         'Startup Subsidy Length': (0, 10)},
                                     outcomes_list=['Still Employed'])

        motivation_model = PySDConnector('../models/Sales_Agent_Market_Building_Dynamics.mdl',
                                         uncertainties_dict={'Startup Subsidy': (0, 3),
                                                             'Startup Subsidy Length': (0, 10)},
                                         outcomes_list=['Still Employed'])

        ensemble = ModelEnsemble()  # instantiate an ensemble
        ensemble.model_structures = [market_model, motivation_model]  # set the model on the ensemble
        results = ensemble.perform_experiments(cases=20)
