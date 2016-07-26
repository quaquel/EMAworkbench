
from __future__ import (division, absolute_import, unicode_literals, 
                        print_function)

import inspect
import os
import unittest

from ema_workbench.em_framework import ModelEnsemble
from ema_workbench.connectors import PySDConnector

class TestPySDConnector(unittest.TestCase):

    def test_connector_basic(self):
        """
        Test that the connector can instantiate the pysd interface object
        Returns
        -------

        """
        
        relative_path_to_file = '../models/Teacup.mdl'
        directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        mdl_file = os.path.join(directory, relative_path_to_file)
        
        
        model = PySDConnector(mdl_file)
        self.assertIsInstance(model, PySDConnector)

    def test_add_uncertainties(self):
        """
        Test actually running an experiment
        Returns
        -------

        """
        relative_path_to_file = '../models/Teacup.mdl'
        directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        mdl_file = os.path.join(directory, relative_path_to_file)
        
        model = PySDConnector(mdl_file,
                              uncertainties_dict={'Room Temperature': (33, 120)})

        ensemble = ModelEnsemble()  # instantiate an ensemble
        ensemble.model_structure = model  # set the model on the ensemble
        ensemble.parallel = False

        nr_runs = 10
        experiments, outcomes = ensemble.perform_experiments(nr_runs)

        self.assertEqual(experiments.shape[0], nr_runs)

    def test_add_outcomes(self):
        relative_path_to_file = '../models/Teacup.mdl'
        directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        mdl_file = os.path.join(directory, relative_path_to_file)
        
        model = PySDConnector(mdl_file,
                              uncertainties_dict={'Room Temperature': (33, 120)},
                              outcomes_list=['Teacup Temperature'])

        ensemble = ModelEnsemble()  # instantiate an ensemble
        ensemble.model_structure = model  # set the model on the ensemble
        ensemble.parallel = False

        nr_runs = 10
        experiments, outcomes = ensemble.perform_experiments(nr_runs)

        self.assertEqual(experiments.shape[0], nr_runs)
        self.assertIn('Teacup Temperature', outcomes.keys())

    def test_parallel_experiment(self):
        """
        Test running an experiment in parallel
        Returns
        -------

        """
        relative_path_to_file = '../models/Teacup.mdl'
        directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        mdl_file = os.path.join(directory, relative_path_to_file)

        model = PySDConnector(mdl_file,
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
        
        relative_path_to_file = '../models/Sales_Agent_Market_Building_Dynamics.mdl'
        directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        mdl_file = os.path.join(directory, relative_path_to_file)
        
        market_model = PySDConnector(mdl_file,
                                     uncertainties_dict={'Startup Subsidy': (0, 3),
                                                         'Startup Subsidy Length': (0, 10)},
                                     outcomes_list=['Still Employed'])

        relative_path_to_file = '../models/Sales_Agent_Market_Building_Dynamics.mdl'
        directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        mdl_file = os.path.join(directory, relative_path_to_file)

        motivation_model = PySDConnector(mdl_file,
                                         uncertainties_dict={'Startup Subsidy': (0, 3),
                                                             'Startup Subsidy Length': (0, 10)},
                                         outcomes_list=['Still Employed'])

        ensemble = ModelEnsemble()  # instantiate an ensemble
        ensemble.model_structures = [market_model, motivation_model]  # set the model on the ensemble
        results = ensemble.perform_experiments(cases=20)


if __name__ == '__main__':
    unittest.main()