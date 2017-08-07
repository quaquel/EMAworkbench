'''

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
import inspect
import os
import unittest

from ema_workbench.em_framework import (ModelEnsemble, RealParameter, 
                                        TimeSeriesOutcome)
from ema_workbench.connectors.pysd_connector import PysdModel

#TODO:: model classes should be tested for their pickleability prior to
# initialization 

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
        
        model = PysdModel(mdl_file=mdl_file)
        self.assertIsInstance(model, PysdModel)
 
    def test_parallel_experiment(self):
        """
        Test running an experiment in parallel
        Returns
        -------
  
        """
        relative_path_to_file = '../models/Teacup.mdl'
        directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        mdl_file = os.path.join(directory, relative_path_to_file)
         
        model = PysdModel(mdl_file=mdl_file)
         
        model.uncertainties = [RealParameter('Room Temperature', 33, 120)]
        model.outcomes = [TimeSeriesOutcome('Teacup Temperature')]
  
        ensemble = ModelEnsemble()  # instantiate an ensemble
        ensemble.model_structures = model  # set the model on the ensemble
        ensemble.parallel = True
        ensemble.perform_experiments(5)
  
    def test_multiple_models(self):
        """
        Test running running with two different pysd models
        Returns
        -------
  
        """
        relative_path_to_file = '../models/Sales_Agent_Market_Building_Dynamics.mdl'
        directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        mdl_file = os.path.join(directory, relative_path_to_file)
         
        market_model = PysdModel(mdl_file=mdl_file)
        market_model.uncertainties = [RealParameter('Startup Subsidy',0, 3),
                                      RealParameter('Startup Subsidy Length', 0, 10)]
        market_model.outcomes = [TimeSeriesOutcome('Still Employed')]
  
        relative_path_to_file = '../models/Sales_Agent_Motivation_Dynamics.mdl'
        directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        mdl_file = os.path.join(directory, relative_path_to_file)
  
        motivation_model = PysdModel(mdl_file=mdl_file)
        motivation_model.uncertainties = [RealParameter('Startup Subsidy', 0, 3),
                                      RealParameter('Startup Subsidy Length', 0, 10)]
        motivation_model.outcomes =[TimeSeriesOutcome('Still Employed')]
  
        ensemble = ModelEnsemble()  # instantiate an ensemble
        ensemble.model_structures = [market_model, motivation_model]  # set the model on the ensemble
        ensemble.perform_experiments(5)

if __name__ == '__main__':
    unittest.main()
