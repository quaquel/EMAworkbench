'''
Created on Aug 11, 2015

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
from ema_workbench.em_framework.util import NamedObjectMap

try:
    import unittest.mock as mock
except ImportError:
    import mock
import unittest

from ema_workbench.em_framework.experiment_runner import ExperimentRunner
from ema_workbench.em_framework.model import Model, AbstractModel
from ema_workbench.util import EMAError, CaseError
from ema_workbench.em_framework.parameters import Policy, Experiment, Scenario

class ExperimentRunnerTestCase(unittest.TestCase):
    
    def test_init(self):
        mockMSI = mock.Mock(spec=Model)
        mockMSI.name = 'test'
        msis = {'test':mockMSI}

        runner = ExperimentRunner(msis)
        
        self.assertEqual(msis, runner.msis)
    
    
    def test_run_experiment(self):
        mockMSI = mock.Mock(spec=Model)
        mockMSI.name = 'test'
        
        msis = NamedObjectMap(AbstractModel)
        msis['test'] = mockMSI

        runner = ExperimentRunner(msis)
        
        experiment = Experiment('test',
                                mockMSI,
                                Policy('none'),  
                                Scenario(a=1, b=2), 0)
        
        runner.run_experiment(experiment)
        
        sc, p = mockMSI.run_model.call_args[0]
        self.assertEqual(sc.name, experiment.scenario.name)
        self.assertEqual(p, experiment.policy)
        
        mockMSI.reset_model.assert_called_once_with()
        
   
        # assert handling of case error
        mockMSI = mock.Mock(spec=Model)
        mockMSI.name = 'test'
        mockMSI.run_model.side_effect = Exception('some exception')
        msis = NamedObjectMap(AbstractModel)
        msis['test'] = mockMSI
        
        runner = ExperimentRunner(msis)
    
        experiment = Experiment('test',mockMSI,Policy('none'),  
                      Scenario(a=1, b=2),0)

        with self.assertRaises(EMAError):
            runner.run_experiment(experiment)
           
        # assert handling of case error
        mockMSI = mock.Mock(spec=Model)
        mockMSI.name = 'test'
        mockMSI.run_model.side_effect = CaseError("message", {})
        msis = NamedObjectMap(AbstractModel)
        msis['test'] = mockMSI
        runner = ExperimentRunner(msis)
    
        experiment = Experiment('test',mockMSI,Policy('none'),  
                      Scenario(a=1, b=2),0)

        runner.run_experiment(experiment)
        
if __name__ == "__main__":
    unittest.main()
