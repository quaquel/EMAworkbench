'''


'''
from __future__ import (unicode_literals, print_function, absolute_import,
                        division)

import mock
import unittest

import ema_workbench
from ema_workbench.em_framework import evaluators
import ipyparallel

# Created on 14 Mar 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

class TestEvaluators(unittest.TestCase):

    @mock.patch('ema_workbench.em_framework.evaluators.DefaultCallback')
    @mock.patch('ema_workbench.em_framework.evaluators.experiment_generator')
    @mock.patch('ema_workbench.em_framework.evaluators.ExperimentRunner')
    def test_sequential_evalutor(self, mocked_runner, mocked_generator,
                                 mocked_callback):
        model = mock.Mock(spec=ema_workbench.Model)
        model.name = "test"
        mocked_generator.return_value = [1]
        mocked_runner.return_value = mocked_runner # return the mock upon initialization
        
        evaluator = evaluators.SequentialEvaluator(model)
        evaluator.evaluate_experiments(10, 10, mocked_callback)
        
        mocked_runner.assert_called_once() # test initalization of runner
        mocked_runner.run_experiment.assert_called_once_with(1)

        
#         searchover
#         union
        
    @mock.patch('ema_workbench.em_framework.evaluators.multiprocessing')   
    @mock.patch('ema_workbench.em_framework.evaluators.DefaultCallback')
    @mock.patch('ema_workbench.em_framework.evaluators.experiment_generator')
    @mock.patch('ema_workbench.em_framework.evaluators.add_tasks')
    def test_multiprocessing_evaluator(self, mocked_add_task, mocked_generator,
                                 mocked_callback, mocked_multiprocessing):
        model = mock.Mock(spec=ema_workbench.Model)
        model.name = "test"
        mocked_generator.return_value = [1]
        
        with evaluators.MultiprocessingEvaluator(model, 2) as evaluator:
            evaluator.evaluate_experiments(10, 10, mocked_callback)
        
            mocked_add_task.assert_called_once()

    @mock.patch('ema_workbench.em_framework.evaluators.set_engine_logger')
    @mock.patch('ema_workbench.em_framework.evaluators.initialize_engines')
    @mock.patch('ema_workbench.em_framework.evaluators.start_logwatcher')
    @mock.patch('ema_workbench.em_framework.evaluators.DefaultCallback')
    @mock.patch('ema_workbench.em_framework.evaluators.experiment_generator')
    def test_ipyparallel_evaluator(self, mocked_generator, mocked_callback,
                                   mocked_start, mocked_initialize, mocked_set):
        model = mock.Mock(spec=ema_workbench.Model)
        model.name = "test"
        mocked_generator.return_value = [1]
        mocked_start.return_value = mocked_start, None
        
        client = mock.MagicMock(spec=ipyparallel.Client)
        lb_view = mock.Mock()
        lb_view.map.return_value = [(1,1)]
        
        client.load_balanced_view.return_value = lb_view 
        
        with evaluators.IpyparallelEvaluator(model, client) as evaluator:
            evaluator.evaluate_experiments(10, 10, mocked_callback)
            lb_view.map.called_once()
    
    def test_perform_experiments(self):
        pass

if __name__ == '__main__':
    unittest.main()