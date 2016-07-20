'''
Created on 28 sep. 2011

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import multiprocessing

try:
    import unittest.mock as mock
except ImportError:
    import mock
import unittest
import time
import threading

<<<<<<< HEAD:src/ema_workbench/test/test_em_framework/test_ema_parallel_multiprocessing.py
from ...em_framework import (Model, ema_parallel_multiprocessing)
from ...util import EMAError
=======
from ema_workbench.em_framework import (ModelStructureInterface,
                                        ema_parallel_multiprocessing)
from ema_workbench.util import EMAError
>>>>>>> master:test/test_em_framework/test_ema_parallel_multiprocessing.py

class MockMSI(Model):

    def run_model(self, case):
        Model.run_model(self, case)

    def model_init(self, policy, kwargs):
        Model.model_init(self, policy, kwargs)
        

class ParallelMultiprocessingPoolTestCase(unittest.TestCase):
    
    @mock.patch('ema_workbench.em_framework.ema_parallel_multiprocessing.os')
    @mock.patch('ema_workbench.em_framework.ema_parallel_multiprocessing.shutil')
    @mock.patch.object(ema_parallel_multiprocessing.CalculatorPool, '_get_worker_name')
    def test_init_normal(self, mock_get_worker_name, mock_shutil, mock_os):
        
        mockMSI = mock.Mock(spec=MockMSI)
        mockMSI.name = 'test'

        # set some proper return values on mocked methods and functions
        mock_get_worker_name.return_value = "workername"
        mockMSI.working_directory = '.'
        mock_os.path.abspath.return_value = '/Domain/model'
        mock_os.path.dirname.return_value = '/Domain'
        
        # instantiate the pool
        pool = ema_parallel_multiprocessing.CalculatorPool([mockMSI], 
                                                           processes=2)
        
        # assert whether the init is functioning correctly
        self.assertEqual(len(pool._pool), 2, "nr. processes not correct")
        self.assertEqual(mock_os.path.dirname.call_count, 1,
                         "os.dirname called too frequent")
        
        mock_os.path.join.assert_called_with("/Domain", "workernametest")
        
        mock_os.reset_mock()

        # instantiate the pool
        pool = ema_parallel_multiprocessing.CalculatorPool([mockMSI], 
                                                           processes=None)
         
        # assert whether the init is functioning correctly
        self.assertGreater(len(pool._pool), 0)
        self.assertEqual(mock_os.path.dirname.call_count, 1,
                         "os.dirname called too frequent")
         
        mock_os.path.join.assert_called_with("/Domain", "workernametest")

class ParallelMultiprocessingLogProcess(unittest.TestCase):
    pass

class ParallelMultiprocessingSubProcessLogHandler(unittest.TestCase):
    pass

class WorkerTestCase(unittest.TestCase):
    
    @mock.patch('ema_workbench.em_framework.ema_parallel_multiprocessing.ExperimentRunner')
    @mock.patch('ema_workbench.em_framework.ema_parallel_multiprocessing.ema_logging')
    def test_worker(self, mocked_logging, mocked_runner):
        mocked_inqueue = mock.Mock(multiprocessing.queues.SimpleQueue())
        mocked_outqueue = mock.Mock(multiprocessing.queues.SimpleQueue())
        
        mockMSI = mock.Mock(spec=MockMSI('test', ''))
        
        # task = None
        mocked_inqueue.get.return_value = None
        ema_parallel_multiprocessing.worker(mocked_inqueue, 
                                            mocked_outqueue, 
                                            [mockMSI])
        mocked_logging.debug.assert_called_with('worker got sentinel -- exiting')
        
        # EOFError, IOError
        mocked_inqueue.get.side_effect = EOFError
        ema_parallel_multiprocessing.worker(mocked_inqueue, 
                                            mocked_outqueue, 
                                            [mockMSI])
        mocked_logging.debug.assert_called_with('worker got EOFError or IOError -- exiting')

        mocked_inqueue.get.side_effect = IOError
        ema_parallel_multiprocessing.worker(mocked_inqueue, 
                                            mocked_outqueue, 
                                            [mockMSI])
        mocked_logging.debug.assert_called_with('worker got EOFError or IOError -- exiting')
        
        # task = tuple of _, experiment dict
        #     - success
        #     - ema error
        #     - exception

        # setup of test, we get a normal case 
        experiment = {'experiment id':0}
        mocked_inqueue.get.return_value = (0, experiment)
        mocked_inqueue.get.side_effect = None        
        
        # running experiment raises EMAError
        mocked_runner().run_experiment.side_effect = EMAError
        feder_thread = threading.Thread(target=ema_parallel_multiprocessing.worker, 
                                        args=(mocked_inqueue, 
                                        mocked_outqueue, 
                                        [mockMSI]))
        feder_thread.deamon = True
        feder_thread.start()
        time.sleep(0.001) # to avoid race conditions
        mocked_inqueue.get.return_value = None

        mocked_runner().run_experiment.assert_called_with(experiment)
#         mocked_outqueue.put.assert_called_once()
        
        # reset mocks
        mocked_outqueue.reset_mock()
        mocked_runner().reset_mock()
        
        # running experiment raises EMAError
        experiment = {'experiment id':0}
        mocked_inqueue.get.return_value = (0, experiment)
        mocked_inqueue.get.side_effect = None   
        
        mocked_runner().run_experiment.side_effect = Exception
        feder_thread = threading.Thread(target=ema_parallel_multiprocessing.worker, 
                                        args=(mocked_inqueue, 
                                        mocked_outqueue, 
                                        [mockMSI]))
        feder_thread.deamon = True
        feder_thread.start()
        time.sleep(0.001) # to avoid race conditions
        mocked_inqueue.get.return_value = None

        mocked_runner().run_experiment.assert_called_with(experiment)
#         mocked_outqueue.put.assert_called_once()
        
        # reset mocks
        mocked_outqueue.reset_mock()
        mocked_runner().reset_mock()
        

        # running experiment works fine
        experiment = {'experiment id':0}
        mocked_inqueue.get.return_value = (0, experiment)
        mocked_inqueue.get.side_effect = None   
        mocked_runner().run_experiment.side_effect = None
        
        feder_thread = threading.Thread(target=ema_parallel_multiprocessing.worker, 
                                        args=(mocked_inqueue, 
                                        mocked_outqueue, 
                                        [mockMSI]))
        feder_thread.deamon = True
        feder_thread.start()
        time.sleep(0.001) # to avoid race conditions
        mocked_inqueue.get.return_value = None

        mocked_runner().run_experiment.assert_called_with(experiment)
#         mocked_outqueue.put.assert_called_once()
        
        # reset mocks
        mocked_outqueue.reset_mock()
        mocked_runner().reset_mock()

if __name__ == '__main__':
    
    unittest.main()
    