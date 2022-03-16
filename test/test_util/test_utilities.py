'''
Created on 22 nov. 2012

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import (absolute_import, print_function, division)
import os
import unittest

import numpy as np
import pandas as pd

from ema_workbench.util.utilities import (save_results, load_results,
                              merge_results, get_ema_project_home_dir)
from ema_workbench.em_framework.outcomes import (ScalarOutcome, ArrayOutcome,
                                                 TimeSeriesOutcome, register)






def setUpModule():
    global cwd 
    cwd = os.getcwd()
    dir_of_module = os.path.dirname(os.path.abspath(__file__))
    register.outcomes = {}  # reset internal dict
    os.chdir(dir_of_module)

def tearDownModule():
    os.chdir(cwd)

class SaveResultsTestCase(unittest.TestCase):
    def test_save_results(self):
        fn = '../data/test.tar.gz'

        # test for 1d
        nr_experiments = 10000
        experiments = pd.DataFrame(index=np.arange(nr_experiments),
                                   columns={'x': float,
                                            'y': float})
        outcome_q = np.random.rand(nr_experiments, 1)

        outcomes = {}
        outcomes[ScalarOutcome('q').name] = outcome_q
        results = (experiments, outcomes)
    
        # test for 2d
        save_results(results, fn)
        os.remove(fn)
        
        nr_experiments = 10000
        nr_timesteps = 100
        experiments = pd.DataFrame(index=np.arange(nr_experiments),
                                   columns={'x': float,
                                            'y': float})
        outcome_r = np.zeros((nr_experiments,nr_timesteps))
        
        outcomes = {}
        outcomes[ArrayOutcome('r').name] = outcome_r
        results = (experiments, outcomes)

        save_results(results, fn)
        os.remove(fn)

        # test for 3d
        # test for very large       
        nr_experiments = 10000
        nr_timesteps = 100
        nr_replications = 10
        experiments = pd.DataFrame(index=np.arange(nr_experiments),
                                   columns={'x': float,
                                            'y': float})
        outcome_s = np.zeros((nr_experiments, nr_timesteps, nr_replications))

        outcomes = {}
        outcomes[ArrayOutcome('s').name] = outcome_s
        results = (experiments, outcomes)

        save_results(results, fn)
        os.remove(fn)

    
class LoadResultsTestCase(unittest.TestCase):
    def test_load_results(self):
        # test for 1d
        
        # test for 3d
    
        nr_experiments = 10000
        
        # test for 2d
        experiments = pd.DataFrame(index=np.arange(nr_experiments),
                                   columns={'x': float,
                                            'y': float})
            
        experiments['x'] = np.random.rand(nr_experiments)
        experiments['y'] = np.random.rand(nr_experiments)
        
        outcome_a = np.zeros((nr_experiments, 1))

        outcomes = {}
        outcomes[ArrayOutcome('a').name] = outcome_a
        results = (experiments, outcomes)

        save_results(results, '../data/test.tar.gz')
        loaded_experiments, outcomes = load_results('../data/test.tar.gz')
        
        self.assertTrue(np.all(np.allclose(outcomes['a'],outcome_a)))
        self.assertTrue(np.all(np.allclose(experiments['x'],loaded_experiments['x'])))
        self.assertTrue(np.all(np.allclose(experiments['y'],loaded_experiments['y'])))        
        
        os.remove('../data/test.tar.gz')
                
        # test 3d
        nr_experiments = 1000
        nr_timesteps = 100
        nr_replications = 10
        experiments = pd.DataFrame(index=np.arange(nr_experiments),
                                   columns={'x': float,
                                            'y': float})
        experiments['x'] = np.random.rand(nr_experiments)
        experiments['y'] = np.random.rand(nr_experiments)
        
        outcome_b = np.zeros((nr_experiments,nr_timesteps,nr_replications))

        outcomes = {}
        outcomes[ArrayOutcome('b').name] = outcome_b
        results = (experiments, outcomes)

        save_results(results, '../data/test.tar.gz')
        loaded_experiments, outcomes = load_results('../data/test.tar.gz')
        
        os.remove('../data/test.tar.gz')
        
        self.assertTrue(np.all(np.allclose(outcomes['b'],outcome_b)))
        self.assertTrue(np.all(np.allclose(experiments['x'],loaded_experiments['x'])))
        self.assertTrue(np.all(np.allclose(experiments['y'],loaded_experiments['y'])))        
        
        
class ExperimentsToScenariosTestCase(unittest.TestCase):
    pass

class MergeResultsTestCase(unittest.TestCase):
    
    def test_merge_results(self):
        results1 = load_results('../data/1000 runs scarcity.tar.gz')
        results2 = load_results('../data/1000 runs scarcity.tar.gz')
        
        n1 = results1[0].shape[0]
        n2 = results2[0].shape[0]
        
        merged = merge_results(results1, results2)
        
        self.assertEqual(merged[0].shape[0], n1+n2)
    

class ConfigTestCase(unittest.TestCase):
    def test_get_home_dir(self):
        _ = get_ema_project_home_dir()
        


if __name__ == '__main__':
    unittest.main()
#     testsuite = unittest.TestSuite()
#     testsuite.addTest(LoadResultsTestCase("test_load_results"))
#     unittest.TextTestRunner().run(testsuite)