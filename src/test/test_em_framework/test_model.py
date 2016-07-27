'''
Created on Jul 28, 2015

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''

from __future__ import (print_function, absolute_import, unicode_literals, 
                        division)

import unittest

try:
    import unittest.mock as mock
except ImportError:
    import mock

from ema_workbench.em_framework.model import Model, FileModel
from ema_workbench.em_framework.parameters import (RealParameter, Policy, 
                                                   Scenario)
from ema_workbench.util import EMAError

class FileModelTest(FileModel):
    def run_model(self, scenario, policy):
        super(FileModelTest, self).run_model(scenario, policy)

class TestFileModel(unittest.TestCase):
    def test_init(self):
        model_name = 'modelname'
        model_file = 'model_file'
        
        with self.assertRaises(ValueError):
            FileModelTest(model_name, '.', model_file)
            
        with mock.patch('ema_workbench.em_framework.model.os') as patch:
            patch.os.is_file.set_return_value(True)
            model = FileModelTest(model_name, '.', model_file)
            self.assertEqual(model.name, model_name, 'FileModel name not equal')
            self.assertEqual(model.model_file, model_file)

        
    def test_run_model(self):
        model_name = 'modelname'
        model_file = 'model_file'
        
            
        with mock.patch('ema_workbench.em_framework.model.os') as patch:
            patch.os.is_file.set_return_value(True)
            model = FileModelTest(model_name, '.', model_file)
            model.run_model(Scenario(a=1), Policy('test', b=2))
            self.assertEqual(model.policy.name, 'test')

class TestModel(unittest.TestCase):

    def test_init(self):
        model_name = 'modelname'
        
        model = Model(model_name, lambda x:x)
        
        self.assertEqual(model.name, model_name)
        self.assertRaises(EMAError, Model, '', 'model name')
        

    def test_model_init(self):
        model_name = 'modelname'
        
        def initial_func(a=1):
            return a
        
        model = Model(model_name,initial_func)
        
        def policy_func(a=1):
            return a
        
        policy = Policy('test', function=policy_func, unknown='a')
        model.model_init(policy)
        
        self.assertEqual(policy, model.policy)
        self.assertEqual(model.function, policy_func)
        
        with self.assertRaises(AttributeError):
            model.unknown
        
        
    def test_run_model(self):
        model_name = 'modelname'
        
        function = mock.Mock()
        
        model = Model(model_name, function)
        model.uncertainties = [RealParameter('a',  0 , 1)]
        model.run_model(Scenario(**{'a':0.1, 'b':1}), Policy('test'))
        function.assert_called_once_with(a=0.1)
    
    def test_cleanup(self):
        model_name = 'modelname'
        
        model = Model(model_name, lambda x:x)
        model.cleanup()

    def test_model_uncertainties(self):
        model_name = 'modelname'
        
        model = Model(model_name, lambda x:x)
        self.assertTrue(len(model.uncertainties.keys())==0)
        
        unc_a = RealParameter('a', 0, 1)
        model.uncertainties = unc_a
        self.assertTrue(len(model.uncertainties.keys())==1)
        self.assertTrue(unc_a.name in model.uncertainties)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()