'''
Created on Jul 28, 2015

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
import unittest

from core import ModelStructureInterface
from util.ema_exceptions import EMAError

class TestMSI(ModelStructureInterface):
    def model_init(self, policy, kwargs):
        self.policy = policy
        self.kwargs = kwargs
        
    def run_model(self, case):
        pass

class Test(unittest.TestCase):

    def test_init(self):
        model_name = 'modelname'
        wd = '/test'
        
        model = TestMSI(wd, model_name)
        
        self.assertEqual(model.name, model_name)
        self.assertEqual(model.working_directory, wd)
        self.assertRaises(EMAError, TestMSI, '', 'model name')
        

    def test_model_init(self):
        model_name = 'modelname'
        wd = '/test'
        
        model = TestMSI(wd, model_name)
        policy = {'name':'test'}
        model.model_init(policy, {})
        
        self.assertEqual(policy, model.policy)
        self.assertEqual({}, model.kwargs)
        
        model.working_directory = wd
    
    def test_run_model(self):
        model_name = 'modelname'
        wd = '/test'
        
        model = TestMSI(wd, model_name)
        model.run_model({})
    
    def test_retrieve_output(self):
        model_name = 'modelname'
        wd = '/test'
        
        model = TestMSI(wd, model_name)
        output = model.retrieve_output()
        
        self.assertEqual({}, output)
    
    def test_cleanup(self):
        model_name = 'modelname'
        wd = '/test'
        
        model = TestMSI(wd, model_name)
        model.cleanup()

    def test_get_model_uncertainties(self):
        model_name = 'modelname'
        wd = '/test'
        
        model = TestMSI(wd, model_name)
        uncs = model.get_model_uncertainties()
        
        self.assertEqual([], uncs)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()