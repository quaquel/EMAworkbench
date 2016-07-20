'''
Created on Jul 28, 2015

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
import unittest

<<<<<<< HEAD:src/ema_workbench/test/test_em_framework/test_model.py
=======
from ema_workbench.em_framework import ModelStructureInterface
from ema_workbench.util.ema_exceptions import EMAError
>>>>>>> master:test/test_em_framework/test_model.py

from ema_workbench.em_framework.model import Model
from ema_workbench.em_framework.parameters import RealParameter
from ema_workbench.util import EMAError

# from ...em_framework import ModelStructureInterface
# from ...util.ema_exceptions import EMAError

class TestMSI(Model):
    def model_init(self, policy, kwargs):
        self.policy = policy
        self.kwargs = kwargs
        
    def run_model(self, case):
        pass

class Test(unittest.TestCase):

    def test_init(self):
        model_name = 'modelname'
        wd = '/test'
        
        model = TestMSI(model_name, wd)
        
        self.assertEqual(model.name, model_name)
        self.assertEqual(model.working_directory, wd)
        self.assertRaises(EMAError, TestMSI, '', 'model name')
        

    def test_model_init(self):
        model_name = 'modelname'
        wd = '/test'
        
        model = TestMSI(model_name, wd)
        policy = {'name':'test'}
        model.model_init(policy, {})
        
        self.assertEqual(policy, model.policy)
        self.assertEqual({}, model.kwargs)
        
        model.working_directory = wd
    
    def test_run_model(self):
        model_name = 'modelname'
        wd = '/test'
        
        model = TestMSI(model_name, wd)
        model.run_model({})
    
    def test_retrieve_output(self):
        model_name = 'modelname'
        wd = '/test'
        
        model = TestMSI(model_name, wd)
        output = model.retrieve_output()
        
        self.assertEqual({}, output)
    
    def test_cleanup(self):
        model_name = 'modelname'
        wd = '/test'
        
        model = TestMSI(model_name, wd)
        model.cleanup()

    def test_model_uncertainties(self):
        model_name = 'modelname'
        wd = '/test'
        
        model = TestMSI(model_name, wd)
        
        self.assertTrue(len(model.uncertainties.keys())==0)
        
        unc_a = RealParameter('a', 0, 1)
        model.uncertainties = unc_a
        self.assertTrue(len(model.uncertainties.keys())==1)
        self.assertTrue(unc_a.name in model.uncertainties)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()