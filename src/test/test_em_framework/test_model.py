'''
Created on Jul 28, 2015

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
import unittest


from ema_workbench.em_framework.model import Model
from ema_workbench.em_framework.parameters import RealParameter
from ema_workbench.util import EMAError
from ema_workbench.em_framework.outcomes import ScalarOutcome

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
        
        model = TestMSI(model_name)
        
        self.assertEqual(model.name, model_name)
        self.assertRaises(EMAError, TestMSI, '', 'model name')
        

    def test_model_init(self):
        model_name = 'modelname'
        
        model = TestMSI(model_name)
        policy = {'name':'test'}
        model.model_init(policy, {})
        
        self.assertEqual(policy, model.policy)
        self.assertEqual({}, model.kwargs)
        
    def test_run_model(self):
        model_name = 'modelname'
        
        model = TestMSI(model_name)
        model.run_model({})
    
    def test_retrieve_output(self):
        model_name = 'modelname'
        
        model = TestMSI(model_name)
        model.outcomes = [ScalarOutcome('a')]
        
        output = model.retrieve_output()
        self.assertEqual({}, output)
        
        output = {'a': 0 }
        model.output = output
        self.assertEqual(output, model.retrieve_output())
    
    def test_cleanup(self):
        model_name = 'modelname'
        
        model = TestMSI(model_name)
        model.cleanup()

    def test_model_uncertainties(self):
        model_name = 'modelname'
        
        model = TestMSI(model_name)
        self.assertTrue(len(model.uncertainties.keys())==0)
        
        unc_a = RealParameter('a', 0, 1)
        model.uncertainties = unc_a
        self.assertTrue(len(model.uncertainties.keys())==1)
        self.assertTrue(unc_a.name in model.uncertainties)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()