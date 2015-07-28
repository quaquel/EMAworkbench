'''
Created on Jul 28, 2015

@author: jhkwakkel
'''
import unittest

from expWorkbench import ModelStructureInterface

class TestMSI(ModelStructureInterface):
    def model_init(self, policy, kwargs):
        self.policy = policy
        self.kwargs = kwargs
        
    def run_model(self, case):
        pass



class Test(unittest.TestCase):

    def test_init(self):
        pass

    def test_model_init(self):
        pass
    
    def test_run_model(self):
        pass
    
    def test_retrieve_output(self):
        pass
    
    def test_cleanup(self):
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()