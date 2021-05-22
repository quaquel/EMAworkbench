'''
Created on Jul 28, 2015

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
import unittest

import unittest.mock as mock


from ema_workbench.em_framework.model import Model, FileModel, ReplicatorModel
from ema_workbench.em_framework.parameters import (RealParameter, Category,
                                                   CategoricalParameter)
from ema_workbench.em_framework.points import Scenario, Policy
from ema_workbench.util import EMAError
from ema_workbench.em_framework.outcomes import ScalarOutcome, ArrayOutcome

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
        
        # test complete translation of scenario
        
        model = Model(model_name, function)
        model.uncertainties = [RealParameter('a',  0 , 1, variable_name=['a', 'b'])]
        
        scenario = Scenario(**{'a':0.1})
        model.run_model(scenario, Policy('test'))
        
        self.assertIn('a', scenario.keys())
        self.assertIn('b', scenario.keys())
        
        model = Model(model_name, function)
        cats = [Category('some name', [1,2]),
                Category('some other name', [3,4])]
        model.uncertainties = [CategoricalParameter('a', cats, 
                                    variable_name=['a', 'b'], multivalue=True)]
        
        scenario = Scenario(**{'a':cats[0].value})
        model.run_model(scenario, Policy('test'))
        
        self.assertIn('a', scenario.keys())
        self.assertIn('b', scenario.keys())
        self.assertEqual(scenario['a'], 1)
        self.assertEqual(scenario['b'], 2)
        
        scenario = Scenario(**{'a':cats[1].value})
        model.run_model(scenario, Policy('test'))
        
        self.assertIn('a', scenario.keys())
        self.assertIn('b', scenario.keys())
        self.assertEqual(scenario['a'], 3)
        self.assertEqual(scenario['b'], 4)
        

        model_name = 'modelname'
        
        function = mock.Mock()
        
        model = Model(model_name, function)
        model.uncertainties = [RealParameter('a',  0 , 1)]
        model.run_model(Scenario(**{'a':0.1, 'b':1}), Policy('test'))
        function.assert_called_once_with(a=0.1)
        
        # test complete translation of scenario
        
        model = Model(model_name, function)
        model.uncertainties = [RealParameter('a',  0 , 1, variable_name=['a', 'b'])]
        
        scenario = Scenario(**{'a':0.1})
        model.run_model(scenario, Policy('test'))
        
        self.assertIn('a', scenario.keys())
        self.assertIn('b', scenario.keys())
        
        model = Model(model_name, function)
        cats = [Category('some name', [1,2]),
                Category('some other name', [3,4])]
        model.uncertainties = [CategoricalParameter('a', cats, 
                                    variable_name=['a', 'b'], multivalue=True)]
        
        scenario = Scenario(**{'a':cats[0].value})
        model.run_model(scenario, Policy('test'))
        
        self.assertIn('a', scenario.keys())
        self.assertIn('b', scenario.keys())
        self.assertEqual(scenario['a'], 1)
        self.assertEqual(scenario['b'], 2)
        
        scenario = Scenario(**{'a':cats[1].value})
        model.run_model(scenario, Policy('test'))
        
        self.assertIn('a', scenario.keys())
        self.assertIn('b', scenario.keys())
        self.assertEqual(scenario['a'], 3)
        self.assertEqual(scenario['b'], 4)        
    
    
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
        
    def test_as_dict(self):
        model_name = 'modelname'
        
        model = Model(model_name, lambda x:x)     
        model.uncertainties = [RealParameter('a',  0 , 1)]
        
        expected_keys = ['class', 'name', 'uncertainties', 'outcomes',
                         'outcomes', 'constants']
        
        dict_ = model.as_dict()
        
        for key in expected_keys:
            self.assertIn(key, dict_)
        

class TestReplicatorModel(unittest.TestCase):

    def test_run_model(self):
        model_name = 'modelname'
        
        function = mock.Mock()
        function.return_value = {'outcome': 2}
        
        model = ReplicatorModel(model_name, function)
        model.uncertainties = [RealParameter('a',  0 , 1)]
        model.outcomes = [ArrayOutcome('outcome')]
        model.replications = 2
        
        model.run_model(Scenario(**{'a':0.1, 'b':1}), Policy('test'))
        self.assertEqual(function.call_count, 2)
        self.assertEqual({'outcome':[2,2]}, model.outcomes_output)
        
        

#         self.assertEqual(scenario['b'], 4)           
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()