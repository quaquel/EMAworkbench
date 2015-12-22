'''
Created on 18 jan. 2013

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
import six

import mock
import numpy as np
import unittest

from ...em_framework.model_ensemble import (ModelEnsemble, UNION, INTERSECTION,
                                            experiment_generator)
from ...em_framework.samplers import LHSSampler
from ...em_framework import (ModelStructureInterface, ParameterUncertainty, 
                             Outcome, model_ensemble)
from ...util.ema_exceptions import EMAError

class DummyInterface(ModelStructureInterface):
    
    def model_init(self, policy, kwargs):
        pass
    
    def run_model(self, case):
        for outcome in self.outcomes:
            self.output[outcome.name] = np.random.rand(10,)
         
class ModelEnsembleTestCase(unittest.TestCase):
    
    def test_policies(self):
        ensemble = ModelEnsemble()
        
        policy = {'name': 'test'}
        ensemble.policies = policy
        
        ensemble = ModelEnsemble()
        
        policies = [{'name': 'test'}, {'name': 'test2'}]
        ensemble.policies = policies
        
    def test_model_structures(self):
        model_a = DummyInterface(None, "A")
        
        ensemble = ModelEnsemble()
        ensemble.model_structure = model_a
        self.assertEqual(ensemble.model_structure, model_a)
        
        model_a = DummyInterface(None, "A")
        model_b = DummyInterface(None, "B")
        ensemble = ModelEnsemble()
        ensemble.model_structures = [model_a, model_b]
        self.assertEqual(list(ensemble.model_structures), [model_a, model_b])
        
    def test_generate_experiments(self):
        # everything shared
        model_a = DummyInterface(None, "A")
        model_b = DummyInterface(None, "B")
        model_c = DummyInterface(None, "C")
        
        # let's add some uncertainties to this
        shared_abc_1 = ParameterUncertainty((0,1), "shared abc 1")
        shared_abc_2 = ParameterUncertainty((0,1), "shared abc 2")
        shared_ab_1 = ParameterUncertainty((0,1), "shared ab 1")
        shared_bc_1 = ParameterUncertainty((0,1), "shared bc 1")
        a_1 = ParameterUncertainty((0,1), "a 1")
        b_1 = ParameterUncertainty((0,1), "b 1")
        model_a.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, a_1]
        model_b.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, shared_bc_1, b_1]
        model_c.uncertainties = [shared_abc_1, shared_abc_2, shared_bc_1]
        
        ensemble = ModelEnsemble()
        ensemble.model_structures = [model_a, model_b, model_c]
        experiments, nr_of_exp, uncertainties = ensemble._generate_experiments(10, UNION )
        
        msg = 'testing UNION'
        
        self.assertIn(shared_abc_1, uncertainties, msg)
        self.assertIn(shared_abc_2, uncertainties, msg)
        self.assertIn(shared_ab_1, uncertainties, msg)
        self.assertIn(shared_bc_1, uncertainties, msg)
        self.assertIn(a_1, uncertainties, msg)
        self.assertIn(b_1, uncertainties, msg)        
        self.assertEqual(nr_of_exp, 10* len(ensemble.model_structures), msg)
        
        for experiment in experiments:
            self.assertIn('policy', experiment.keys(), msg)
            self.assertIn('model', experiment.keys(), msg)
            self.assertIn('experiment id', experiment.keys(), msg)
            
            model = ensemble._msis[experiment['model']]
            for unc in model.uncertainties:
                self.assertIn(unc.name, experiment.keys())
            self.assertEqual(len(experiment.keys()), len(model.uncertainties)+3)
            

        experiments, nr_of_exp, uncertainties = ensemble._generate_experiments(10, INTERSECTION )
         
        msg = 'testing INTERSECTION'
        
        self.assertIn(shared_abc_1, uncertainties, msg)
        self.assertIn(shared_abc_2, uncertainties, msg)
        self.assertNotIn(shared_ab_1, uncertainties, msg)
        self.assertNotIn(shared_bc_1, uncertainties, msg)
        self.assertNotIn(a_1, uncertainties, msg)
        self.assertNotIn(b_1, uncertainties, msg)          
        
        self.assertEqual(nr_of_exp, 10* len(ensemble.model_structures), msg)
        experiment = six.next(experiments)
        self.assertIn('policy', experiment.keys(), msg)
        self.assertIn('model', experiment.keys(), msg)
        self.assertIn('experiment id', experiment.keys(), msg)


        for experiment in experiments:
            self.assertIn('policy', experiment.keys(), msg)
            self.assertIn('model', experiment.keys(), msg)
            self.assertIn('experiment id', experiment.keys(), msg)
            
            model = ensemble._msis[experiment['model']]
            for unc in [shared_abc_1, shared_abc_2]:
                self.assertIn(unc.name, experiment.keys())
            self.assertNotEqual(len(experiment.keys()), len(model.uncertainties)+3)
            self.assertEqual(len(experiment.keys()), 5)
            
            
        # predefined experiments
        model_a = DummyInterface(None, "A")
        
        # let's add some uncertainties to this
        shared_abc_1 = ParameterUncertainty((0,1), "shared abc 1")
        shared_abc_2 = ParameterUncertainty((0,1), "shared abc 2")
        shared_ab_1 = ParameterUncertainty((0,1), "shared ab 1")
        a_1 = ParameterUncertainty((0,1), "a 1")
        model_a.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, a_1]
        ensemble = ModelEnsemble()
        ensemble.model_structure = model_a
        
        cases = [{"shared abc 1":1, "shared abc 2":2, "shared ab 1":3, "a 1": 4}]
        
        experiments, nr_of_exp, uncertainties = ensemble._generate_experiments(cases, UNION)
        
        self.assertEqual(nr_of_exp, 1)
        self.assertIn(shared_abc_1, uncertainties)
        self.assertIn(shared_abc_2, uncertainties)
        self.assertIn(shared_ab_1, uncertainties)
        self.assertIn(a_1, uncertainties)
        
        # test raises EMAError
        self.assertRaises(EMAError, ensemble._generate_experiments, 'a string', UNION)
        
        

    def test_determine_unique_attributes(self):
        # everything shared
        model_a = DummyInterface(None, "A")
        model_b = DummyInterface(None, "B")
        model_c = DummyInterface(None, "C")
        
        # let's add some uncertainties to this
        shared_abc_1 = ParameterUncertainty((0,1), "shared abc 1")
        shared_abc_2 = ParameterUncertainty((0,1), "shared abc 2")
        shared_ab_1 = ParameterUncertainty((0,1), "shared ab 1")
        shared_bc_1 = ParameterUncertainty((0,1), "shared bc 1")
        a_1 = ParameterUncertainty((0,1), "a 1")
        b_1 = ParameterUncertainty((0,1), "b 1")
        model_a.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, a_1]
        model_b.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, shared_bc_1, b_1]
        model_c.uncertainties = [shared_abc_1, shared_abc_2, shared_bc_1]    
        
        ensemble = ModelEnsemble()
        ensemble.model_structures = [model_a, model_b, model_c]
        
        overview_dict, element_dict  = ensemble._determine_unique_attributes('uncertainties')
        
        msg = 'checking uncertainties'
        self.assertIn(shared_abc_1.name, element_dict.keys(), msg)
        self.assertIn(shared_abc_2.name, element_dict.keys(), msg)
        self.assertIn(shared_ab_1.name, element_dict.keys(), msg)
        self.assertIn(shared_bc_1.name, element_dict.keys(), msg)
        self.assertIn(a_1.name, element_dict.keys(), msg)
        self.assertIn(b_1.name, element_dict.keys(), msg)
        
        self.assertEqual(len(overview_dict.keys()),5, msg)
        
        # let's add some uncertainties to this
        shared_abc_1 = ParameterUncertainty((0,1), "shared abc 1")
        shared_abc_2 = ParameterUncertainty((0,2), "shared abc 1")
        shared_ab_1 = ParameterUncertainty((0,1), "shared ab 1")
        a_1 = ParameterUncertainty((0,1), "a 1")
        b_1 = ParameterUncertainty((0,1), "b 1")
        model_a.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, a_1]
        model_b.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, shared_bc_1, b_1]
        
        ensemble = ModelEnsemble()
        ensemble.model_structures = [model_a, model_b, model_c]
        
        self.assertRaises(EMAError, ensemble._determine_unique_attributes, 'uncertainties')
        

    def test_perform_experiments(self):
        # everything shared
        model_a = DummyInterface(None, "A")
        model_b = DummyInterface(None, "B")
        model_c = DummyInterface(None, "C")
        
        # let's add some uncertainties to this
        shared_abc_1 = ParameterUncertainty((0,1), "shared abc 1")
        shared_abc_2 = ParameterUncertainty((0,1), "shared abc 2")
        shared_ab_1 = ParameterUncertainty((0,1), "shared ab 1")
        shared_bc_1 = ParameterUncertainty((0,1), "shared bc 1")
        a_1 = ParameterUncertainty((0,1), "a 1")
        b_1 = ParameterUncertainty((0,1), "b 1")
        model_a.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, a_1]
        model_b.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, shared_bc_1, b_1]
        model_c.uncertainties = [shared_abc_1, shared_abc_2, shared_bc_1]
        
        #let's add an outcome to this
        outcome_shared = Outcome("test", time=True)
        model_a.outcomes = [outcome_shared]
        model_b.outcomes = [outcome_shared]
        model_c.outcomes = [outcome_shared]
        
        ensemble = ModelEnsemble()
        ensemble.model_structures = [model_a, model_b, model_c]
        
        ensemble.perform_experiments(10, which_uncertainties=UNION, 
                                         which_outcomes=UNION,
                                         reporting_interval=1 )

        ensemble.perform_experiments(10, which_uncertainties=UNION, 
                                         which_outcomes=INTERSECTION,
                                         reporting_interval=1 )

        ensemble.perform_experiments(10, which_uncertainties=INTERSECTION, 
                                         which_outcomes=UNION,
                                         reporting_interval=1 )

        ensemble.perform_experiments(10, which_uncertainties=INTERSECTION, 
                                         which_outcomes=INTERSECTION,
                                         reporting_interval=1 )
        
        self.assertRaises(ValueError, ensemble.perform_experiments,
                         10, which_uncertainties=INTERSECTION, 
                         which_outcomes='Label')
        
        with mock.patch('ema_workbench.em_framework.model_ensemble.MultiprocessingPool') as MockPool:
            ensemble.parallel = True
            ensemble.perform_experiments(10, which_uncertainties=UNION, 
                                             which_outcomes=UNION,
                                             reporting_interval=1 )
            
            self.assertEqual(2, len(MockPool.mock_calls))
            
    
    def test_experiment_generator(self):
        sampler = LHSSampler()
        
        shared_abc_1 = ParameterUncertainty((0,1), "shared ab 1")
        shared_abc_2 = ParameterUncertainty((0,1), "shared ab 2")
        unique_a = ParameterUncertainty((0,1), "unique a ")
        unique_b = ParameterUncertainty((0,1), "unique b ")
        uncertainties = [shared_abc_1, shared_abc_2, unique_a, unique_b]
        designs, _ = sampler.generate_designs(uncertainties, 10)
        
        # everything shared
        model_a = DummyInterface(None, "A")
        model_b = DummyInterface(None, "B")
        
        model_a.uncertainties = [shared_abc_1, shared_abc_2, unique_a]
        model_b.uncertainties = [shared_abc_1, shared_abc_2, unique_b]
        model_structures = [model_a, model_b]
        
        policies = [{'name':'policy 1'},
                    {'name':'policy 2'},
                    {'name':'policy 3'},]
        
        gen = experiment_generator(designs, model_structures, policies)
        
        experiments = []
        for entry in gen:
            experiments.append(entry)
        self.assertEqual(len(experiments), 2*3*10)

class MockMSI(ModelStructureInterface):

    def run_model(self, case):
        ModelStructureInterface.run_model(self, case)

    def model_init(self, policy, kwargs):
        ModelStructureInterface.model_init(self, policy, kwargs)

if __name__ == "__main__":
    unittest.main()