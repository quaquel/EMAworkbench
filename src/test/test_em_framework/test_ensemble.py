'''
Created on 18 jan. 2013

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
import unittest

import mock

from ema_workbench.em_framework.ensemble import (ModelEnsemble, 
                                                 experiment_generator)
from ema_workbench.em_framework.samplers import LHSSampler
from ema_workbench.em_framework import (RealParameter, TimeSeriesOutcome)
from ema_workbench.util.ema_exceptions import EMAError
from ema_workbench.em_framework.callbacks import DefaultCallback
from ema_workbench.em_framework.parameters import Policy, Scenario
from ema_workbench.em_framework.model import Model

         
class EnsembleTestCase(unittest.TestCase):
    
    def test_policies(self):
        ensemble = ModelEnsemble()
        
        policy = Policy('test')
        ensemble.policies = policy
        
        ensemble = ModelEnsemble()
        
        policies = [Policy('test'), Policy('name')]
        ensemble.policies = policies
        
    def test_model_structures(self):
        function = mock.Mock()
        model_a = Model("A", function)
        
        ensemble = ModelEnsemble()
        ensemble.model_structures = model_a
        self.assertEqual(ensemble.model_structures[0], model_a)
        
        model_a = Model("A", function)
        model_b = Model("B", function)
        ensemble = ModelEnsemble()
        ensemble.model_structures = [model_a, model_b]
        self.assertEqual(list(ensemble.model_structures), [model_a, model_b])
        
#     def test_generate_experiments(self):
#         function = mock.Mock()
#         
#         # everything shared
#         model_a = Model("A", function())
#         model_b = Model("B", function())
#         model_c = Model("C", function())
#         
#         # let's add some uncertainties to this
#         shared_abc_1 = RealParameter("shared abc 1", 0, 1)
#         shared_abc_2 = RealParameter("shared abc 2", 0, 1)
#         shared_ab_1 = RealParameter("shared ab 1", 0, 1)
#         shared_bc_1 = RealParameter("shared bc 1", 0, 1)
#         a_1 = RealParameter("a 1", 0, 1)
#         b_1 = RealParameter("b 1", 0, 1)
#         model_a.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, a_1]
#         model_b.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, shared_bc_1, b_1]
#         model_c.uncertainties = [shared_abc_1, shared_abc_2, shared_bc_1]
#         
#         ensemble = ModelEnsemble()
#         ensemble.model_structures = [model_a, model_b, model_c]
#         ensemble.policies =[Policy('none')]
#         experiments, nr_of_exp, uncertainties = ensemble._generate_experiments(10, UNION )
#         
#         msg = 'testing UNION'
#         
#         self.assertIn(shared_abc_1, uncertainties, msg)
#         self.assertIn(shared_abc_2, uncertainties, msg)
#         self.assertIn(shared_ab_1, uncertainties, msg)
#         self.assertIn(shared_bc_1, uncertainties, msg)
#         self.assertIn(a_1, uncertainties, msg)
#         self.assertIn(b_1, uncertainties, msg)        
#         self.assertEqual(nr_of_exp, 10* len(ensemble.model_structures), msg)
#         
#         for experiment in experiments:
#             self.assertEqual(experiment.policy.name, 'none', msg)
#             self.assertIn(experiment.model.name, ["A", "B", "C"], msg)
#             
#             model = experiment.model
#             for unc in model.uncertainties:
#                 self.assertIn(unc.name, experiment.scenario.keys())
# 
#         experiments, nr_of_exp, uncertainties = ensemble._generate_experiments(10, INTERSECTION )
#          
#         msg = 'testing INTERSECTION'
#         
#         self.assertIn(shared_abc_1, uncertainties, msg)
#         self.assertIn(shared_abc_2, uncertainties, msg)
#         self.assertNotIn(shared_ab_1, uncertainties, msg)
#         self.assertNotIn(shared_bc_1, uncertainties, msg)
#         self.assertNotIn(a_1, uncertainties, msg)
#         self.assertNotIn(b_1, uncertainties, msg)          
#         
#         self.assertEqual(nr_of_exp, 10* len(ensemble.model_structures), msg)
# 
#         for experiment in experiments:
#             self.assertEqual(experiment.policy.name, 'none', msg)
#             self.assertIn(experiment.model.name, ["A", "B", "C"], msg)
#                         
#             model = experiment.model
#             for unc in [shared_abc_1, shared_abc_2]:
#                 self.assertIn(unc.name, experiment.scenario.keys())
#             self.assertNotEqual(len(experiment.scenario.keys()), len(model.uncertainties))
#             self.assertEqual(len(experiment.scenario.keys()), 2)
#             
#             
#         # predefined experiments
#         model_a = Model("A", mock.Mock())
#         
#         # let's add some uncertainties to this
#         shared_abc_1 = RealParameter("shared abc 1", 0, 1)
#         shared_abc_2 = RealParameter("shared abc 2", 0, 1)
#         shared_ab_1 = RealParameter("shared ab 1", 0, 1)
#         a_1 = RealParameter("a 1", 0, 1)
#         model_a.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, a_1]
#         ensemble = ModelEnsemble()
#         ensemble.policies = [Policy('none')]
#         ensemble.model_structure = model_a
#         
#         cases = [{"shared abc 1":1, "shared abc 2":2, "shared ab 1":3, "a 1": 4}]
#         
#         experiments, nr_of_exp, uncertainties = ensemble._generate_experiments(cases, UNION)
#         
#         self.assertEqual(nr_of_exp, 1)
#         self.assertIn(shared_abc_1, uncertainties)
#         self.assertIn(shared_abc_2, uncertainties)
#         self.assertIn(shared_ab_1, uncertainties)
#         self.assertIn(a_1, uncertainties)
#         
#         # test raises EMAError
#         self.assertRaises(EMAError, ensemble._generate_experiments, 'a string', UNION)
        
        

#     def test_determine_unique_attributes(self):
#         function = mock.Mock()
#         # everything shared
#         model_a = Model("A", function)
#         model_b = Model("B", function)
#         model_c = Model("C", function)
#         
#         # let's add some uncertainties to this
#         shared_abc_1 = RealParameter("shared abc 1", 0, 1)
#         shared_abc_2 = RealParameter("shared abc 2", 0, 1)
#         shared_ab_1 = RealParameter("shared ab 1", 0, 1)
#         shared_bc_1 = RealParameter("shared bc 1", 0, 1)
#         a_1 = RealParameter("a 1", 0, 1)
#         b_1 = RealParameter("b 1", 0, 1)
#         model_a.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, a_1]
#         model_b.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, shared_bc_1, b_1]
#         model_c.uncertainties = [shared_abc_1, shared_abc_2, shared_bc_1]    
#         
#         ensemble = ModelEnsemble()
#         ensemble.model_structures = [model_a, model_b, model_c]
#         
#         overview_dict, element_dict  = ensemble._determine_unique_attributes('uncertainties')
#         
#         msg = 'checking uncertainties'
#         self.assertIn(shared_abc_1.name, element_dict.keys(), msg)
#         self.assertIn(shared_abc_2.name, element_dict.keys(), msg)
#         self.assertIn(shared_ab_1.name, element_dict.keys(), msg)
#         self.assertIn(shared_bc_1.name, element_dict.keys(), msg)
#         self.assertIn(a_1.name, element_dict.keys(), msg)
#         self.assertIn(b_1.name, element_dict.keys(), msg)
#         
#         self.assertEqual(len(overview_dict.keys()),5, msg)
#         
#         # let's add some uncertainties to this
#         shared_abc_1 = RealParameter("shared abc 1", 0, 1)
#         shared_abc_2 = RealParameter("shared abc 1", 0, 2)
#         shared_ab_1 = RealParameter("shared ab 1", 0, 1)
#         a_1 = RealParameter("a 1", 0, 1)
#         b_1 = RealParameter("b 1", 0, 1)
#         model_a.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, a_1]
#         model_b.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, shared_bc_1, b_1]
#         
#         ensemble = ModelEnsemble()
#         ensemble.model_structures = [model_a, model_b, model_c]
#         
#         self.assertRaises(EMAError, ensemble._determine_unique_attributes, 'uncertainties')
        

    def test_perform_experiments(self):
        
        # everything shared
        model_a = Model("A", mock.Mock())
        model_b = Model("B", mock.Mock())
        model_c = Model("C", mock.Mock())
        models = [model_a, model_b, model_c]
        
        # let's add some uncertainties to this
        shared_abc_1 = RealParameter("shared abc 1", 0, 1)
        shared_abc_2 = RealParameter("shared abc 2", 0, 1)
        shared_ab_1 = RealParameter("shared ab 1", 0, 1)
        shared_bc_1 = RealParameter("shared bc 1", 0, 1)
        a_1 = RealParameter("a 1", 0, 1)
        b_1 = RealParameter("b 1", 0, 1)
        model_a.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, a_1]
        model_b.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, shared_bc_1, b_1]
        model_c.uncertainties = [shared_abc_1, shared_abc_2, shared_bc_1]
        
        #let's add an outcome to this
        outcome_shared = TimeSeriesOutcome("test")
        model_a.outcomes = [outcome_shared]
        model_b.outcomes = [outcome_shared]
        model_c.outcomes = [outcome_shared]
        
        for model in models:
            model.function.return_value = {a: [0.1]*10 for a in outcome_shared.variable_name}
        
        ensemble = ModelEnsemble()
        ensemble.model_structures = [model_a, model_b, model_c]
        ensemble.policies = [Policy('None')]
        
        ensemble.perform_experiments(10, uncertainty_union=True, 
                                         outcome_union=True,
                                         reporting_interval=1 )
#         for model in models:
#             model.function.assert_has_calls() TODO::
        

        ensemble.perform_experiments(10, uncertainty_union=True, 
                                         outcome_union=False,
                                         reporting_interval=1 )

        ensemble.perform_experiments(10, uncertainty_union=False, 
                                         outcome_union=True,
                                         reporting_interval=1 )

        ensemble.perform_experiments(10, uncertainty_union=False, 
                                         outcome_union=False,
                                         reporting_interval=1 )
#         
#         self.assertRaises(ValueError, ensemble.perform_experiments,
#                          10, uncertainty_union=False, 
#                          union_outcomes='Label')
        
        with mock.patch('ema_workbench.em_framework.ensemble.MultiprocessingPool') as MockPool:
            ensemble.parallel = True
            
            mockedCallback = mock.Mock(DefaultCallback)
            mockedCallback.configure_mock(**{'i':30})
            mockedCallback.return_value = mockedCallback
            
            ensemble.perform_experiments(10, uncertainty_union=True, 
                                             outcome_union=True,
                                             reporting_interval=1,
                                             callback=mockedCallback)
            
            self.assertEqual(2, len(MockPool.mock_calls))
            
            MockPool.reset_mock()
            mockedCallback = mock.Mock(DefaultCallback)
            mockedCallback.configure_mock(**{'i':10})
            mockedCallback.return_value = mockedCallback
            
            self.assertRaises(EMAError, ensemble.perform_experiments,
                              10, uncertainty_union=True, 
                              outcome_union=True, reporting_interval=1,
                             callback=mockedCallback)

            
    
    def test_experiment_generator(self):
        sampler = LHSSampler()
        
        shared_abc_1 = RealParameter("shared ab 1", 0, 1)
        shared_abc_2 = RealParameter("shared ab 2", 0, 1)
        unique_a = RealParameter("unique a ", 0, 1)
        unique_b = RealParameter("unique b ", 0, 1)
        uncertainties = [shared_abc_1, shared_abc_2, unique_a, unique_b]
        designs  = sampler.generate_designs(uncertainties, 10)
        designs.kind = Scenario
        
        # everything shared
        model_a = Model("A", mock.Mock())
        model_b = Model("B", mock.Mock())
        
        model_a.uncertainties = [shared_abc_1, shared_abc_2, unique_a]
        model_b.uncertainties = [shared_abc_1, shared_abc_2, unique_b]
        model_structures = [model_a, model_b]
        
        policies = [Policy('policy 1'),
                    Policy('policy 2'),
                    Policy('policy 3')]
        
        gen = experiment_generator(designs, model_structures, policies)
        
        experiments = []
        for entry in gen:
            experiments.append(entry)
        self.assertEqual(len(experiments), 2*3*10)

if __name__ == "__main__":
    unittest.main()