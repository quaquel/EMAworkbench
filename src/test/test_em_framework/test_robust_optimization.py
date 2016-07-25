'''
Created on Mar 1, 2012

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import division

import numpy as np

from ema_workbench.em_framework import (ModelEnsemble, MINIMIZE, UNION,
                                        ModelStructureInterface, Outcome,
                                        ParameterUncertainty)
from ema_workbench.em_framework.ema_optimization import epsNSGA2
from ema_workbench.util import ema_logging


class DummyModel(ModelStructureInterface):

    uncertainties = [ParameterUncertainty((0,1), 'a'),
                     ParameterUncertainty((0,1), 'b')]

    outcomes = [Outcome('a', time=False),
                Outcome('b', time=False)]

    def model_init(self, policy, kwargs):
        pass

    def run_model(self, case):
        self.output = {outcome.name:case[outcome.name] for outcome in self.outcomes}

def obj_func(outcomes):

    a = outcomes['a']
    b = outcomes['b']

    a_mean = np.mean(a)
    b_mean = np.mean(b)

    if a_mean < 0.5 or b_mean < 0.5:
        return (np.inf,) * 2
    else:
        return a_mean, b_mean

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)
    model = DummyModel(r"", "dummy")
    
    np.random.seed(123456789)
       
    ensemble = ModelEnsemble()
    ensemble.set_model_structure(model)

    
    policy_levers = {'Trigger a': {'type':'list', 'values':[0, 0.25, 0.5, 0.75, 1]},
                     'Trigger b': {'type':'list', 'values':[0, 0.25, 0.5, 0.75, 1]},
                     'Trigger c': {'type':'list', 'values':[0, 0.25, 0.5, 0.75, 1]}}
    
    cases = ensemble._generate_samples(10, UNION)[0]
    ensemble.add_policy({"name":None})
    experiments = [entry for entry in ensemble._generate_experiments(cases)]
    for entry in experiments:
        entry.pop("model")
        entry.pop("policy")
    cases = experiments    
    
    stats, pop   = ensemble.perform_robust_optimization(cases=cases,
                                               reporting_interval=100,
                                               obj_function=obj_func,
                                               policy_levers=policy_levers,
                                               weights = (MINIMIZE,)*2,
                                               nr_of_generations=20,
                                               algorithm=epsNSGA2,
                                               pop_size=4,
                                               crossover_rate=0.5, 
                                               mutation_rate=0.02,
                                               caching=True,
                                               eps=[0.01, 0.01]
                                               )
