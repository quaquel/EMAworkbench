'''
Created on Mar 16, 2012

@author: chamarat
'''
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

from expWorkbench import EMAlogging
from analysis.optimization_plots import graph_errorbars_raw, graph_pop_heatmap_raw


from expWorkbench import ModelEnsemble, ModelStructureInterface,\
                         ParameterUncertainty, Outcome

class TestModel(ModelStructureInterface):
    

    outcomes = [Outcome('y')]
                
    #specify uncertainties
    uncertainties = [ParameterUncertainty((0, 1), "U1"),
                     ParameterUncertainty((0, 1), "U2")]
   
    def model_init(self, policy, kwargs):
        super(TestModel, self).model_init(policy, kwargs)
        self.policy = policy
    
    def run_model(self, case):
        self.output['y'] = case["U1"] + case["U2"] + self.policy["L1"] + self.policy["L2"]
        
        

def outcome_optimize():
    EMAlogging.log_to_stderr(EMAlogging.INFO)
 
    model = TestModel("", 'simpleModel') #instantiate the model
    ensemble = ModelEnsemble() #instantiate an ensemble
    ensemble.set_model_structure(model) #set the model on the ensemble
    policy = {"name": "test",
              "L1": 1,
              "L2": 1}
    ensemble.add_policy(policy)
    
    def obj_func(results):
        return results['y']        
    
    results = ensemble.perform_outcome_optimization(obj_function=obj_func, 
                                          minimax = 'minimize', 
                                          nrOfGenerations = 1000, 
                                          nrOfPopMembers = 10)

    graph_errorbars_raw(results['stats'])
    plt.show()
    
def robust_optimize():
    EMAlogging.log_to_stderr(EMAlogging.INFO)
 
    model = TestModel("", 'simpleModel') #instantiate the model
    ensemble = ModelEnsemble() #instantiate an ensemble
    ensemble.set_model_structure(model) #set the model on the ensemble
    
    
    policy_levers = { "L1": (0,1),
                      "L2": (0,1)}
    
    def obj_func(results):
        return np.average(results['y'])        
    
    results = ensemble.perform_robust_optimization(cases=1000, 
                                                   obj_function=obj_func, 
                                                   policy_levers=policy_levers, 
                                                   minimax='minimize', 
                                                   nrOfGenerations=50, 
                                                   nrOfPopMembers=20 )
    graph_errorbars_raw(results['stats'])
    plt.show()


def maxmin_optimize():
    EMAlogging.log_to_stderr(EMAlogging.INFO)
 
    model = TestModel("", 'simpleModel') #instantiate the model
    ensemble = ModelEnsemble() #instantiate an ensemble
    ensemble.set_model_structure(model) #set the model on the ensemble
    ensemble.parallel = True
    ensemble.processes = 12
    
    def obj_function1(outcomes):
        return outcomes['y']
    
    policy_levers = { "L1": (0,1),
                      "L2": (0,1)}
    
    
    
    results = ensemble.perform_maximin_optimization(obj_function1 = obj_function1, 
                                                policy_levers = policy_levers,
                                                minimax1='minimize',
                                                nrOfGenerations1=50,
                                                nrOfPopMembers1=200,
                                                minimax2 = "maximize",                                   
                                                nrOfGenerations2 = 50,
                                                nrOfPopMembers2 = 100,
                                                )

    graph_errorbars_raw(results['stats'])
    plt.show()

    
    
def visualize_optimization():
    import cPickle
    results = cPickle.load(open("..\..\..\models\CANER\Flu\SD\maximintest.cPickle"))
#    best_case, best_individual_score, results = results
    graph_errorbars_raw(results['stats'])
    graph_pop_heatmap_raw(results['raw'])
    plt.show()

    
if __name__ == '__main__':
#    outcome_optimize()
#    robust_optimize()
    maxmin_optimize()
#    visualize_optimization()
