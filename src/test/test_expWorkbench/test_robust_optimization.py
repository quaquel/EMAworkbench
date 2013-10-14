'''
Created on Mar 1, 2012

@author: jhkwakkel
'''
from __future__ import division

import numpy as np

from expWorkbench.uncertainties import CategoricalUncertainty, ParameterUncertainty
from expWorkbench import ema_logging
from connectors import vensim
from connectors.vensim import VensimModelStructureInterface, VensimError
from expWorkbench.outcomes import Outcome
from expWorkbench.ema_exceptions import CaseError 

from expWorkbench import ModelEnsemble, MAXIMIZE, save_optimization_results

class EnergyTrans(VensimModelStructureInterface):
    model_file = r'\CESUN_optimized_new.vpm'
    
    #outcomes    
    outcomes = [Outcome('total fraction new technologies' , time=True),  
                Outcome('total capacity installed' , time=True),  
                Outcome("monitor for Trigger subsidy T2", time=True),
                Outcome("monitor for Trigger subsidy T3", time=True), 
                Outcome("monitor for Trigger subsidy T4", time=True),
                Outcome("monitor for Trigger addnewcom", time=True)]
    
    activation = [Outcome("monitor for Trigger subsidy T2", time=True),
                  Outcome("monitor for Trigger subsidy T3", time=True), 
                  Outcome("monitor for Trigger subsidy T4", time=True),
                  Outcome("monitor for Trigger addnewcom", time=True)]

    uncertainties = [ParameterUncertainty((14000,16000), "ini cap T1"),
                     ParameterUncertainty((1,2), "ini cap T2"),
                     ParameterUncertainty((1,2), "ini cap T3"),
                     ParameterUncertainty((1,2), "ini cap T4"),
                     ParameterUncertainty((500000,1500000), "ini cost T1"), #1000000
                     ParameterUncertainty((5000000,10000000), "ini cost T2"), #8000000
                     ParameterUncertainty((5000000,10000000), "ini cost T3"), #8000000
                     ParameterUncertainty((5000000,10000000), "ini cost T4"), #8000000
                     ParameterUncertainty((5000000,10000000), "ini cum decom cap T1"), 
                     ParameterUncertainty((1,100), "ini cum decom cap T2"), 
                     ParameterUncertainty((1,100), "ini cum decom cap T3"), 
                     ParameterUncertainty((1,100), "ini cum decom cap T4"), 
                     ParameterUncertainty((1,5), "average planning and construction period T1"), 
                     ParameterUncertainty((1,5), "average planning and construction period T2"), 
                     ParameterUncertainty((1,5), "average planning and construction period T3"), 
                     ParameterUncertainty((1,5), "average planning and construction period T4"), 
                     ParameterUncertainty((0.85,0.95), "ini PR T1"),
                     ParameterUncertainty((0.7,0.95), "ini PR T2"),
                     ParameterUncertainty((0.7,0.95), "ini PR T3"), 
                     ParameterUncertainty((0.7,0.95), "ini PR T4"), 

                     ParameterUncertainty((30,50), "lifetime T1"),
                     ParameterUncertainty((15,20), "lifetime T2"),
                     ParameterUncertainty((15,20), "lifetime T3"),
                     ParameterUncertainty((15,20), "lifetime T4"),      

                     #One uncertain development over time -- smoothed afterwards
                     ParameterUncertainty((0.03,0.035), "ec gr t1"), #0.03                        
                     ParameterUncertainty((-0.01,0.03), "ec gr t2"), #0.03
                     ParameterUncertainty((-0.01,0.03), "ec gr t3"), #0.03
                     ParameterUncertainty((-0.01,0.03), "ec gr t4"), #0.03
                     ParameterUncertainty((-0.01,0.03), "ec gr t5"), #0.03
                     ParameterUncertainty((-0.01,0.03), "ec gr t6"), #0.03                        
                     ParameterUncertainty((-0.01,0.03), "ec gr t7"), #0.03
                     ParameterUncertainty((-0.01,0.03), "ec gr t8"), #0.03
                     ParameterUncertainty((-0.01,0.03), "ec gr t9"),#0.03
                     ParameterUncertainty((-0.01,0.03), "ec gr t10"), #0.03                
                    
                     #Uncertainties in Random Functions
                     ParameterUncertainty((0.9,1), "random PR min"),        
                     ParameterUncertainty((1,1.1), "random PR max"),
                     ParameterUncertainty((1,100), "seed PR T1", integer=True), 
                     ParameterUncertainty((1,100), "seed PR T2", integer=True),
                     ParameterUncertainty((1,100), "seed PR T3", integer=True),
                     ParameterUncertainty((1,100), "seed PR T4", integer=True),
            
                     #Uncertainties in Preference Functions
                     ParameterUncertainty((2,5), "absolute preference for MIC"),
                     ParameterUncertainty((1,3), "absolute preference for expected cost per MWe"),
                     ParameterUncertainty((2,5), "absolute preference against unknown"),  
                     ParameterUncertainty((1,3), "absolute preference for expected progress"),
                     ParameterUncertainty((2,5), "absolute preference against specific CO2 emissions"),  
                     
                     #Uncertainties DIE NOG AANGEPAST MOETEN WORDEN
                     ParameterUncertainty((1,2), "performance expected cost per MWe T1"),
                     ParameterUncertainty((1,5), "performance expected cost per MWe T2"),
                     ParameterUncertainty((1,5), "performance expected cost per MWe T3"),
                     ParameterUncertainty((1,5), "performance expected cost per MWe T4"),
                     ParameterUncertainty((4,5), "performance CO2 avoidance T1"),
                     ParameterUncertainty((1,5), "performance CO2 avoidance T2"),
                     ParameterUncertainty((1,5), "performance CO2 avoidance T3"),
                     ParameterUncertainty((1,5), "performance CO2 avoidance T4"),
                    
                     #Switches op technologies
                     CategoricalUncertainty((0,1), "SWITCH T3", default = 1),
                     CategoricalUncertainty((0,1), "SWITCH T4", default = 1),

                     CategoricalUncertainty([(0, 0, 0, 0, 1),
                                             (0, 0, 0, 1, 0),
                                             (0, 0, 0, 1, 1),
                                             (0, 0, 1, 0, 0),
                                             (0, 0, 1, 0, 1),
                                             (0, 0, 1, 1, 0),
                                             (0, 0, 1, 1, 1),
                                             (0, 1, 0, 0, 0),
                                             (0, 1, 0, 0, 1),
                                             (0, 1, 0, 1, 0),
                                             (0, 1, 0, 1, 1),
                                             (0, 1, 1, 0, 0),
                                             (0, 1, 1, 0, 1),
                                             (0, 1, 1, 1, 0),
                                             (0, 1, 1, 1, 1),
                                             (1, 0, 0, 0, 0),
                                             (1, 0, 0, 0, 1),
                                             (1, 0, 0, 1, 0),
                                             (1, 0, 0, 1, 1),
                                             (1, 0, 1, 0, 0),
                                             (1, 0, 1, 0, 1),
                                             (1, 0, 1, 1, 0),
                                             (1, 0, 1, 1, 1),
                                             (1, 1, 0, 0, 0),
                                             (1, 1, 0, 0, 1),
                                             (1, 1, 0, 1, 0),
                                             (1, 1, 0, 1, 1),
                                             (1, 1, 1, 0, 0),
                                             (1, 1, 1, 0, 1),
                                             (1, 1, 1, 1, 0),
                                             (1, 1, 1, 1, 1)], 
                                            "preference switches"),
                     ]
       
       
    def model_init(self, policy, kwargs):
        super(EnergyTrans, self).model_init(policy, kwargs)

        #pop name
        policy.pop('name')
        self.policy = policy

    def run_model(self, case):
        
        for key, value in self.policy.items():
            vensim.set_value(key, value)
        
        switches = case.pop("preference switches")

        case["SWITCH preference for MIC"] = switches[0]
        case["SWITCH preference for expected cost per MWe"]= switches[1]
        case["SWITCH preference against unknown"]= switches[2]
        case["SWITCH preference for expected progress"]= switches[3]
        case["SWITCH preference against specific CO2 emissions"]= switches[4]
            
        for key, value in case.items():
            vensim.set_value(key, value)
        ema_logging.debug("model parameters set successfully")
        
        ema_logging.debug("run simulation, results stored in " + self.working_directory+self.result_file)
        try:
            vensim.run_simulation(self.working_directory+self.result_file)
        except VensimError:
            raise

        results = {}
        error = False
        for output in self.outcomes:
            ema_logging.debug("getting data for %s" %output.name)
            result = vensim.get_data(self.working_directory+self.result_file, 
                              output.name 
                              )
            ema_logging.debug("successfully retrieved data for %s" %output.name)
            if not result == []:
                if result.shape[0] != self.run_length:
                    a = np.zeros((self.run_length))
                    a[0:result.shape[0]] = result
                    result = a
                    error = True
            
            else:
                result = result[0::self.step]
            try:
                results[output.name] = result
            except ValueError:
                print "what"

        a = results.keys()
        for output in self.activation:
            value = results[output.name]
            time = results["TIME"]
            activationTimeStep = time[value>0]
            if activationTimeStep.shape[0] > 0:
                activationTimeStep = activationTimeStep[0]
            else:
                activationTimeStep = np.array([0])
            results[output.name] = activationTimeStep
            
        
        self.output = results   
        if error:
            raise CaseError("run not completed", case) 

def obj_func(outcomes):
    outcome = outcomes['total fraction new technologies']
    zeros = np.zeros((outcome.shape[0], 1))
    zeros[outcome[:,-1]>0.6] = 1
    value = np.sum(zeros)/zeros.shape[0] 
    return value,

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)
    model = EnergyTrans(r"..\data", "ESDMAElecTrans")
       
    ensemble = ModelEnsemble()
    ensemble.set_model_structure(model)
#     ensemble.parallel = True
    
    policy_levers = {'Trigger subsidy T2': {'type':'range float', 'values':(0,1)},
                   'Trigger subsidy T3': {'type':'range float', 'values':(0,1)},
                   'Trigger subsidy T4': {'type':'range float', 'values':(0,1)},
                   'Trigger addnewcom': {'type':'list', 'values':[0, 0.25, 0.5, 0.75, 1]}}
    
    stats_callback, pop   = ensemble.perform_robust_optimization(cases=10,
                                               reporting_interval=100,
                                               obj_function=obj_func,
                                               policy_levers=policy_levers,
                                               weights = (MAXIMIZE,),
                                               nr_of_generations=10,
                                               pop_size=10,
                                               crossover_rate=0.5, 
                                               mutation_rate=0.02
                                               )
#     save_optimization_results((stats_callback, pop), '../data/robust test.bz2')