from __future__ import division
'''
Created on 3 feb. 2011

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
                chamarat <c.hamarat  (at) tudelft (dot) nl>
                epruyt <e.pruyt (at) tudelft (dot) nl>
'''
import matplotlib.pyplot as plt

from expWorkbench import ModelEnsemble, CategoricalUncertainty,\
                         ParameterUncertainty, Outcome
from expWorkbench import ema_logging
from expWorkbench.vensim import VensimModelStructureInterface

from analysis.plotting import envelopes

SVN_ID = '$Id: energytrans_example.py 1056 2012-12-14 11:23:14Z jhkwakkel $'

class EnergyTrans(VensimModelStructureInterface):
    modelFile = r'\CESUN_adaptive.vpm'
    
    #outcomes    
    outcomes = [Outcome('total fraction new technologies' , time=True),  
                Outcome('total capacity installed' , time=True)  
                ]
    
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
        '''initializes the model'''
        
        try:
            self.modelFile = policy['file']
        except KeyError:
            ema_logging.warning("key 'file' not found in policy")
        super(EnergyTrans, self).model_init(policy, kwargs)

    def run_model(self, case):
        switches = case.pop("preference switches")

        case["SWITCH preference for MIC"] = switches[0]
        case["SWITCH preference for expected cost per MWe"]= switches[1]
        case["SWITCH preference against unknown"]= switches[2]
        case["SWITCH preference for expected progress"]= switches[3]
        case["SWITCH preference against specific CO2 emissions"]= switches[4]
            
        super(EnergyTrans, self).run_model(case)


if __name__ == "__main__":
    logger = ema_logging.log_to_stderr(ema_logging.INFO)
    model = EnergyTrans(r'..\..\models\EnergyTrans', "ESDMAElecTrans")
    model.step = 4 #reduce data to be stored
    ensemble = ModelEnsemble()
    ensemble.set_model_structure(model)
    
    policies = [{'name': 'no policy',
                 'file': r'\ESDMAElecTrans_NoPolicy.vpm'},
                {'name': 'basic policy',
                 'file': r'\ESDMAElecTrans_basic_policy.vpm'},
                {'name': 'tech2',
                 'file': r'\ESDMAElecTrans_tech2.vpm'},
                {'name': 'econ',
                 'file': r'\ESDMAElecTrans_econ.vpm'},
                {'name': 'adaptive policy',
                 'file': r'\ESDMAElecTrans_adaptive_policy.vpm'},
                {'name': 'ap with op',
                 'file': r'\ESDMAElecTrans_ap_with_op.vpm'},
                ]
    ensemble.add_policies(policies)

    #turn on parallel
    ensemble.parallel = True
    
    #run policy with old cases
    results = ensemble.perform_experiments(10)

    envelopes(results, column='policy')
    plt.show()


