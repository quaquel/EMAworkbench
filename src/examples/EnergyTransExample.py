from __future__ import division
'''
Created on 3 feb. 2011
@authors: epruyt, chamarat, and jkwakkel
'''
from expWorkbench import SimpleModelEnsemble, CategoricalUncertainty,\
                         ParameterUncertainty, save_results, Outcome

import expWorkbench.EMAlogging as logging
from expWorkbench.vensim import VensimModelStructureInterface

class EnergyTrans(VensimModelStructureInterface):


    modelFile = r'\ESDMAElecTrans_NoPolicy.vpm'
    
    #outcomes
    outcomes = [Outcome('total fraction new technologies' , time=True),
                Outcome('installed capacity T1' , time=True),
                Outcome('installed capacity T2' , time=True),
                Outcome('installed capacity T3' , time=True),
                Outcome('installed capacity T4' , time=True),
                Outcome('total capacity installed' , time=True)]
#        self.outcomes.append(Outcome('actual planning time T4' , time=True),
#        self.outcomes.append(Outcome('actual planning time T3' , time=True),
#        self.outcomes.append(Outcome('actual planning time T2' , time=True),
#        self.outcomes.append(Outcome('actual planning time T1' , time=True),
        
    #Initial values
    uncertainties = [ParameterUncertainty((14000,16000), "ini cap T1"),
                     ParameterUncertainty((1,2), "ini cap T2"),
                     ParameterUncertainty((1,2), "ini cap T3"),
                     ParameterUncertainty((1,2), "ini cap T4"),
                     ParameterUncertainty((500000,1500000), "ini cost T1"),
                     ParameterUncertainty((5000000,10000000), "ini cost T2"),
                     ParameterUncertainty((5000000,10000000), "ini cost T3"),
                     ParameterUncertainty((5000000,10000000), "ini cost T4"),
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
                                     
                    #Plain Parametric Uncertainties 
                     ParameterUncertainty((30,50), "lifetime T1"),
                     ParameterUncertainty((15,40), "lifetime T2"),
                     ParameterUncertainty((15,40), "lifetime T3"),
                     ParameterUncertainty((15,40), "lifetime T4"),        
    
                    #One uncertain development over time -- smoothed afterwards
                     ParameterUncertainty((0.03,0.035), "ec gr t1"), #0.03                        
                     ParameterUncertainty((-0.01,0.03), "ec gr t2"), #0.03
                     ParameterUncertainty((-0.01,0.03), "ec gr t3"), #0.03
                     ParameterUncertainty((-0.01,0.03), "ec gr t4"), #0.03
                     ParameterUncertainty((-0.01,0.03), "ec gr t5"), #0.03
                     ParameterUncertainty((-0.01,0.03), "ec gr t6"), #0.03                        
                     ParameterUncertainty((-0.01,0.03), "ec gr t7"), #0.03
                     ParameterUncertainty((-0.01,0.03), "ec gr t8"), #0.03
                     ParameterUncertainty((-0.01,0.03), "ec gr t9"), #0.03
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
            #TOEVOEGEN SWITCHES ZODAT BOVENSTAANDE CRITERIA WEL OF NIET GEBRUIKT WORDEN...
                     CategoricalUncertainty((0,1), "SWITCH preference for MIC", default = 1),
                     CategoricalUncertainty((0,1), "SWITCH preference for expected cost per MWe", default = 0),
                     CategoricalUncertainty((0,1), "SWITCH preference against unknown", default = 0),
                     CategoricalUncertainty((0,1), "SWITCH preference for expected progress", default = 0),
                     CategoricalUncertainty((0,1), "SWITCH preference against specific CO2 emissions", default = 0),
            #Uncertainties DIE NOG AANGEPAST MOETEN WORDEN
                     ParameterUncertainty((1,2), "performance expected cost per MWe T1"),
                     ParameterUncertainty((1,5), "performance expected cost per MWe T2"),
                     ParameterUncertainty((1,5), "performance expected cost per MWe T3"),
                     ParameterUncertainty((1,5), "performance expected cost per MWe T4"),
                     ParameterUncertainty((4,5), "performance CO2 avoidance T1"),
                     ParameterUncertainty((1,5), "performance CO2 avoidance T2"),
                     ParameterUncertainty((1,5), "performance CO2 avoidance T3"),
                     ParameterUncertainty((1,5), "performance CO2 avoidance T4"),
                    
            #        #Switches op technologies
                     ParameterUncertainty((0,1), "SWITCH T3", integer=True),
                     ParameterUncertainty((0,1), "SWITCH T4", integer=True),
                   
            #        #ORDERS OF DELAYS
                     CategoricalUncertainty((1,3,10,1000), "order lifetime T1", default = 3),
                     CategoricalUncertainty((1,3,10,1000), "order lifetime T2", default = 3),
                     CategoricalUncertainty((1,3,10,1000), "order lifetime T3", default = 3),
                     CategoricalUncertainty((1,3,10,1000), "order lifetime T4", default = 3)]

    def model_init(self, policy, kwargs):
        try:
            self.modelFile = policy['file']
        except:
            logging.debug("no policy specified")
        super(EnergyTrans, self).model_init(policy, kwargs)


if __name__ == "__main__":
    logger = logging.log_to_stderr(logging.INFO)
    model = EnergyTrans(r'..\..\models\EnergyTrans', "ESDMAElecTrans")
    model.step = 4 #reduce data to be stored
    ensemble = SimpleModelEnsemble()
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
    results = ensemble.perform_experiments(100)

    save_results(results, r'C:\workspace\EMA-workbench\src\analysis\eng_trans_100.cPickle')
    


