
#from pyevolve import G1DList
#from pyevolve import GSimpleGA
#
#def eval_func(chromosome):
#    score = 0.0
#    # iterate over the chromosome
#    for value in chromosome:
#        if value==0:
#            score += 1
#    return score
#
#genome = G1DList.G1DList(2)
#genome.evaluator.set(eval_func)
#ga = GSimpleGA.GSimpleGA(genome)
#ga.evolve(freq_stats=10)
#print ga.bestIndividual()

from __future__ import division
'''
Created on 3 feb. 2011
@authors: epruyt, chamarat, and jkwakkel
'''

import copy
import expWorkbench.EMAlogging as logging
from expWorkbench.model import SimpleModelEnsemble
from expWorkbench.uncertainties import CategoricalUncertainty, ParameterUncertainty
from expWorkbench import EMAlogging, vensim
from analysis.graphs import TIME
from expWorkbench.vensim import VensimModelStructureInterface
from expWorkbench.outcomes import Outcome
from analysis.graphs import lines, envelopes
import matplotlib.pyplot as plt
from expWorkbench.util import save_results, load_results
from analysis import prim
import numpy as np

class EnergyTrans(VensimModelStructureInterface):
    def __init__(self, workingDirectory, name):
        """interface to the model"""
        super(EnergyTrans, self).__init__(workingDirectory, name )

        self.modelFile = r'\ESDMAElecTrans_PolicyCombined.vpm'
    
        #outcomes    
        self.outcomes.append(Outcome('total fraction new technologies' , time=True))  
        self.outcomes.append(Outcome('total capacity installed' , time=True))  
        self.outcomes.append(Outcome('installed capacity T1' , time=True))  
        self.outcomes.append(Outcome('installed capacity T2' , time=True))  
        self.outcomes.append(Outcome('installed capacity T3' , time=True))  
        self.outcomes.append(Outcome('installed capacity T4' , time=True))  
           
        #Initial values
        self.uncertainties.append(ParameterUncertainty((14000,16000), "ini cap T1")) #
        self.uncertainties.append(ParameterUncertainty((1,2), "ini cap T2")) #
        self.uncertainties.append(ParameterUncertainty((1,2), "ini cap T3")) #
        self.uncertainties.append(ParameterUncertainty((1,2), "ini cap T4")) #
        self.uncertainties.append(ParameterUncertainty((500000,1500000), "ini cost T1")) #1000000
        self.uncertainties.append(ParameterUncertainty((5000000,10000000), "ini cost T2")) #8000000
        self.uncertainties.append(ParameterUncertainty((5000000,10000000), "ini cost T3")) #8000000
        self.uncertainties.append(ParameterUncertainty((5000000,10000000), "ini cost T4")) #8000000
        self.uncertainties.append(ParameterUncertainty((5000000,10000000), "ini cum decom cap T1")) 
        self.uncertainties.append(ParameterUncertainty((1,100), "ini cum decom cap T2")) 
        self.uncertainties.append(ParameterUncertainty((1,100), "ini cum decom cap T3")) 
        self.uncertainties.append(ParameterUncertainty((1,100), "ini cum decom cap T4")) 
        self.uncertainties.append(ParameterUncertainty((1,5), "average planning and construction period T1")) 
        self.uncertainties.append(ParameterUncertainty((1,5), "average planning and construction period T2")) 
        self.uncertainties.append(ParameterUncertainty((1,5), "average planning and construction period T3")) 
        self.uncertainties.append(ParameterUncertainty((1,5), "average planning and construction period T4")) 
        self.uncertainties.append(ParameterUncertainty((0.85,0.95), "ini PR T1")) 
        self.uncertainties.append(ParameterUncertainty((0.7,0.95), "ini PR T2")) 
        self.uncertainties.append(ParameterUncertainty((0.7,0.95), "ini PR T3")) 
        self.uncertainties.append(ParameterUncertainty((0.7,0.95), "ini PR T4")) 
        
        #Plain Parametric Uncertainties 
        self.uncertainties.append(ParameterUncertainty((30,50), "lifetime T1"))
        self.uncertainties.append(ParameterUncertainty((15,20), "lifetime T2"))
        self.uncertainties.append(ParameterUncertainty((15,20), "lifetime T3"))
        self.uncertainties.append(ParameterUncertainty((15,20), "lifetime T4"))        
#        
#        #One uncertain development over time -- smoothed afterwards
        self.uncertainties.append(ParameterUncertainty((0.03,0.035), "ec gr t1")) #0.03                        
        self.uncertainties.append(ParameterUncertainty((-0.01,0.03), "ec gr t2")) #0.03
        self.uncertainties.append(ParameterUncertainty((-0.01,0.03), "ec gr t3")) #0.03
        self.uncertainties.append(ParameterUncertainty((-0.01,0.03), "ec gr t4")) #0.03
        self.uncertainties.append(ParameterUncertainty((-0.01,0.03), "ec gr t5")) #0.03
        self.uncertainties.append(ParameterUncertainty((-0.01,0.03), "ec gr t6")) #0.03                        
        self.uncertainties.append(ParameterUncertainty((-0.01,0.03), "ec gr t7")) #0.03
        self.uncertainties.append(ParameterUncertainty((-0.01,0.03), "ec gr t8")) #0.03
        self.uncertainties.append(ParameterUncertainty((-0.01,0.03), "ec gr t9")) #0.03
        self.uncertainties.append(ParameterUncertainty((-0.01,0.03), "ec gr t10")) #0.03                
        
        #Uncertainties in Random Functions
        self.uncertainties.append(ParameterUncertainty((0.9,1), "random PR min"))        
        self.uncertainties.append(ParameterUncertainty((1,1.1), "random PR max")) 
        self.uncertainties.append(ParameterUncertainty((1,100), "seed PR T1", integer=True)) 
        self.uncertainties.append(ParameterUncertainty((1,100), "seed PR T2", integer=True))
        self.uncertainties.append(ParameterUncertainty((1,100), "seed PR T3", integer=True))
        self.uncertainties.append(ParameterUncertainty((1,100), "seed PR T4", integer=True))

#Uncertainties in Preference Functions
        self.uncertainties.append(ParameterUncertainty((2,5), "absolute preference for MIC"))
        self.uncertainties.append(ParameterUncertainty((1,3), "absolute preference for expected cost per MWe"))
        self.uncertainties.append(ParameterUncertainty((2,5), "absolute preference against unknown"))        
        self.uncertainties.append(ParameterUncertainty((1,3), "absolute preference for expected progress"))        
        self.uncertainties.append(ParameterUncertainty((2,5), "absolute preference against specific CO2 emissions"))  
#TOEVOEGEN SWITCHES ZODAT BOVENSTAANDE CRITERIA WEL OF NIET GEBRUIKT WORDEN...
        self.uncertainties.append(CategoricalUncertainty((0,1), "SWITCH preference for MIC", default = 1))
        self.uncertainties.append(CategoricalUncertainty((0,1), "SWITCH preference for expected cost per MWe", default = 0))
        self.uncertainties.append(CategoricalUncertainty((0,1), "SWITCH preference against unknown", default = 0))
        self.uncertainties.append(CategoricalUncertainty((0,1), "SWITCH preference for expected progress", default = 0))
        self.uncertainties.append(CategoricalUncertainty((0,1), "SWITCH preference against specific CO2 emissions", default = 0))
#Uncertainties DIE NOG AANGEPAST MOETEN WORDEN
        self.uncertainties.append(ParameterUncertainty((1,2), "performance expected cost per MWe T1"))
        self.uncertainties.append(ParameterUncertainty((1,5), "performance expected cost per MWe T2"))
        self.uncertainties.append(ParameterUncertainty((1,5), "performance expected cost per MWe T3"))
        self.uncertainties.append(ParameterUncertainty((1,5), "performance expected cost per MWe T4"))
        self.uncertainties.append(ParameterUncertainty((4,5), "performance CO2 avoidance T1"))
        self.uncertainties.append(ParameterUncertainty((1,5), "performance CO2 avoidance T2"))
        self.uncertainties.append(ParameterUncertainty((1,5), "performance CO2 avoidance T3"))
        self.uncertainties.append(ParameterUncertainty((1,5), "performance CO2 avoidance T4"))
        
#        #Switches op technologies
        self.uncertainties.append(CategoricalUncertainty((0,1), "SWITCH T3", default = 1))
        self.uncertainties.append(CategoricalUncertainty((0,1), "SWITCH T4", default = 1))
 
    def model_init(self, policy, kwargs):
#        try:
#            self.modelFile = policy['file']
#        except:
#            EMAlogging.debug("no policy specified")
        super(EnergyTrans, self).model_init(policy, kwargs)
        
        #pop name
        policy = copy.copy(policy)
        policy.pop('name')
        
        for key, value in policy.items():
            vensim.set_value(key, value)
        
        
if __name__ == "__main__":
    logger = logging.log_to_stderr(logging.INFO)
    
    model = EnergyTrans(r"..\VensimModels\TFSC", "ESDMAElecTrans")
    ensemble = SimpleModelEnsemble()
    ensemble.set_model_structure(model)
    
    cases, uncertainties = ensemble._generate_cases(1)
    
    valuelist = [15.467089994193 , 18.3948367845855 , 17.5216359599053 , 0.0323513175268276 , 0.0267216806566911 , 0.0252897989265933 , 0.0211748970259063 , 0.0192967619764282 , 0.0298868721235403 , 0.026846492561752 , 0.0282265728603356 , 0.0274643497911105 , 0.0206173186487346 , 0.930953610229856 , 1.05807449426449 , 58.6261672319115 , 1.0959476696141 , 48.4897275078371 , 79.8968117041453 , 2.03012275630195 , 2.33576352581696 , 2.60266175740213 , 1.24700542123355 , 3.06884098418713 , 1 , 0 , 0 , 0 , 0 , 1.45807445678444 , 3.53395235847141 , 1.75257486371618 , 2.9795030911447 , 4.00199168664975 , 1.97473349200058 , 4.74196793795403 , 4.72730891245437 , 0 , 0 , 14826.4074143275 , 1.24609526886412 , 1.18827514220571 , 1.09824115488565 , 1245886.83942348 , 6282282.69560999 , 6118827.67237203 , 9531496.10651471 , 8693813.50295679 , 32.948697875027 , 17.1705785135149 , 13.0971274404015 , 3.74255065304761 , 1.36231655867486 , 1.92101352688469 , 3.8941723138427 , 0.898745338298322 , 0.782806406356795 , 0.817631734201507 , 0.705822656618514 , 43.3820783577107]


    newcases = [] 
    case = {}
    i=0
    for uncertainty in uncertainties:
        print uncertainty.name
        case[uncertainty.name] = valuelist[i]
        i+=1
#    case['desired fraction'] = 1.0
#    uncertainties.add(ParameterUncertainty((0.7,1), "desired fraction"))    
    newcases.append(case)
    
#    uncertainties.append('desired fraction')
        
    results = ensemble.perform_experiments(newcases)
        
    print results[1]['total fraction new technologies'][0, -1]
    print results[1]['total capacity installed'][0, -1]
    
    lines(results)
    
    plt.show()
    
