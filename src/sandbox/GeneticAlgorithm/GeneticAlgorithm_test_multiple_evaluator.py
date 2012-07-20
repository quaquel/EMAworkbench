from __future__ import division
'''
Created on 3 feb. 2011
@authors: epruyt, chamarat, and jkwakkel
'''

import expWorkbench.EMAlogging as logging
from expWorkbench.model import SimpleModelEnsemble
from expWorkbench.uncertainties import CategoricalUncertainty, ParameterUncertainty
from expWorkbench import EMAlogging
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

        self.modelFile = r'\ESDMAElecTrans_adaptive_policy.vpm'
    
        #outcomes    
        self.outcomes.append(Outcome('total fraction new technologies' , time=True))  
        self.outcomes.append(Outcome('total capacity installed' , time=True))  
           
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
       
##        #ORDERS OF DELAYS
#        self.uncertainties.append(CategoricalUncertainty((1,3,10,1000), "order lifetime T1", default = 3))
#        self.uncertainties.append(CategoricalUncertainty((1,3,10,1000), "order lifetime T2", default = 3))
#        self.uncertainties.append(CategoricalUncertainty((1,3,10,1000), "order lifetime T3", default = 3))
#        self.uncertainties.append(CategoricalUncertainty((1,3,10,1000), "order lifetime T4", default = 3))

    def model_init(self, policy, kwargs):
        try:
            self.modelFile = policy['file']
        except:
            EMAlogging.debug("no policy specified")
        super(EnergyTrans, self).model_init(policy, kwargs)


if __name__ == "__main__":
#    logger = logging.log_to_stderr(logging.INFO)
    import cPickle
    model = EnergyTrans(r"..\..\..\models\EnergyTrans", "ESDMAElecTrans")
    ensemble = SimpleModelEnsemble()
    ensemble.set_model_structure(model)
    
    storeduncertainties = []
       
    def eval_func(chromosome):
        
        cases = []
        case = {}
        for i, uncertainty in enumerate(uncertainties):
            case[uncertainty.name] = chromosome[i]
        cases.append(case)

        results = ensemble.perform_experiments(cases)
        score = results[1]['total fraction new technologies'][0, -1]
        score2 = results[1]['total capacity installed'][0, -1]/28800000
        if score2 < 0:
            score2 = 0
        elif score2 >1:
            score2 = 1 
            
        if score == 1:
            mylist = []
            for i in range(len(uncertainties)):
                mylist.append(chromosome[i])
            storeduncertainties.append(mylist)
        
        return (score + score2)/2
        
    from pyevolve import G1DList #@UnresolvedImport
    from pyevolve import GSimpleGA #@UnresolvedImport
    from pyevolve import GAllele #@UnresolvedImport
    from pyevolve import Initializators #@UnresolvedImport
    from pyevolve import Crossovers #@UnresolvedImport
    from pyevolve import Mutators #@UnresolvedImport
    from pyevolve import DBAdapters #@UnresolvedImport
    
    setofAlleles = GAllele.GAlleles()
    
    uncertainties = model.get_model_uncertainties()
    
    for uncertainty in uncertainties:
        if type(uncertainty) == ParameterUncertainty:
            upperbound = uncertainty.values[1]
            lowerbound = uncertainty.values[0]
            allele = GAllele.GAlleleRange(lowerbound, upperbound, real=True)
            setofAlleles.add(allele)
        elif type(uncertainty) == CategoricalUncertainty:
            allele = GAllele.GAlleleRange(0, 1)
            setofAlleles.add(allele)
            
    genome = G1DList.G1DList(len(uncertainties))
    genome.setParams(allele=setofAlleles)
    genome.initializator.set(Initializators.G1DListInitializatorAllele)
    
    genome.evaluator.set(eval_func)
#    genome.evaluator.add(eval_func2)
    
    genome.crossover.set(Crossovers.G1DListCrossoverTwoPoint)
    genome.mutator.set(Mutators.G1DListMutatorAllele)
    ga = GSimpleGA.GSimpleGA(genome)
    
    ga.setGenerations(100)
    ga.setCrossoverRate(0.75)
    ga.setMutationRate(0.05)
    ga.setPopulationSize(100)
    ga.terminationCriteria.set(GSimpleGA.RawStatsCriteria)
        
    ga.evolve()
    print ga.bestIndividual()

    ga.printTimeElapsed()
    cPickle.dump(storeduncertainties, open("D:\Workspace\EMAworkbench\src\sandbox\GeneticAlgorithm\storeduncertainties.cPickle", 'wb'), protocol=2)

