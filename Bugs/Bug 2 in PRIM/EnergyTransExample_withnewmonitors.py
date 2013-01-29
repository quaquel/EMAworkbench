from __future__ import division
'''
Created on 3 feb. 2011
@authors: epruyt, chamarat, and jkwakkel
'''
import numpy as np
import cPickle
import matplotlib.pyplot as plt

from expWorkbench.model import SimpleModelEnsemble
from expWorkbench import CategoricalUncertainty, ParameterUncertainty, \
                         save_results, Outcome, load_results, vensim, \
                         EMAlogging

import expWorkbench.EMAlogging as logging
from expWorkbench.vensim import VensimModelStructureInterface
from analysis.graphs import lines, envelopes

class EnergyTrans(VensimModelStructureInterface):
    def __init__(self, workingDirectory, name):
        """interface to the model"""
        super(EnergyTrans, self).__init__(workingDirectory, name )

        self.modelFile = r'\CESUN_optimized_new.vpm'
    
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
        self.uncertainties.append(CategoricalUncertainty(([(0, 0, 0, 0, 1),
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
                                             (1, 1, 1, 1, 1)]), "preference switches"))
        
        #switch for addfactor activation
#        self.uncertainties.append(CategoricalUncertainty((0,1), "switchaddfactorco2", default = 1))
       
    def model_init(self, policy, kwargs):
        try:
            self.modelFile = policy['file']
        except:
            logging.debug("no policy specified")
        super(EnergyTrans, self).model_init(policy, kwargs)

    def run_model(self, case):
        switches = case.pop("preference switches")

        case["SWITCH preference for MIC"] = switches[0]
        case["SWITCH preference for expected cost per MWe"]= switches[1]
        case["SWITCH preference against unknown"]= switches[2]
        case["SWITCH preference for expected progress"]= switches[3]
        case["SWITCH preference against specific CO2 emissions"]= switches[4]
        
        if np.sum(switches) == 0:
            print "sifir olan cikti haci!"
            
        for key, value in case.items():
            vensim.set_value(key, value)
        EMAlogging.debug("model parameters set successfully")
        
        EMAlogging.debug("run simulation, results stored in " + self.workingDirectory+self.resultFile)
        try:
            vensim.run_simulation(self.workingDirectory+self.resultFile)
        except VensimError as e:
            raise

        results = {}
        error = False
        for output in self.outcomes:
            EMAlogging.debug("getting data for %s" %output.name)
            result = vensim.get_data(self.workingDirectory+self.resultFile, 
                              output.name 
                              )
            EMAlogging.debug("successfully retrieved data for %s" %output.name)
            if not result == []:
                if result.shape[0] != self.runLength:
                    a = np.zeros((self.runLength))
                    a[0:result.shape[0]] = result
                    result = a
                    error = True
            
            else:
                result = result[0::self.step]
            try:
                results[output.name] = result
            except ValueError as e:
                print "what"

#        for output in self.activation:
#            value = results[output.name]
#            time = results["TIME"]
#            activationTimeStep = time[value>0]
#            if len(activationTimeStep) > 0:
#                activationTimeStep = activationTimeStep[0]
#            else:
#                activationTimeStep = np.array([0])
##            if activationTimeStep.shape[0] > 0:
##                activationTimeStep = activationTimeStep
##            else:
##                activationTimeStep = np.array([0])
#            results[output.name] = activationTimeStep
            
        
        self.output = results   
        if error:
            raise CaseError("run not completed", case) 


if __name__ == "__main__":
    logger = logging.log_to_stderr(logging.INFO)
    model = EnergyTrans(r'..\..\models\CANER\CESUN', "ESDMAElecTrans")
    model.step = 4 #reduce data to be stored
    ensemble = SimpleModelEnsemble()
    ensemble.set_model_structure(model)
    ensemble.parallel = True
    policies = [
                {'name': 'No Policy',
                 'file': r'\CESUN_no.vpm'},
                {'name': 'Basic Policy',
                 'file': r'\CESUN_basic.vpm'},
                {'name': 'Adaptive Policy',
                 'file': r'\CESUN_adaptive.vpm'},
#                {'name': 'optimized adaptive',
#                 'file': r'\CESUN_optimized2.vpm'},
                {'name': 'Optimized Adaptive Policy',
                 'file': r'\CESUN_optimized_new.vpm'}
                
                ]
    ensemble.add_policies(policies)
    
#    results = ensemble.perform_experiments(10000)
#    save_results(results, r'CESUN_optimized_10000_new.cPickle')

#    save_results(results, r'Optimized_results1000.cPickle')

    results = load_results(r'CESUN_optimized_10000_new.cPickle')
    
    outcomes=['total fraction new technologies']

##    Check the fraction of cases below a level
#    experiments, results = results
#
#    #extract results for 1 policy
#    logicalIndex = experiments['policy'] == 'Optimized Adaptive Policy'
#    newExperiments = experiments[ logicalIndex ]
#    newResults = {}
#    for key, value in results.items():
#        newResults[key] = value[logicalIndex]
#    
#    checkarray = newResults['total fraction new technologies']
#    
#    count = 0 
#    for i in range(10000):
#        if checkarray[i,-1]< 0.6:
#            count +=1
#    
#    print count
    
    
#    envelopes(load_results(r'CESUN_4Policies_1000.cPickle'), outcomes=outcomes, 
#              column = 'policy',categories=['no policy', 
#                                            'basic policy',
#                                            'adaptive policy',
#                                            'optimized adaptive'],
#              fill=True, categorieslabels= ['No Policy', 'Basic Policy',
#                                   'Adaptive Policy', 
#                                   'Optimized Adaptive Policy'])
#      
  
    envelopes(results, outcomes=outcomes, column = 'policy', fill=True)

    plt.show()

#    import matplotlib
#    
#    matplotlib.rc('font', weight='bold')
#    matplotlib.rc('font', size=14)
#    
#
##    results = load_results(r'CESUN_optimized_1000.cPickle')
##
#    monitor1 = results[1]['monitor for Trigger subsidy T2']
#    monitor2 = results[1]['monitor for Trigger subsidy T3']
#    monitor3 = results[1]['monitor for Trigger subsidy T4']
#    monitor4 = results[1]['monitor for Trigger addnewcom']
#    
#    mon1 = []
#    mon2 = []
#    mon3 = []
#    mon4 = []
#
#    for i in range(1000):
#        if monitor1[i][0] == 0:
#            pass
##            mon1.append(2000)
#        else: mon1.append(monitor1[i][0])
#        if monitor2[i][0] == 0:
#            pass
##            mon2.append(2000)
#        else: mon2.append(monitor2[i][0])
#        if monitor3[i][0] == 0:
#            pass
##            mon3.append(2000)
#        else: mon3.append(monitor3[i][0])
#        if monitor4[i][0] == 0:
#            pass
##            mon4.append(2000)
#        else: mon4.append(monitor4[i][0])
#
##        mon1.append(monitor1[i][0])
##        mon2.append(monitor2[i][0])
##        mon3.append(monitor3[i][0])
#    
##    plt.plot(monitor1, range(len(monitor1)))
##    x = mon1
##    y = results[1]['total fraction new technologies'][:,-1]
##    plt.scatter(x,y)
##    plt.title('Subsidy for T2')
##    
##    plt.show()
#    x = (mon1,mon2,mon3,mon4)
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    plt.boxplot(x, notch = 0) 
#    labels= ['Subsidy for Tech 2', 'Subsidy for Tech 3','Subsidy for Tech 4', 'Additional commissioning']
#    xtickNames = plt.setp(ax, xticklabels=labels)
#    plt.setp(xtickNames, rotation=45, fontsize=14)
#    plt.show()
    
    