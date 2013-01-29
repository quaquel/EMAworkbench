'''
Created on Mar 16, 2012

@author: chamarat
'''
from __future__ import division
import copy
import cPickle

import numpy as np
import matplotlib.pyplot as plt

from expWorkbench.uncertainties import CategoricalUncertainty, ParameterUncertainty
from expWorkbench import EMAlogging, vensim
from expWorkbench.vensim import VensimModelStructureInterface, VensimError
from expWorkbench.outcomes import Outcome
from expWorkbench.EMAexceptions import CaseError 

from expWorkbench import ModelEnsemble
from analysis.optimization_plots import graph_errorbars_raw, graph_pop_heatmap_raw

class FluModel(VensimModelStructureInterface):
    #base case model
    modelFile = r'\FLU_Adaptive_Policyc.vpm'

    #outcomes
    outcomes = [Outcome('deceased population region 1', time=True)]
#                Outcome('infected fraction region 1', time=True),
#                Outcome('deceased population region 2', time=True),
#                Outcome('infected fraction region 2', time=True)] 
    
    #Plain Parametric Uncertainties 
    uncertainties = [
        ParameterUncertainty((0, 0.5), 
                             "additional seasonal immune population fraction R1"),
        ParameterUncertainty((0, 0.5), 
                             "additional seasonal immune population fraction R2"),
        ParameterUncertainty((0.0001, 0.1), 
                             "fatality ratio region 1"),
        ParameterUncertainty((0.0001, 0.1), 
                             "fatality rate region 2"),
        ParameterUncertainty((0, 0.5), 
                             "initial immune fraction of the population of region 1"),
        ParameterUncertainty((0, 0.5), 
                             "initial immune fraction of the population of region 2"),
        ParameterUncertainty((0, 0.9), 
                             "normal interregional contact rate"),
        ParameterUncertainty((0, 0.5), 
                             "permanent immune population fraction R1"),
        ParameterUncertainty((0, 0.5), 
                             "permanent immune population fraction R2"),
        ParameterUncertainty((0.2, 0.8), 
                             "recovery time region 1"),
        ParameterUncertainty((0.2, 0.8), 
                             "recovery time region 2"),
#        ParameterUncertainty((0.5,2), 
#                             "susceptible to immune population delay time region 1"),
#        ParameterUncertainty((0.5,2), 
#                             "susceptible to immune population delay time region 2"),
        ParameterUncertainty((1, 10), 
                             "root contact rate region 1"),
        ParameterUncertainty((1, 10), 
                             "root contact ratio region 2"),
        ParameterUncertainty((0, 0.1), 
                             "infection rate region 1"),
        ParameterUncertainty((0, 0.1), 
                             "infection rate region 2"),
        ParameterUncertainty((10, 200), 
                             "normal contact rate region 1"),
        ParameterUncertainty((10, 200), 
                             "normal contact rate region 2")]
    
    def model_init(self, policy, kwargs):
            super(FluModel, self).model_init(policy, kwargs)

            #pop name
            policy = copy.copy(policy)
            policy.pop('name')
            
            for key, value in policy.items():
                vensim.set_value(key, value)

    def run_model(self, case):
            
        for key, value in case.items():
            vensim.set_value(key, value)
        EMAlogging.debug("model parameters set successfully")
        
        EMAlogging.debug("run simulation, results stored in " + self.workingDirectory+self.resultFile)
        try:
            vensim.run_simulation(self.workingDirectory+self.resultFile)
        except VensimError:
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
            except ValueError:
                print "what"
    
        self.output = results   
        if error:
            raise CaseError("run not completed", case) 


def obj_func(outcomes):
    outcome = outcomes['deceased population region 1']
    zeros = np.zeros((outcome.shape[0], 1))
    zeros[outcome[:,-1]<1000000] = 1
    value = np.sum(zeros)/zeros.shape[0] 
    return value

if __name__ == "__main__":
#    EMAlogging.log_to_stderr(EMAlogging.INFO)
#    model = FluModel(r'..\..\..\models\CANER\Flu\SD', "fluCase")
#       
#    ensemble = ModelEnsemble()
#    ensemble.set_model_structure(model)
#    ensemble.parallel = True
#    
#    policy_levers = {'trackperiod': (1,8),
#                     'delaytime': (0.01,2)}
#
#    res = ensemble.perform_robust_optimization(cases=1000, 
#                                               obj_function=obj_func, 
#                                               policy_levers = policy_levers,
#                                               nrOfPopMembers=50,
#                                               nrOfGenerations=50,
#                                               crossoverRate=0.7,
#                                               mutationRate=0.01)
#    cPickle.dump(res, open(r'FLU robust optimization results.cPickle', 'w'))
    
    res = cPickle.load(open(r'FLU robust optimization results.cPickle', 'r'))
    graph_pop_heatmap_raw(res['raw'])
    graph_errorbars_raw(res['stats'])
    
    plt.show()