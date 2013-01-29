'''
Created on Jun 26, 2012

@author: sibeleker
'''
from expWorkbench import Outcome, save_results
from modelEnsemble import ModelEnsemble
import expWorkbench.EMAlogging as logging
from uncertainties import ParameterUncertainty, CategoricalUncertainty, LookupUncertainty

from vensim import VensimModelStructureInterface
from expWorkbench import load_results
import matplotlib.pyplot as plt
from analysis.graphs import lines, envelopes
from expWorkbench import vensimDLLwrapper

class lookup_model(VensimModelStructureInterface): 
    def __init__(self, workingDirectory, name):
        self.modelFile = r'\sampleModel.vpm'
        super(lookup_model, self).__init__(workingDirectory, name)

       # vensim.load_model(self.modelFile)
        self.outcomes = [Outcome('TF2', time=True),
                         Outcome('flow1', time=True),
                         Outcome('TF', time=True),
                         Outcome('TF3', time=True)]
 
        '''
        each lookup uncertainty defined and added to the uncertainties list must be deleted immediately. it is not possible to do that in the constructor of lookups.
        or i can delete it later before generating the cases.
        '''
        self.uncertainties.append(LookupUncertainty([(0, 5), (-1, 2), (0, 1.5), (0, 1.5), (0, 1), (0.5, 1.5)], "TF", 'hearne', self, 0, 2))
        #self.uncertainties.pop()
        self.uncertainties.append(LookupUncertainty([(0, 4), (1, 5), (1, 5), (0, 2), (0, 2)], "TF2", 'approximation', self, 0, 10))
        #self.uncertainties.pop()
        self.uncertainties.append(ParameterUncertainty((0.02, 0.08), "rate1"))
        self.uncertainties.append(ParameterUncertainty((0.02, 0.08), "rate2"))
        self.uncertainties.append(LookupUncertainty([[(0.0, 0.05), (0.25, 0.15), (0.5, 0.4), (0.75, 1), (1, 1.25)], 
                                                     [(0.0, 0.1), (0.25, 0.25), (0.5, 0.75), (1, 1.25)],
                                                     [(0.0, 0.0), (0.1, 0.2), (0.3, 0.6), (0.6, 0.9), (1, 1.25)]], "TF3", 'categories', self, 0, 2))
        #self.uncertainties.pop()   
        self.delete_lookup_uncertainties()                   

        
if __name__ == "__main__":
    logger = logging.log_to_stderr(logging.INFO)
    model = lookup_model(r'..\lookups', "sampleModel")

    #model.step = 4 #reduce data to be stored
    ensemble = ModelEnsemble()
    ensemble.set_model_structure(model)

    #turn on parallel
    ensemble.parallel = False
    
    #run policy with old cases
    results = ensemble.perform_experiments(10)
    save_results(results, 'lookup_3methods.cpickle')
    
    results = load_results('lookup_3methods.cpickle')
    outcomes =['TF', 'TF2', 'TF3', 'flow1']
    lines(results, outcomes, density=True, hist=True)
    plt.show()  