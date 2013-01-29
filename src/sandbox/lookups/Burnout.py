'''
Created on Oct 1, 2012

@author: sibeleker
'''
from expWorkbench import Outcome, save_results
from modelEnsemble import ModelEnsemble
import expWorkbench.EMAlogging as logging
from uncertainties import  LookupUncertainty

from vensim import VensimModelStructureInterface
from expWorkbench import load_results
import matplotlib.pyplot as plt
from analysis.graphs import lines, envelopes
from expWorkbench import vensimDLLwrapper

class Burnout(VensimModelStructureInterface): 
    def __init__(self, workingDirectory, name):
        self.modelFile = r'\BURNOUT.vpm'
        super(Burnout, self).__init__(workingDirectory, name)


        self.outcomes = [Outcome('Accomplishments to Date', time=True),
                         Outcome('Energy Level', time=True),
                         Outcome('Hours Worked Per Week', time=True),
                         
                         Outcome('accomplishments per hour lookup', time=True),
                         Outcome('fractional change in expectations from perceived adequacy lookup', time=True),
                         Outcome('effect of perceived adequacy on energy drain lookup', time=True),
                         Outcome('effect of perceived adequacy of hours worked lookup', time=True),
                         Outcome('effect of energy levels on hours worked lookup', time=True),
                         Outcome('effect of high energy on further recovery lookup', time=True),
                         Outcome('effect of hours worked on energy recovery lookup', time=True),
                         Outcome('effect of hours worked on energy drain lookup', time=True),
                         Outcome('effect of low energy on further depletion lookup', time=True)]
 

        
#        self.uncertainties.append(LookupUncertainty([(-1, 3), (-2, 1), (0, 0.9), (0.1, 1), (0.99, 1.01), (0.99, 1.01)], "accomplishments per hour lookup", 'hearne', self, 0, 1))
#        self.uncertainties.append(LookupUncertainty([(-0.75, 0.75), (-0.75, 0.75), (0, 1.5), (0.1, 1.6), (-0.3, 1.5), (0.25, 2.5)], "fractional change in expectations from perceived adequacy lookup", 'hearne', self, -1, 1))
#        self.uncertainties.append(LookupUncertainty([(-2, 2), (-1, 2), (0, 1.5), (0.1, 1.6), (0.5, 2), (0.5, 2)], "effect of perceived adequacy on energy drain lookup", 'hearne', self, 0, 10))
#        self.uncertainties.append(LookupUncertainty([(-2, 2), (-1, 2), (0, 1.5), (0.1, 1.6), (0.5, 1.5), (0.1, 2)], "effect of perceived adequacy of hours worked lookup", 'hearne', self, 0, 2.5))
#        self.uncertainties.append(LookupUncertainty([(-1, 1), (-1, 1), (0, 0.9), (0.1, 1), (0.5, 1.5), (1, 1.5)], "effect of energy levels on hours worked lookup", 'hearne', self, 0, 1.5))
#        self.uncertainties.append(LookupUncertainty([(-1, 1), (-1, 1), (0, 0.9), (0.1, 1), (0.5, 1.5), (1, 1.5)], "effect of high energy on further recovery lookup", 'hearne', self, 0, 1.25))
#        self.uncertainties.append(LookupUncertainty([(-2, 2), (-1, 1), (0, 100), (20, 120), (0.5, 1.5), (0.5, 2)], "effect of hours worked on energy recovery lookup", 'hearne', self, 0, 1.5))
#        self.uncertainties.append(LookupUncertainty([(-0.5, 2), (-2, 1), (0, 100), (20, 120), (0, 2), (0.5, 1)], "effect of hours worked on energy drain lookup", 'hearne', self, 0, 3))
#        self.uncertainties.append(LookupUncertainty([(-0.5, 1), (-0.5, 0.5), (0, 0.15), (0.05, 0.2), (0, 2), (0.5, 1)], "effect of low energy on further depletion lookup", 'hearne', self, 0, 1))
        
        self.uncertainties.append(LookupUncertainty([(-0.5, 0.1), (0.9, 1.5), (1, 10), (0, 0.1), (0, 1)], "accomplishments per hour lookup", 'approximation', self, 0, 1))
        self.uncertainties.append(LookupUncertainty([(-1.5, -0.5), (0.2, 0.8), (1, 10), (-1, 0.25), (0, 1.6)], "fractional change in expectations from perceived adequacy lookup", 'approximation', self, -1, 1))
        self.uncertainties.append(LookupUncertainty([(4, 8), (-0.1, 0.3), (1, 10), (4, 6), (0, 1.6)], "effect of perceived adequacy on energy drain lookup", 'approximation', self, 0, 10))
        self.uncertainties.append(LookupUncertainty([(2, 5), (0, 0.8), (1, 10), (2, 3), (0, 1.6)], "effect of perceived adequacy of hours worked lookup", 'approximation', self, 0, 2.5))
        self.uncertainties.append(LookupUncertainty([(-1, 0.1), (0.5, 1.5), (1, 10), (0, 0.2), (0, 1)], "effect of energy levels on hours worked lookup", 'approximation', self, 0, 1.5))
        self.uncertainties.append(LookupUncertainty([(1, 2), (-1, 0.1), (1, 10), (0.9, 1.2), (0.8, 1)], "effect of high energy on further recovery lookup", 'approximation', self, 0, 1.25))
        self.uncertainties.append(LookupUncertainty([(1, 2), (-1, 0.3), (1, 10), (1, 1.5), (0, 120)], "effect of hours worked on energy recovery lookup", 'approximation', self, 0, 1.5))
        self.uncertainties.append(LookupUncertainty([(-0.5, 0.35), (3, 5), (1, 10), (0.2, 0.4), (0, 120)], "effect of hours worked on energy drain lookup", 'approximation', self, 0, 3))
        self.uncertainties.append(LookupUncertainty([(-1, 0), (0.8, 2), (1, 10), (0, 0.1), (0, 0.2)], "effect of low energy on further depletion lookup", 'approximation', self, 0, 1))        

        self.delete_lookup_uncertainties()                   

        
if __name__ == "__main__":
    logger = logging.log_to_stderr(logging.INFO)
    model = Burnout(r'..\lookups', "burnout")

    model.step = 4 #reduce data to be stored
    ensemble = ModelEnsemble()
    ensemble.set_model_structure(model)

    #turn on parallel
    ensemble.parallel = False
    
    #run policy with old cases
    results = ensemble.perform_experiments(10)
    save_results(results, 'burnout_10_approx.cpickle')
    
#    results = load_results('burnout_100_2.cpickle')
#    outcome1 =['effect of hours worked on energy drain lookup']
#    outcome2 =['effect of hours worked on energy drain']
#    lines(results, outcome1, density=True, hist=True)
#    lines(results, outcome2, density=True, hist=True)
#    plt.show()