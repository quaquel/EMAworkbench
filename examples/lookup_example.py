'''
Created on Oct 1, 2012

This is a simple example of the lookup uncertainty provided for 
use in conjuction with vensim models. This example is largely based on 
`Eker et al. (2014) <http://onlinelibrary.wiley.com/doi/10.1002/sdr.1518/suppinfo>`_

@author: sibeleker
@author: jhkwakkel
'''
import matplotlib.pyplot as plt

from core import Outcome, ModelEnsemble
from util import ema_logging

from connectors.vensim import  LookupUncertainty, VensimModelStructureInterface


from analysis.plotting import lines
from analysis.plotting_util import BOXPLOT

class Burnout(VensimModelStructureInterface): 
    model_file = r'\BURNOUT.vpm'
    outcomes = [Outcome('Accomplishments to Date', time=True),
                Outcome('Energy Level', time=True),
                Outcome('Hours Worked Per Week', time=True),
                Outcome('accomplishments per hour', time=True)]
    
    def __init__(self, working_directory, name):
        super(Burnout, self).__init__(working_directory, name)
        
        uncertainties = [LookupUncertainty('hearne2',[(-1, 3), (-2, 1), (0, 0.9), (0.1, 1), (0.99, 1.01), (0.99, 1.01)], 
                                        "accomplishments per hour lookup", self, 0, 1),
                         LookupUncertainty('hearne2', [(-0.75, 0.75), (-0.75, 0.75), (0, 1.5), (0.1, 1.6), (-0.3, 1.5), (0.25, 2.5)], 
                                         "fractional change in expectations from perceived adequacy lookup", self, -1, 1),
                         LookupUncertainty('hearne2', [(-2, 2), (-1, 2), (0, 1.5), (0.1, 1.6), (0.5, 2), (0.5, 2)], 
                                        "effect of perceived adequacy on energy drain lookup", self, 0, 10),
                         LookupUncertainty('hearne2', [(-2, 2), (-1, 2), (0, 1.5), (0.1, 1.6), (0.5, 1.5), (0.1, 2)], 
                                        "effect of perceived adequacy of hours worked lookup", self, 0, 2.5),
                         LookupUncertainty('hearne2', [(-1, 1), (-1, 1), (0, 0.9), (0.1, 1), (0.5, 1.5), (1, 1.5)], 
                                        "effect of energy levels on hours worked lookup", self, 0, 1.5),
                         LookupUncertainty('hearne2', [(-1, 1), (-1, 1), (0, 0.9), (0.1, 1), (0.5, 1.5), (1, 1.5)], 
                                        "effect of high energy on further recovery lookup", self, 0, 1.25),
                         LookupUncertainty('hearne2', [(-2, 2), (-1, 1), (0, 100), (20, 120), (0.5, 1.5), (0.5, 2)], 
                                        "effect of hours worked on energy recovery lookup", self, 0, 1.5),
                         LookupUncertainty('approximation', [(-0.5, 0.35), (3, 5), (1, 10), (0.2, 0.4), (0, 120)],
                                        "effect of hours worked on energy drain lookup", self, 0, 3),
                         LookupUncertainty('hearne1', [(0, 1), (0, 0.15), (1, 1.5), (0.75, 1.25)], 
                                        "effect of low energy on further depletion lookup", self, 0, 1)]     

        self._delete_lookup_uncertainties()                   

        
if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)
    model = Burnout(r'./models/burnout', "burnout")

    ensemble = ModelEnsemble()
    ensemble.model_structure = model
    
    #run policy with old cases
    results = ensemble.perform_experiments(100)
    lines(results, 'Energy Level', density=BOXPLOT)
    plt.show()

    
