'''
Created on Oct 1, 2012

@author: sibeleker
'''
from expWorkbench import Outcome, ModelEnsemble, ema_logging
from connectors.vensim import  LookupUncertainty, VensimModelStructureInterface


class Burnout(VensimModelStructureInterface): 
    model_file = r'\BURNOUT.vpm'
    outcomes = [Outcome('Accomplishments to Date', time=True),
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
    
    def __init__(self, workingDirectory, name):
        super(Burnout, self).__init__(workingDirectory, name)
        
        self.uncertainties.append(LookupUncertainty('approximation', 
                                                    [(-0.5, 0.1), (0.9, 1.5), (1, 10), (0, 0.1), (0, 1)], 
                                                    "accomplishments per hour lookup",
                                                    self, 0, 1))
        self.uncertainties.append(LookupUncertainty('approximation', 
                                                    [(-1.5, -0.5), (0.2, 0.8), (1, 10), (-1, 0.25), (0, 1.6)],
                                                    "fractional change in expectations from perceived adequacy lookup",
                                                    self, -1, 1))
        self.uncertainties.append(LookupUncertainty('approximation', 
                                                    [(4, 8), (-0.1, 0.3), (1, 10), (4, 6), (0, 1.6)], 
                                                    "effect of perceived adequacy on energy drain lookup", 
                                                    self, 0, 10))
        self.uncertainties.append(LookupUncertainty('approximation',
                                                    [(2, 5), (0, 0.8), (1, 10), (2, 3), (0, 1.6)],
                                                    "effect of perceived adequacy of hours worked lookup", 
                                                    self, 0, 2.5))
        self.uncertainties.append(LookupUncertainty('approximation',
                                                    [(-1, 0.1), (0.5, 1.5), (1, 10), (0, 0.2), (0, 1)],
                                                    "effect of energy levels on hours worked lookup",
                                                    self, 0, 1.5))
        self.uncertainties.append(LookupUncertainty('approximation', 
                                                    [(1, 2), (-1, 0.1), (1, 10), (0.9, 1.2), (0.8, 1)],
                                                    "effect of high energy on further recovery lookup",
                                                    self, 0, 1.25))
        self.uncertainties.append(LookupUncertainty('approximation',
                                                    [(1, 2), (-1, 0.3), (1, 10), (1, 1.5), (0, 120)],
                                                    "effect of hours worked on energy recovery lookup",
                                                    self, 0, 1.5))
        self.uncertainties.append(LookupUncertainty('approximation',
                                                    [(-0.5, 0.35), (3, 5), (1, 10), (0.2, 0.4), (0, 120)],
                                                    "effect of hours worked on energy drain lookup",
                                                    self, 0, 3))
        self.uncertainties.append(LookupUncertainty('approximation',
                                                    [(-1, 0), (0.8, 2), (1, 10), (0, 0.1), (0, 0.2)],
                                                    "effect of low energy on further depletion lookup",
                                                    self, 0, 1))        

        self._delete_lookup_uncertainties()                   

        
if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)
    model = Burnout(r'.', "burnout")

    ensemble = ModelEnsemble()
    ensemble.set_model_structure(model)
    
    #run policy with old cases
    results = ensemble.perform_experiments(10)

    
