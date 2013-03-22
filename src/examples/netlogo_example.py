'''
Created on 20 mrt. 2013

@author: localadmin
'''
import matplotlib.pyplot as plt

from connectors.netlogo import NetLogoModeStructureInterface

from expWorkbench import ParameterUncertainty, CategoricalUncertainty, Outcome,\
                         ModelEnsemble, ema_logging
from analysis import plotting


class PredatorPrey(NetLogoModeStructureInterface):
    model_file = r"\Wolf Sheep Predation.nlogo"
    
    run_length = 1000
    
    uncertainties = [ParameterUncertainty((1, 99), "grass-regrowth-time"),
                     ParameterUncertainty((1, 250), "initial-number-sheep"),
                     ParameterUncertainty((1, 250), "initial-number-wolves"),
                     CategoricalUncertainty(("true", "true"), "grass?") 
                     ]
    
    outcomes = [Outcome('sheep', time=True),
                Outcome('wolves', time=True),
                Outcome('grass', time=True) # TODO patches not working in reporting
                ]
    
if __name__ == "__main__":
    #turn on logging
    ema_logging.log_to_stderr(ema_logging.INFO)
    
    #instantiate a model
    vensimModel = PredatorPrey( r"C:\git\EMAworkbench\models\predatorPreyNetlogo", "simpleModel")
    
    #instantiate an ensemble
    ensemble = ModelEnsemble()
    
    #set the model on the ensemble
    ensemble.set_model_structure(vensimModel)
    
    #run in parallel, if not set, FALSE is assumed
#    ensemble.parallel = True
    
    #perform experiments
    results = ensemble.perform_experiments(10)

    plotting.envelopes(results, density=plotting.KDE)
    plt.show()
