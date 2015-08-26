'''
Created on 20 mrt. 2013

@author: localadmin
'''
import matplotlib.pyplot as plt

from connectors.netlogo import NetLogoModelStructureInterface

from core import (ParameterUncertainty, CategoricalUncertainty, Outcome,
                  ModelEnsemble)
from util import ema_logging
from analysis import plotting, plotting_util


class PredatorPrey(NetLogoModelStructureInterface):
    model_file = r"/Wolf Sheep Predation.nlogo"
    
    run_length = 1000
    
    uncertainties = [ParameterUncertainty((1, 99), "grass-regrowth-time"),
                     ParameterUncertainty((1, 250), "initial-number-sheep"),
                     ParameterUncertainty((1, 250), "initial-number-wolves"),
                     ParameterUncertainty((1, 20), "sheep-reproduce"),
                     ParameterUncertainty((1, 20), "wolf-reproduce"),
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
    fh = r"./models/predatorPreyNetlogo"
    model = PredatorPrey(fh, "simpleModel")
    
    #instantiate an ensemble
    ensemble = ModelEnsemble()
    
    #set the model on the ensemble
    ensemble.model_structure = model
    
    #run in parallel, if not set, FALSE is assumed
    ensemble.parallel = True
    
    #perform experiments
    results = ensemble.perform_experiments(100, reporting_interval=1)

    plotting.lines(results, density=plotting_util.KDE)
    plt.show()
