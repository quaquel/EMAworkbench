'''
Created on 20 mrt. 2013

@author: localadmin
'''
import matplotlib.pyplot as plt

from ema_workbench.connectors.netlogo import NetLogoModelStructureInterface

from ema_workbench.em_framework import (ParameterUncertainty, 
                                        CategoricalUncertainty, 
                                        TimeSeriesOutcome,
                                        ModelEnsemble)
from ema_workbench.util import ema_logging
from ema_workbench.analysis import plotting, plotting_util

if __name__ == '__main__':
    
    model = NetLogoModelStructureInterface('predprey', 
                                   wd="./models/predatorPreyNetlogo", 
                                   model_file="/Wolf Sheep Predation.nlogo")
    model.run_length = 1000
    
    model.uncertainties = [ParameterUncertainty((1, 99), "grass-regrowth-time"),
                     ParameterUncertainty((1, 250), "initial-number-sheep"),
                     ParameterUncertainty((1, 250), "initial-number-wolves"),
                     ParameterUncertainty((1, 20), "sheep-reproduce"),
                     ParameterUncertainty((1, 20), "wolf-reproduce"),
                     CategoricalUncertainty(("true", "true"), "grass?") 
                     ]
    
    model.outcomes = [TimeSeriesOutcome('sheep'),
                      TimeSeriesOutcome('wolves'),
                      TimeSeriesOutcome('grass') ]
    
    #turn on logging
    ema_logging.log_to_stderr(ema_logging.DEBUG)
    ema_logging.info('in main')
     
    #instantiate an ensemble
    ensemble = ModelEnsemble()
    
    #set the model on the ensemble
    ensemble.model_structure = model
    
    #run in parallel, if not set, FALSE is assumed
    ensemble.parallel = True
    ensemble.processes = 2
    
    #perform experiments
    results = ensemble.perform_experiments(10, reporting_interval=1)

    plotting.lines(results, density=plotting_util.KDE)
    plt.show()
