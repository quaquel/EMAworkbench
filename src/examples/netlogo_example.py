'''
Created on 20 mrt. 2013

@author: localadmin
'''
from __future__ import unicode_literals, absolute_import
import matplotlib.pyplot as plt

from ema_workbench.connectors.netlogo import NetLogoModel

from ema_workbench.em_framework import (TimeSeriesOutcome,CategoricalParameter,
                                        ModelEnsemble, RealParameter)
from ema_workbench.util import ema_logging
from ema_workbench.analysis import plotting, plotting_util
from ema_workbench.em_framework.parameters import CategoricalParameter

if __name__ == '__main__':
    
    model = NetLogoModel('predprey', 
                          wd="./models/predatorPreyNetlogo", 
                          model_file="Wolf Sheep Predation.nlogo")
    model.run_length = 1000
    
    model.uncertainties = [RealParameter("grass-regrowth-time", 1, 99),
                           RealParameter("initial-number-sheep", 1, 250),
                           RealParameter("initial-number-wolves", 1, 250),
                           RealParameter("sheep-reproduce", 1, 20),
                           RealParameter("wolf-reproduce", 1, 20),
                           CategoricalParameter("grass?", ("true", "false")) 
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
    ensemble.model_structures = model
    
    #run in parallel, if not set, FALSE is assumed
#     ensemble.parallel = True
    ensemble.processes = 2
    
    #perform experiments
    results = ensemble.perform_experiments(10, reporting_interval=1)

    plotting.lines(results, density=plotting_util.KDE)
    plt.show()
