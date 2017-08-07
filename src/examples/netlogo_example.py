'''
Created on 20 mrt. 2013

@author: localadmin
'''
from __future__ import unicode_literals, absolute_import

import matplotlib.pyplot as plt


from ema_workbench.connectors.netlogo import NetLogoModel

from ema_workbench.em_framework import (TimeSeriesOutcome, RealParameter,
                                        perform_experiments)
from ema_workbench.util import ema_logging
from ema_workbench.analysis import plotting, plotting_util

if __name__ == '__main__':
    #turn on logging
    ema_logging.log_to_stderr(ema_logging.DEBUG)

    model = NetLogoModel('predprey', 
                          wd="./models/predatorPreyNetlogo", 
                          model_file="Wolf Sheep Predation.nlogo")
    model.run_length = 100
    
    model.uncertainties = [RealParameter("grass-regrowth-time", 1, 99),
                           RealParameter("initial-number-sheep", 1, 200),
                           RealParameter("initial-number-wolves", 1, 200),
                           RealParameter("sheep-reproduce", 1, 20),
                           RealParameter("wolf-reproduce", 1, 20),
                     ]
    
    model.outcomes = [TimeSeriesOutcome('sheep'),
                      TimeSeriesOutcome('wolves'),
                      TimeSeriesOutcome('grass') ]
     
    #perform experiments
    n = 100
    results = perform_experiments(model, n, parallel=True)