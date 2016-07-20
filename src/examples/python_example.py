'''
Created on 20 dec. 2010

This file illustrated the use the EMA classes for a contrived example
It's main purpose has been to test the parallel processing functionality

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

from ema_workbench.em_framework import (ModelEnsemble, Model, 
                                        RealParameter, ScalarOutcome)
from ema_workbench.util import ema_logging

def some_model(x1=None, x2=None, x3=None):
    return {'y':x1*x2+x3}

if __name__ == '__main__':
    ema_logging.LOG_FORMAT = '[%(name)s/%(levelname)s/%(processName)s] %(message)s'
    ema_logging.log_to_stderr(ema_logging.INFO)
    
    model = Model('simpleModel', function=some_model) #instantiate the model

    #specify uncertainties
    model.uncertainties = [RealParameter((0.1, 10), "x1"),
                           RealParameter((-0.01,0.01), "x2"),
                           RealParameter((-0.01,0.01), "x3")]
    #specify outcomes 
    model.outcomes = [ScalarOutcome('y')]

    ensemble = ModelEnsemble() #instantiate an ensemble
    ensemble.model_structure = model #set the model on the ensemble
    results = ensemble.perform_experiments(100, reporting_interval=1) #run 1000 experiments
    

