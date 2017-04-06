'''
Created on 20 dec. 2010

This file illustrated the use the EMA classes for a contrived example
It's main purpose has been to test the parallel processing functionality

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

from ema_workbench import (Model, RealParameter, ScalarOutcome, ema_logging,
                           perform_experiments)

def some_model(x1=None, x2=None, x3=None):
    return {'y':x1*x2+x3}

if __name__ == '__main__':
    ema_logging.LOG_FORMAT = '[%(name)s/%(levelname)s/%(processName)s] %(message)s'
    ema_logging.log_to_stderr(ema_logging.INFO)
    
    model = Model('simpleModel', function=some_model) #instantiate the model

    #specify uncertainties
    model.uncertainties = [RealParameter("x1", 0.1, 10),
                           RealParameter("x2", -0.01,0.01),
                           RealParameter("x3", -0.01,0.01)]
    #specify outcomes 
    model.outcomes = [ScalarOutcome('y')]

    results = perform_experiments(model, 100)
