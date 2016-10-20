'''


'''
from __future__ import (unicode_literals, print_function, absolute_import,
                                        division)



# Created on Jul 23, 2016
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

from ema_workbench.connectors import PysdModel
from ema_workbench.em_framework import (RealParameter, TimeSeriesOutcome, 
                                        ModelEnsemble)
from ema_workbench.util import ema_logging

if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.DEBUG)    
    
    mdl_file = './models/pysd/Teacup.mdl'
    
    model = PysdModel(mdl_file=mdl_file)
    
    model.uncertainties = [RealParameter('Room Temperature', 33, 120)]
    model.outcomes = [TimeSeriesOutcome('Teacup Temperature')]
    
    ensemble = ModelEnsemble()  # instantiate an ensemble
    ensemble.model_structures = model  # set the model on the ensemble
#     ensemble.parallel = True
    ensemble.processes = 1
    ensemble.perform_experiments(100, reporting_interval=1)