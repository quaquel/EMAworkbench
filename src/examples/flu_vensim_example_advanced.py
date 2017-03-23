'''
Created on 20 May, 2011

This module shows how you can use vensim models directly
instead of coding the model in Python. The underlying case
is the same as used in fluExample

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
                epruyt <e.pruyt (at) tudelft (dot) nl>
'''
from __future__ import (absolute_import, print_function, unicode_literals, 
                        division)

import numpy as np

from ema_workbench import (TimeSeriesOutcome, ScalarOutcome,ema_logging,
                           perform_experiments)
from ema_workbench.em_framework.parameters import Policy, create_parameters

from ema_workbench.connectors.vensim import VensimModel 
from ema_workbench.em_framework.evaluators import MultiprocessingEvaluator

def time_of_max(infected_fraction, time):
    index = np.where(infected_fraction==np.max(infected_fraction))
    timing = time[index][0]
    return timing
    
if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)

    model = VensimModel("fluCase", wd=r'./models/flu',
                        model_file = r'/FLUvensimV1basecase.vpm')
            
    #outcomes
    model.outcomes = [TimeSeriesOutcome('deceased population region 1'),
                      TimeSeriesOutcome('infected fraction R1'),
                      ScalarOutcome('max infection fraction', 
                                    variable_name='infected fraction R1', 
                                    function=np.max),
                      ScalarOutcome('time of max', 
                                    variable_name=['infected fraction R1', 'TIME'], 
                                    function=time_of_max)]
    
    #create uncertainties based on csv 
    model.uncertainties = create_parameters('./models/flu/flu_uncertainties.csv') 
    
    #add policies
    policies = [Policy('no policy',
                       model_file=r'/FLUvensimV1basecase.vpm'),
                Policy('static policy',
                       model_file=r'/FLUvensimV1static.vpm'),
                Policy('adaptive policy',
                       model_file=r'/FLUvensimV1dynamic.vpm')
                ]
     
     
    with MultiprocessingEvaluator(model, 2) as evaluator:
        results = perform_experiments(model, 1000, policies=policies, 
                                      evaluator==evaluator)

