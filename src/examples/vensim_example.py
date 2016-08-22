'''
Created on 3 Jan. 2011

This file illustrated the use the EMA classes for a contrived vensim
example


.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
                chamarat <c.hamarat (at) tudelft (dot) nl>
'''
from __future__ import (division, unicode_literals, absolute_import, 
                        print_function)

from ema_workbench.em_framework import (TimeSeriesOutcome, perform_experiments)
from ema_workbench.util import ema_logging 
from ema_workbench.connectors.vensim import VensimModelStructureInterface
from ema_workbench.em_framework.parameters import RealParameter

if __name__ == "__main__":
    #turn on logging
    ema_logging.log_to_stderr(ema_logging.INFO)
    
    #instantiate a model
    wd = r'./models/vensim example'
    vensimModel = VensimModelStructureInterface("simpleModel", wd=wd,
                                                model_file=r'\model.vpm')
    vensimModel.uncertainties = [RealParameter("x11", 0, 2.5),
                                 RealParameter("x12", -2.5, 2.5)]
    
    vensimModel.outcomes = [TimeSeriesOutcome('a', time=True)]
    
    results = perform_experiments(vensimModel, 1000, parallel=True)
    