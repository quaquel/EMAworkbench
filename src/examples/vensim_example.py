'''
Created on 3 Jan. 2011

This file illustrated the use the EMA classes for a contrived example
It's main purpose is to test the parallel processing functionality


.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
                chamarat <c.hamarat (at) tudelft (dot) nl>
'''
from ema_workbench.em_framework import (ModelEnsemble, ParameterUncertainty, 
                                        TimeSeriesOutcome)
from ema_workbench.util import ema_logging 
from ema_workbench.connectors.vensim import VensimModelStructureInterface


if __name__ == "__main__":
    #turn on logging
    ema_logging.log_to_stderr(ema_logging.INFO)
    
    #instantiate a model
    wd = r'./models/vensim example'
    vensimModel = VensimModelStructureInterface("simpleModel", wd=wd,
                                                model_file=r'\model.vpm')
    vensimModel.outcomes = [TimeSeriesOutcome('a', time=True)]
    vensimModel.uncertainties = [ParameterUncertainty((0, 2.5), "x11"),
                                 ParameterUncertainty((-2.5, 2.5), "x12")]
    
    #instantiate an ensemble
    ensemble = ModelEnsemble()
    
    #set the model on the ensemble
    ensemble.model_structures = vensimModel
    
    #run in parallel, if not set, FALSE is assumed
    ensemble.parallel = True
    
    #perform experiments
    result = ensemble.perform_experiments(1000)
    