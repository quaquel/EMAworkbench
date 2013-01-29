'''
Created on 3 Jan. 2011

This file illustrated the use the EMA classes for a contrived example
It's main purpose is to test the parallel processing functionality


.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
                chamarat <c.hamarat (at) tudelft (dot) nl>
'''
from expWorkbench import ModelEnsemble, ParameterUncertainty, Outcome,\
                         ema_logging 
from expWorkbench.vensim import VensimModelStructureInterface

SVN_ID = '$Id: vensim_example.py 1055 2012-12-14 10:56:51Z jhkwakkel $'

class VensimExampleModel(VensimModelStructureInterface):
    '''
    example of the most simple case of doing EMA on
    a Vensim model.
    
    '''
    #note that this reference to the model should be relative
    #this relative path will be combined with the workingDirectory
    modelFile = r'\model.vpm'

    #specify outcomes
    outcomes = [Outcome('a', time=True)]

    #specify your uncertainties
    uncertainties = [ParameterUncertainty((0, 2.5), "x11"),
                     ParameterUncertainty((-2.5, 2.5), "x12")]

if __name__ == "__main__":
    #turn on logging
    ema_logging.log_to_stderr(ema_logging.INFO)
    
    #instantiate a model
    vensimModel = VensimExampleModel(r"..\..\models\vensim example", "simpleModel")
    
    #instantiate an ensemble
    ensemble = ModelEnsemble()
    
    #set the model on the ensemble
    ensemble.set_model_structure(vensimModel)
    
    #run in parallel, if not set, FALSE is assumed
    ensemble.parallel = True
    
    #perform experiments
    result = ensemble.perform_experiments(1000)