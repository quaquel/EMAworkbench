'''
Created on 3 Jan. 2011

This file illustrated the use the EMA classes for a contrived example
It's main purpose is to test the parallel processing functionality


.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
                chamarat <c.hamarat (at) tudelft (dot) nl>
'''
from expWorkbench import SimpleModelEnsemble, ParameterUncertainty, Outcome 
import expWorkbench.EMAlogging as EMAlogging

from expWorkbench.vensim import VensimModelStructureInterface

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
    EMAlogging.log_to_stderr(EMAlogging.INFO)
    
    #instantiate a model
    vensimModel = VensimExampleModel(r"..\..\models\vensim example", "simpleModel")
    
    #instantiate an ensemble
    ensemble = SimpleModelEnsemble()
    
    #set the model on the ensemble
    ensemble.set_model_structure(vensimModel)
    
    #run in parallel, if not set, FALSE is assumed
    ensemble.parallel = True
    
    #perform experiments
    result = ensemble.perform_experiments(1000)