'''
Created on 20 dec. 2010

This file illustrated the use the EMA classes for a contrived example
It's main purpose has been to test the parallel processing functionality

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
<<<<<<< HEAD:src/ema_workbench/test/python_example.py
from ..em_framework import (ModelEnsemble, ModelStructureInterface,
                            RealParameter, ScalarOutcome)
=======
from ema_workbench.em_framework import (ModelEnsemble, ModelStructureInterface,
                                        ParameterUncertainty, Outcome)
>>>>>>> master:test/python_example.py

class SimplePythonModel(ModelStructureInterface):
    '''
    This class represents a simple example of how one can extent the basic
    ModelStructureInterface in order to do EMA on a simple model coded in
    Python directly
    '''
    
<<<<<<< HEAD:src/ema_workbench/test/python_example.py
    #specify uncertainties
    uncertainties = [RealParameter((0.1, 10), "x1"),
                     RealParameter((-0.01,0.01), "x2"),
                     RealParameter((-0.01,0.01), "x3")]
   
    #specify outcomes 
    outcomes = [ScalarOutcome('y')]
=======
    # specify uncertainties
    uncertainties = [ParameterUncertainty((0.1, 10), "x1"),
                     ParameterUncertainty((-0.01,0.01), "x2"),
                     ParameterUncertainty((-0.01,0.01), "x3")]
   
    # specify outcomes
    outcomes = [Outcome('y')]
>>>>>>> master:test/python_example.py

    def model_init(self, policy, kwargs):
        pass
    
    def run_model(self, case):
        """Method for running an instantiated model structure """
        self.output[self.outcomes[0].name] =  case['x1']*case['x2']+case['x3']
    

if __name__ == '__main__':
    model = SimplePythonModel(None, 'simpleModel') #instantiate the model
    ensemble = ModelEnsemble() #instantiate an ensemble
#     ensemble.parallel = True
    ensemble.set_model_structure(model) #set the model on the ensemble
    results = ensemble.perform_experiments(1000) #run 1000 experiments