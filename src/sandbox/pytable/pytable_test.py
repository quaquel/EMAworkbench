'''
Created on Apr 16, 2012

@author: localadmin
'''
from expWorkbench.outcomes import Outcome
SVN_ID = '$Id: pytable_test.py 820 2012-05-03 06:08:16Z jhkwakkel $'

import os

from examples.FLUvensimExample import FluModel
from examples.EnergyTransExample import EnergyTrans
from expWorkbench import ModelEnsemble, ema_logging, ModelStructureInterface
from pytable_callback import *


def flu_test():
    
    model = FluModel(r'..\..\..\models\flu', "fluCase")
    ensemble = ModelEnsemble()
    ensemble.set_model_structure(model)
    
    ensemble.perform_experiments(cases = 10,
                                 callback=HDF5Callback)
    
def transition_test():
    
    model = EnergyTrans(r'..\..\..\models\EnergyTrans', "fluCase")
    ensemble = ModelEnsemble()
    ensemble.set_model_structure(model)
    
    ensemble.perform_experiments(cases = 10,
                                 callback=HDF5Callback)


def test_create_hdf5_ema__project():
    create_hdf5_ema__project(projectName='test', 
                             projectOwner='jhk', 
                             projectDescription='this is a test', 
                             fileName='test.h5')

def test_inspect():
    import inspect_test
    model = FluModel(r'..\..\..\models\flu', "fluCase")
    ensemble = ModelEnsemble()
    ensemble.set_model_structure(model)
    
    ensemble.perform_experiments(cases = 10,
                                 callback=inspect_test.InspectCallback)

def test_multiple_models():
    class Model1(ModelStructureInterface):
        uncertainties = [ParameterUncertainty((0,1),"a"),
                         ParameterUncertainty((0,1),"b")]
        
        outcomes = [Outcome("test")]
        
        def model_init(self, policy, kwargs):
            pass
        
        def run_model(self, case):
            self.output['test'] = 1

    class Model2(ModelStructureInterface):
        uncertainties = [ParameterUncertainty((0,1),"b"),
                         ParameterUncertainty((0,1),"c")]
        
        outcomes = [Outcome("test")]
        
        def model_init(self, policy, kwargs):
            pass
        
        def run_model(self, case):
            self.output['test'] = 1
    
#    os.remove('test.h5')

    nrOfExperiments = 10
    fileName = 'test.h5'
    experimentName = "one_exp_test"
    
    ensemble = ModelEnsemble()
    ensemble.add_model_structure(Model1('', "test1"))
    ensemble.add_model_structure(Model2('', "test2"))
    
    ensemble.perform_experiments(nrOfExperiments,
                                 callback=HDF5Callback,
                                 fileName=fileName,
                                 experimentName=experimentName)

def test_save_results():
    os.remove('test.h5')


    nrOfExperiments = 10
    fileName = 'test.h5'
    experimentName = "one_exp_test"
    
    ensemble = ModelEnsemble()
    ensemble.set_model_structure(FluModel(r'..\..\..\models\flu', "fluCase"))
    
    ensemble.perform_experiments(nrOfExperiments,
                                 callback=HDF5Callback,
                                 fileName=fileName,
                                 experimentName=experimentName)



if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)
    test_multiple_models()
#    test_save_results()
#    test_inspect()

