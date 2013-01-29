'''
Created on 20 May, 2011

This module shows how you can use vensim models directly
instead of coding the model in Python. The underlying case
is the same as used in fluExample

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
                epruyt <e.pruyt (at) tudelft (dot) nl>
'''
import numpy as np
import matplotlib.pyplot as plt

from sandbox.workbench_core_revision.uncertainties import ParameterUncertainty

from expWorkbench import Outcome, save_results, ema_logging
from expWorkbench.vensim import VensimModelStructureInterface 
import expWorkbench.ema_logging as logging
from analysis.plotting import lines


SVN_ID = '$Id: flu_vensim_example.py 1056 2012-12-14 11:23:14Z jhkwakkel $'

class FluModel(VensimModelStructureInterface):

    #base case model
    modelFile = r'\FLUvensimV1basecase.vpm'
        
    #outcomes
    outcomes = [Outcome('deceased population region 1', time=True),
                Outcome('infected fraction R1', time=True)]
 
    #Plain Parametric Uncertainties 
    uncertainties = [
        ParameterUncertainty((0, 0.5), 
                             "additional seasonal immune population fraction R1"),
        ParameterUncertainty((0, 0.5), 
                             "additional seasonal immune population fraction R2"),
        ParameterUncertainty((0.0001, 0.1), 
                             "fatality ratio region 1"),
        ParameterUncertainty((0.0001, 0.1), 
                             "fatality rate region 2"),
        ParameterUncertainty((0, 0.5), 
                             "initial immune fraction of the population of region 1"),
        ParameterUncertainty((0, 0.5), 
                             "initial immune fraction of the population of region 2"),
        ParameterUncertainty((0, 0.9), 
                             "normal interregional contact rate"),
        ParameterUncertainty((0, 0.5), 
                             "permanent immune population fraction R1"),
        ParameterUncertainty((0, 0.5), 
                             "permanent immune population fraction R2"),
        ParameterUncertainty((0.1, 0.75), 
                             "recovery time region 1"),
        ParameterUncertainty((0.1, 0.75), 
                             "recovery time region 2"),
        ParameterUncertainty((0.5,2), 
                             "susceptible to immune population delay time region 1"),
        ParameterUncertainty((0.5,2), 
                             "susceptible to immune population delay time region 2"),
        ParameterUncertainty((0.01, 5), 
                             "root contact rate region 1"),
        ParameterUncertainty((0.01, 5), 
                             "root contact ratio region 2"),
        ParameterUncertainty((0, 0.15), 
                             "infection ratio region 1"),
        ParameterUncertainty((0, 0.15), 
                             "infection rate region 2"),
        ParameterUncertainty((10, 100), 
                             "normal contact rate region 1"),
        ParameterUncertainty((10, 200), 
                             "normal contact rate region 2")]
                         
    def model_init(self, policy, kwargs):
        '''initializes the model'''
        
        try:
            self.modelFile = policy['file']
        except KeyError:
            logging.warning("key 'file' not found in policy")
        super(FluModel, self).model_init(policy, kwargs)
    

def run_test(ensemble, cases):
    np.random.seed(100)
    #add policies
    policies = [{'name': 'no policy',
                 'file': r'\FLUvensimV1basecase.vpm'},
                {'name': 'static policy',
                 'file': r'\FLUvensimV1static.vpm'},
                {'name': 'adaptive policy',
                 'file': r'\FLUvensimV1dynamic.vpm'}
                ]
    ensemble.add_policies(policies)
    
    import time

    ensemble.parallel = True #turn on parallel processing

    timing = []
    for i in range(5):
        start_time = time.time()
        results = ensemble.perform_experiments(cases, reporting_interval=1000)
        timing.append(time.time()-start_time)
    for entry in timing:
        ema_logging.info(str(entry))

def test_new_code(cases):
    from sandbox.workbench_core_revision.model_ensemble import ModelEnsemble
    model = FluModel(r'D:\workspace\EMA-workbench\models\flu', "fluCase")
    ensemble = ModelEnsemble()
    ensemble.set_model_structure(model)
    
    ema_logging.info('------------- New Code -------------')
    
    run_test(ensemble,cases)
    del ensemble, model

def test_old_code(cases):
    from expWorkbench import ModelEnsemble    
    model = FluModel(r'D:\workspace\EMA-workbench\models\flu', "fluCase")
    ensemble = ModelEnsemble()
    ensemble.set_model_structure(model)
    
    ema_logging.info('------------- Old Code -------------')
    run_test(ensemble, cases)
    del ensemble, model

if __name__ == "__main__":
    logging.log_to_stderr(logging.INFO)
    test_old_code(1000)
    test_new_code(1000)
    
    