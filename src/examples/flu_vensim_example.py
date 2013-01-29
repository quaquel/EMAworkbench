'''
Created on 20 May, 2011

This module shows how you can use vensim models directly
instead of coding the model in Python. The underlying case
is the same as used in fluExample

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
                epruyt <e.pruyt (at) tudelft (dot) nl>
'''
from expWorkbench import ModelEnsemble, ParameterUncertainty,\
                         Outcome
from expWorkbench.vensim import VensimModelStructureInterface 
import expWorkbench.EMAlogging as logging

SVN_ID = '$Id: flu_vensim_example.py 1113 2013-01-27 14:21:16Z jhkwakkel $'

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
        
if __name__ == "__main__":
    logging.log_to_stderr(logging.DEBUG)
        
    model = FluModel(r'..\..\models\flu', "fluCase")
    ensemble = ModelEnsemble()
    ensemble.set_model_structure(model)
    
    #add policies
    policies = [{'name': 'no policy',
                 'file': r'\FLUvensimV1basecase.vpm'},
                {'name': 'static policy',
                 'file': r'\FLUvensimV1static.vpm'},
                {'name': 'adaptive policy',
                 'file': r'\FLUvensimV1dynamic.vpm'}
                ]
    ensemble.add_policies(policies)

    ensemble.parallel = True #turn on parallel processing

    results = ensemble.perform_experiments(10)
    
#    save_results(results, r'./data/1000 flu cases.bz2')