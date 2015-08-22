'''
Created on 20 May, 2011

This module shows how you can use vensim models directly
instead of coding the model in Python. The underlying case
is the same as used in fluExample

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
                epruyt <e.pruyt (at) tudelft (dot) nl>
'''
from core import ModelEnsemble, ParameterUncertainty,\
                         Outcome, save_results, ema_logging
from connectors.vensim import VensimModelStructureInterface 

class FluModel(VensimModelStructureInterface):

    #base case model
    model_file = r'\FLUvensimV1basecase.vpm'
        
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
            self.model_file = policy['file']
        except KeyError:
            ema_logging.warning("key 'file' not found in policy")
        super(FluModel, self).model_init(policy, kwargs)
        
if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)
        
    model = FluModel(r'./models/flu', "flucase")
    ensemble = ModelEnsemble()
    ensemble.model_structure = model
    
    #add policies
    policies = [{'name': 'no policy',
                 'file': r'\FLUvensimV1basecase.vpm'},
                {'name': 'static policy',
                 'file': r'\FLUvensimV1static.vpm'},
                {'name': 'adaptive policy',
                 'file': r'\FLUvensimV1dynamic.vpm'}
                ]
    ensemble.policies = policies
    
    #turn on parallel processing
    ensemble.parallel = True 
    
    # run 1000 experiments
    nr_runs = 1000
    results = ensemble.perform_experiments(nr_runs)
    
    # save the results
    save_results(results, r'./data/{} flu cases.tar.gz'.format(nr_runs))
