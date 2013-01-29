'''
Created on Mar 15, 2012

@author: jhkwakkel
'''

from analysis import clusterer
from expWorkbench import EMAlogging

from expWorkbench import ModelEnsemble
from test.scarcity_example import ScarcityModel

if __name__ == "__main__":
    EMAlogging.log_to_stderr(EMAlogging.INFO)
    model = ScarcityModel(r'..\..\src\test', "fluCase")
       
    ensemble = ModelEnsemble()
    ensemble.set_model_structure(model)
    ensemble.parallel = True
    
    results = ensemble.perform_experiments(200)
    
    clusterer.cluster(data=results, 
                      outcome='relative market price', 
                      distance='gonenc', 
                      cMethod='maxclust', 
                      cValue=5,
                      plotDendrogram=False)