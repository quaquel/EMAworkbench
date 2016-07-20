'''
Created on Mar 15, 2012

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''

from ema_workbench.analysis import clusterer
from ema_workbench.util import ema_logging

from ..scarcity_example import ScarcityModel
from ema_workbench.em_framework import ModelEnsemble

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)
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