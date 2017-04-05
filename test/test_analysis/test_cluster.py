'''
Created on Mar 15, 2012

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''

from ema_workbench.analysis import clusterer
from ema_workbench.util import ema_logging

from ..utilities import load_scarcity_data

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)
    
    results = load_scarcity_data()
    
    clusterer.cluster(data=results, 
                      outcome='relative market price', 
                      distance='gonenc', 
                      cMethod='maxclust', 
                      cValue=5,
                      plotDendrogram=False)