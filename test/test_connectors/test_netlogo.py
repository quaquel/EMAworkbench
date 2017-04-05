'''
Created on 18 mrt. 2013

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
import os
import unittest
from ema_workbench.connectors.netlogo import NetLogoModel

# should be made conditional on the presence of jpype
__test__ = False

from ema_workbench.em_framework import (RealParameter, 
                                        CategoricalParameter,
                                        TimeSeriesOutcome)
from ema_workbench.em_framework.parameters import Policy
from ema_workbench.connectors import netlogo

def setUpModule():
    global cwd 
    cwd = os.getcwd()
    dir_of_module = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dir_of_module)

def tearDownModule():
    os.chdir(cwd)



class Test(unittest.TestCase):

    def test_init(self):
        wd = r"../models"
        model_file = r"/Wolf Sheep Predation.nlogo"
        
        model = NetLogoModel("predPreyNetlogo", wd=wd,
                                                       model_file=model_file)
        
    def test_run_model(self):
        wd = r"../models"
        
        model_file = r"/Wolf Sheep Predation.nlogo"
        
        model = NetLogoModel("predPreyNetlogo", wd=wd,
                                               model_file=model_file)
        
        model.run_length = 1000
    
        model.uncertainties = [RealParameter("grass-regrowth-time", 10, 100),
                         CategoricalParameter("grass?", ("true", "false")) ]
    
        model.outcomes = [TimeSeriesOutcome('sheep'),
                TimeSeriesOutcome('wolves')]
        model.model_init(Policy('no policy'))
        
        case = {"grass-regrowth-time": 35,
                "grass?": "true"}
        
        model.run_model(case)
        _ =  model.retrieve_output()

        model.cleanup()
        

if __name__ == "__main__":
    unittest.main()
