'''
Created on 20 May, 2011

This module shows how you can use vensim models directly
instead of coding the model in Python. The underlying case
is the same as used in fluExample

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
                epruyt <e.pruyt (at) tudelft (dot) nl>
'''
from __future__ import absolute_import
from ..em_framework import RealUncertainty, TimeSeriesOutcome
from ..connectors.vensim import VensimModelStructureInterface 

class FluModel(VensimModelStructureInterface):
    #base case model
    model_file = r'\FLUvensimV1basecase.vpm'
        
    #outcomes
    outcomes = [TimeSeriesOutcome('deceased population region 1'),
                TimeSeriesOutcome('infected fraction R1')]
 
    #Plain Parametric Uncertainties 
    uncertainties = [
        RealUncertainty("additional seasonal immune population fraction R1", 
                        0, 0.5),
        RealUncertainty("additional seasonal immune population fraction R2",
                        0, 0.5),
        RealUncertainty("fatality ratio region 1", 0.0001, 0.1),
        RealUncertainty("fatality rate region 2", 0.0001, 0.1),
        RealUncertainty("initial immune fraction of the population of region 1",
                        0, 0.5),
        RealUncertainty("initial immune fraction of the population of region 2",
                        0, 0.5),
        RealUncertainty("normal interregional contact rate", 0, 0.9),
        RealUncertainty("permanent immune population fraction R1", 0, 0.5),
        RealUncertainty("permanent immune population fraction R2", 0, 0.5),
        RealUncertainty("recovery time region 1", 0.1, 0.75),
        RealUncertainty("recovery time region 2", 0.1, 0.75),
        RealUncertainty("susceptible to immune population delay time region 1", 
                        0.5,2),
        RealUncertainty("susceptible to immune population delay time region 2",
                        0.5,2),
        RealUncertainty("root contact rate region 1", 0.01, 5),
        RealUncertainty("root contact ratio region 2", 0.01, 5),
        RealUncertainty("infection ratio region 1", 0, 0.15),
        RealUncertainty("infection rate region 2", 0, 0.15),
        RealUncertainty("normal contact rate region 1", 10, 100),
        RealUncertainty("normal contact rate region 2", 10, 200)]
                         
