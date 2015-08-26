'''
Created on 8 mrt. 2011

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
                epruyt <e.pruyt (at) tudelft (dot) nl>
'''
from __future__ import division
from math import exp

from core import (ParameterUncertainty, CategoricalUncertainty, Outcome, 
                  ModelEnsemble)
from util import ema_logging
from connectors.vensim import VensimModelStructureInterface

class ScarcityModel(VensimModelStructureInterface):
    model_file = r'\MetalsEMA.vpm'
        
    outcomes = [Outcome('relative market price', time=True),
                Outcome('supply demand ratio', time=True),
                Outcome('real annual demand', time=True),
                Outcome('produced of intrinsically demanded', time=True),
                Outcome('supply', time=True),
                Outcome('Installed Recycling Capacity', time=True),
                Outcome('Installed Extraction Capacity', time=True)]
                
    uncertainties = [ParameterUncertainty((0, 0.5), 
                                          "price elasticity of demand"),
                     ParameterUncertainty((0.6, 1.2), 
                                          "fraction of maximum extraction capacity used"),
                     ParameterUncertainty((1,4), 
                                          "initial average recycling cost"),
                     ParameterUncertainty((0, 15000),
                                          "exogenously planned extraction capacity"),
                     ParameterUncertainty((0.1, 0.5),
                                          "absolute recycling loss fraction"),
                     ParameterUncertainty((0, 0.4),
                                          "normal profit margin"),
                     ParameterUncertainty((100000, 120000),
                                          "initial annual supply"),
                     ParameterUncertainty((1500000, 2500000),
                                          "initial in goods"),
                     ParameterUncertainty((1,  10),
                                          "average construction time extraction capacity"),
                     ParameterUncertainty((20,  40),
                                          "average lifetime extraction capacity"),
                     ParameterUncertainty((20, 40),
                                          "average lifetime recycling capacity"),
                     ParameterUncertainty((5000,  20000),
                                          "initial extraction capacity under construction"),
                     ParameterUncertainty((5000, 20000),
                                          "initial recycling capacity under construction"),
                     ParameterUncertainty((5000, 20000),
                                          "initial recycling infrastructure"),
                     #order of delay
                     CategoricalUncertainty((1,4,10, 1000), 
                                            "order in goods delay", 
                                            default = 4),
                     CategoricalUncertainty((1,4,10), 
                                            "order recycling capacity delay", 
                                            default = 4),
                     CategoricalUncertainty((1,4,10), 
                                            "order extraction capacity delay", 
                                            default = 4),
                     #uncertainties associated with lookups
                     ParameterUncertainty((20, 50),"lookup shortage loc"),
                     ParameterUncertainty((1, 5),"lookup shortage speed"),
                     ParameterUncertainty((0.1, 0.5),
                                          "lookup price substitute speed"),
                     ParameterUncertainty((3, 7),
                                          "lookup price substitute begin"),
                     ParameterUncertainty((15, 25),
                                          "lookup price substitute end"),
                     ParameterUncertainty((0.01, 0.2),
                                          "lookup returns to scale speed"),
                     ParameterUncertainty((0.3, 0.7),
                                          "lookup returns to scale scale"),
                     ParameterUncertainty((0.01, 0.2),
                                          "lookup approximated learning speed"),
                     ParameterUncertainty((0.3, 0.6),
                                          "lookup approximated learning scale"),
                     ParameterUncertainty((30, 60),
                                          "lookup approximated learning start")]

    def returnsToScale(self, x, speed, scale):
    
        return (x*1000, scale*1/(1+exp(-1* speed * (x-50))))
     
    def approxLearning(self, x, speed, scale, start):
        x = x-start
        loc = 1 - scale
        a = (x*10000, scale*1/(1+exp(speed * x))+loc)
        return a

    def f(self,x, speed, loc):
        return (x/10, loc*1/(1+exp(speed * x)))

    def priceSubstite(self, x, speed, begin, end):
        scale = 2 * end
        start = begin - scale/2
        
        return (x+2000, scale*1/(1+exp(-1* speed * x)) +start)

    def run_model(self, kwargs):
        """Method for running an instantiated model structure """
        
        loc = kwargs.pop("lookup shortage loc")
        speed = kwargs.pop("lookup shortage speed")
        lookup = [self.f(x/10, speed, loc) for x in range(0,100)]
        kwargs['shortage price effect lookup'] =  lookup
        
        speed = kwargs.pop("lookup price substitute speed")
        begin = kwargs.pop("lookup price substitute begin")
        end = kwargs.pop("lookup price substitute end")
        lookup = [self.priceSubstite(x, speed, begin, end) for x in range(0,100, 10)]
        kwargs['relative price substitute lookup'] = lookup
                        
        scale = kwargs.pop("lookup returns to scale speed")
        speed = kwargs.pop("lookup returns to scale scale")
        lookup = [self.returnsToScale(x, speed, scale) for x in range(0, 101, 10)]
        kwargs['returns to scale lookup'] = lookup
        
        scale = kwargs.pop("lookup approximated learning speed")
        speed = kwargs.pop("lookup approximated learning scale")
        start = kwargs.pop("lookup approximated learning start")
        lookup = [self.approxLearning(x, speed, scale, start) for x in range(0, 101, 10)]
        kwargs['approximated learning effect lookup'] = lookup
        
        super(ScarcityModel, self).run_model(kwargs)


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.DEBUG)
    
    model = ScarcityModel(r'./models/scarcity', "scarcity")
    
    ensemble = ModelEnsemble()
    ensemble.model_structure = model
    ensemble.parallel = True
    results = ensemble.perform_experiments(2)
