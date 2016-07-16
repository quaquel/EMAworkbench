'''
Created on 8 mrt. 2011

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
                epruyt <e.pruyt (at) tudelft (dot) nl>
'''
from __future__ import division
import numpy as np
from math import exp

from ..em_framework import ModelEnsemble, TimeSeriesOutcome
from ..em_framework import RealUncertainty, CategoricalUncertainty

from ..util import ema_logging

from ..connectors.vensim import VensimModelStructureInterface

class ScarcityModel(VensimModelStructureInterface):
    model_file = r'\MetalsEMA.vpm'
        
    outcomes = [TimeSeriesOutcome('relative market price'),
                TimeSeriesOutcome('supply demand ratio'),
                TimeSeriesOutcome('real annual demand'),
                TimeSeriesOutcome('produced of intrinsically demanded'),
                TimeSeriesOutcome('supply'),
                TimeSeriesOutcome('Installed Recycling Capacity'),
                TimeSeriesOutcome('Installed Extraction Capacity')]
                
    uncertainties = [
             RealUncertainty("price elasticity of demand", 0, 0.5),
             RealUncertainty("fraction of maximum extraction capacity used", 
                             0.6, 1.2),
             RealUncertainty("initial average recycling cost", 1,4),
             RealUncertainty("exogenously planned extraction capacity",
                             0, 15000),
             RealUncertainty("absolute recycling loss fraction", 0.1, 0.5),
             RealUncertainty("normal profit margin", 0, 0.4),
             RealUncertainty("initial annual supply", 100000, 120000),
             RealUncertainty("initial in goods", 1500000, 2500000),
                  
             RealUncertainty("average construction time extraction capacity",
                             1,  10),
             RealUncertainty("average lifetime extraction capacity", 20,  40),
             RealUncertainty("average lifetime recycling capacity", 20, 40),
             RealUncertainty("initial extraction capacity under construction",
                             5000,  20000),
             RealUncertainty("initial recycling capacity under construction",
                             5000, 20000),
             RealUncertainty("initial recycling infrastructure", 5000, 20000),
                  
             #order of delay
             CategoricalUncertainty("order in goods delay", (1,4,10, 1000)),
             CategoricalUncertainty("order recycling capacity delay", (1,4,10)),
             CategoricalUncertainty("order extraction capacity delay", (1,4,10)),
                  
             #uncertainties associated with lookups
             RealUncertainty("lookup shortage loc", 20, 50),
             RealUncertainty("lookup shortage speed", 1, 5),
          
             RealUncertainty("lookup price substitute speed", 0.1, 0.5),
             RealUncertainty("lookup price substitute begin", 3, 7),
             RealUncertainty("lookup price substitute end", 15, 25),
                  
             RealUncertainty("lookup returns to scale speed", 0.01, 0.2),
             RealUncertainty("lookup returns to scale scale", 0.3, 0.7),
          
             RealUncertainty("lookup approximated learning speed", 0.01, 0.2),
             RealUncertainty("lookup approximated learning scale", 0.3, 0.6),
             RealUncertainty("lookup approximated learning start", 30, 60)]


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
        kwargs['shortage price effect lookup'] =  [self.f(x/10, speed, loc) for x in range(0,100)]
        
        speed = kwargs.pop("lookup price substitute speed")
        begin = kwargs.pop("lookup price substitute begin")
        end = kwargs.pop("lookup price substitute end")
        kwargs['relative price substitute lookup'] = [self.priceSubstite(x, speed, begin, end) for x in range(0,100, 10)]
                
        scale = kwargs.pop("lookup returns to scale speed")
        speed = kwargs.pop("lookup returns to scale scale")
        kwargs['returns to scale lookup'] = [self.returnsToScale(x, speed, scale) for x in range(0, 101, 10)]
        
        scale = kwargs.pop("lookup approximated learning speed")
        speed = kwargs.pop("lookup approximated learning scale")
        start = kwargs.pop("lookup approximated learning start")
        kwargs['approximated learning effect lookup'] = [self.approxLearning(x, speed, scale, start) for x in range(0, 101, 10)]    
        
        super(ScarcityModel, self).run_model(kwargs)
        
def determineBehavior(value):
    #value is a time series
  
    #vanaf hier test
    firstOrder = [value[i+1] - value[i]
                  for i in xrange(value.shape[0]-1)
                  ]
    firstOrder = np.asarray(firstOrder)
    a = np.absolute(firstOrder)
    #a = firstOrder
    secondOrder = [(firstOrder[i], a[i+1] - a[i])
                  for i in xrange(a.shape[0]-1)
                  ]
    
    atomicBehavior = []
    last = None
    steps = 0
    for a, entry in secondOrder:
        if a >= 0:
            a = "pos"
        else:
            a = "neg"
        if entry == 0:
            entry = a+" lin"
        elif entry <= 0:
            entry = a+" log"
        elif entry > 0:
            entry = a+" exp"
        
        if not last:
            last = entry
            steps+=1
        else:
            if last == entry:
                steps+=1
                continue
            else:
                atomicBehavior.append([last, steps])
                last = entry
                steps = 0
    atomicBehavior.append([last, steps])
    
    behavior = []
    behavior.append(atomicBehavior.pop(0))
    for entry in atomicBehavior:
        if entry[0] != behavior[-1][0] and entry[1] >2:
            behavior.append(entry)
        elif entry[1] <2:
            continue
        else:
            behavior[-1][1] =+ entry[1]
    behavior = [entry[0] for entry in behavior]
    
    return behavior   


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)
    
    model = ScarcityModel(r'..\data', "scarcity")
    
    ensemble = ModelEnsemble()
    ensemble.set_model_structure(model)
    ensemble.parallel = True
    results = ensemble.perform_experiments(100)
#    determineBehavior(results)
