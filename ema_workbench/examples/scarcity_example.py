'''
Created on 8 mrt. 2011

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
                epruyt <e.pruyt (at) tudelft (dot) nl>
'''
from __future__ import (division, unicode_literals, print_function,
                        absolute_import)
from math import exp

from ema_workbench.em_framework import (RealParameter, CategoricalParameter,
                                        TimeSeriesOutcome, perform_experiments)
from ema_workbench.util import ema_logging
from ema_workbench.connectors.vensim import VensimModel


class ScarcityModel(VensimModel):
    def returnsToScale(self, x, speed, scale):

        return (x * 1000, scale * 1 / (1 + exp(-1 * speed * (x - 50))))

    def approxLearning(self, x, speed, scale, start):
        x = x - start
        loc = 1 - scale
        a = (x * 10000, scale * 1 / (1 + exp(speed * x)) + loc)
        return a

    def f(self, x, speed, loc):
        return (x / 10, loc * 1 / (1 + exp(speed * x)))

    def priceSubstite(self, x, speed, begin, end):
        scale = 2 * end
        start = begin - scale / 2

        return (x + 2000, scale * 1 / (1 + exp(-1 * speed * x)) + start)

    def run_model(self, scenario, policy):
        """Method for running an instantiated model structure """
        kwargs = scenario
        loc = kwargs.pop("lookup shortage loc")
        speed = kwargs.pop("lookup shortage speed")
        lookup = [self.f(x / 10, speed, loc) for x in range(0, 100)]
        kwargs['shortage price effect lookup'] = lookup

        speed = kwargs.pop("lookup price substitute speed")
        begin = kwargs.pop("lookup price substitute begin")
        end = kwargs.pop("lookup price substitute end")
        lookup = [self.priceSubstite(x, speed, begin, end)
                  for x in range(0, 100, 10)]
        kwargs['relative price substitute lookup'] = lookup

        scale = kwargs.pop("lookup returns to scale speed")
        speed = kwargs.pop("lookup returns to scale scale")
        lookup = [self.returnsToScale(x, speed, scale)
                  for x in range(0, 101, 10)]
        kwargs['returns to scale lookup'] = lookup

        scale = kwargs.pop("lookup approximated learning speed")
        speed = kwargs.pop("lookup approximated learning scale")
        start = kwargs.pop("lookup approximated learning start")
        lookup = [self.approxLearning(x, speed, scale, start)
                  for x in range(0, 101, 10)]
        kwargs['approximated learning effect lookup'] = lookup

        super(ScarcityModel, self).run_model(kwargs, policy)


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.DEBUG)

    model = ScarcityModel("scarcity", wd=r'./models/scarcity',
                          model_file=r'\MetalsEMA.vpm')

    model.outcomes = [TimeSeriesOutcome('relative market price'),
                      TimeSeriesOutcome('supply demand ratio'),
                      TimeSeriesOutcome('real annual demand'),
                      TimeSeriesOutcome('produced of intrinsically demanded'),
                      TimeSeriesOutcome('supply'),
                      TimeSeriesOutcome('Installed Recycling Capacity'),
                      TimeSeriesOutcome('Installed Extraction Capacity')]

    model.uncertainties = [
        RealParameter("price elasticity of demand", 0, 0.5),
        RealParameter("fraction of maximum extraction capacity used",
                      0.6, 1.2),
        RealParameter("initial average recycling cost", 1, 4),
        RealParameter("exogenously planned extraction capacity",
                      0, 15000),
        RealParameter("absolute recycling loss fraction", 0.1, 0.5),
        RealParameter("normal profit margin", 0, 0.4),
        RealParameter("initial annual supply", 100000, 120000),
        RealParameter("initial in goods", 1500000, 2500000),

        RealParameter("average construction time extraction capacity",
                      1, 10),
        RealParameter("average lifetime extraction capacity", 20, 40),
        RealParameter("average lifetime recycling capacity", 20, 40),
        RealParameter("initial extraction capacity under construction",
                      5000, 20000),
        RealParameter("initial recycling capacity under construction",
                      5000, 20000),
        RealParameter("initial recycling infrastructure", 5000, 20000),

        # order of delay
        CategoricalParameter("order in goods delay", (1, 4, 10, 1000)),
        CategoricalParameter("order recycling capacity delay", (1, 4, 10)),
        CategoricalParameter("order extraction capacity delay", (1, 4, 10)),

        # uncertainties associated with lookups
        RealParameter("lookup shortage loc", 20, 50),
        RealParameter("lookup shortage speed", 1, 5),

        RealParameter("lookup price substitute speed", 0.1, 0.5),
        RealParameter("lookup price substitute begin", 3, 7),
        RealParameter("lookup price substitute end", 15, 25),

        RealParameter("lookup returns to scale speed", 0.01, 0.2),
        RealParameter("lookup returns to scale scale", 0.3, 0.7),

        RealParameter("lookup approximated learning speed", 0.01, 0.2),
        RealParameter("lookup approximated learning scale", 0.3, 0.6),
        RealParameter("lookup approximated learning start", 30, 60)]

    results = perform_experiments(model, 50)
