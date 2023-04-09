"""
Created on 3 Jan. 2011

This file illustrated the use the EMA classes for a contrived vensim
example


.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
                chamarat <c.hamarat (at) tudelft (dot) nl>
"""
from ema_workbench import TimeSeriesOutcome, perform_experiments, RealParameter, ema_logging

from ema_workbench.connectors.vensim import VensimModel

if __name__ == "__main__":
    # turn on logging
    ema_logging.log_to_stderr(ema_logging.INFO)

    # instantiate a model
    wd = "./models/vensim example"
    vensimModel = VensimModel("simpleModel", wd=wd, model_file="model.vpm")
    vensimModel.uncertainties = [RealParameter("x11", 0, 2.5), RealParameter("x12", -2.5, 2.5)]

    vensimModel.outcomes = [TimeSeriesOutcome("a")]

    results = perform_experiments(vensimModel, 1000)
