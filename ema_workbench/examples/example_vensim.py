"""Basic example of how to connect a Vensim model to the ema_workbench."""

# Created on 3 Jan. 2011
#
# This file illustrated the use the EMA classes for a contrived vensim
# example

from ema_workbench import (
    RealParameter,
    TimeSeriesOutcome,
    ema_logging,
    perform_experiments,
)
from ema_workbench.connectors.vensim import VensimModel

if __name__ == "__main__":
    # turn on logging
    ema_logging.log_to_stderr(ema_logging.INFO)

    # instantiate a model
    wd = "./models/vensim example"
    vensim_model = VensimModel("simple_model", wd=wd, model_file="model.vpm")
    vensim_model.uncertainties = [
        RealParameter("x11", 0, 2.5),
        RealParameter("x12", -2.5, 2.5),
    ]

    vensim_model.outcomes = [TimeSeriesOutcome("a")]

    results = perform_experiments(vensim_model, 1000)
