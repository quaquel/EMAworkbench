'''


'''


# Created on Jul 23, 2016
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

from ema_workbench import (RealParameter, TimeSeriesOutcome, ema_logging,
                           perform_experiments)

from ema_workbench.connectors.pysd_connector import PysdModel

if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)

    mdl_file = './models/pysd/Teacup.mdl'

    model = PysdModel(mdl_file=mdl_file)

    model.uncertainties = [RealParameter('Room Temperature', 33, 120)]
    model.outcomes = [TimeSeriesOutcome('Teacup Temperature')]

    perform_experiments(model, 100)
