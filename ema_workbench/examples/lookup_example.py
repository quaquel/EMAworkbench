'''
Created on Oct 1, 2012

This is a simple example of the lookup uncertainty provided for
use in conjuction with vensim models. This example is largely based on
`Eker et al. (2014) <http://onlinelibrary.wiley.com/doi/10.1002/sdr.1518/suppinfo>`_

@author: sibeleker
@author: jhkwakkel
'''
from __future__ import (division, unicode_literals, print_function,
                        absolute_import)

import matplotlib.pyplot as plt

from ema_workbench import TimeSeriesOutcome, perform_experiments, ema_logging

from ema_workbench.connectors.vensim import (LookupUncertainty,
                                             VensimModel)
from ema_workbench.analysis import lines, Density


class Burnout(VensimModel):
    model_file = r'\BURNOUT.vpm'
    outcomes = [TimeSeriesOutcome('Accomplishments to Date'),
                TimeSeriesOutcome('Energy Level'),
                TimeSeriesOutcome('Hours Worked Per Week'),
                TimeSeriesOutcome('accomplishments per hour')]

    def __init__(self, working_directory, name):
        super(Burnout, self).__init__(working_directory, name)

        self.uncertainties = [LookupUncertainty('hearne2', [(-1, 3), (-2, 1), (0, 0.9), (0.1, 1), (0.99, 1.01), (0.99, 1.01)],
                                                "accomplishments per hour lookup", self, 0, 1),
                              LookupUncertainty('hearne2', [(-0.75, 0.75), (-0.75, 0.75), (0, 1.5), (0.1, 1.6), (-0.3, 1.5), (0.25, 2.5)],
                                                "fractional change in expectations from perceived adequacy lookup", self, -1, 1),
                              LookupUncertainty('hearne2', [(-2, 2), (-1, 2), (0, 1.5), (0.1, 1.6), (0.5, 2), (0.5, 2)],
                                                "effect of perceived adequacy on energy drain lookup", self, 0, 10),
                              LookupUncertainty('hearne2', [(-2, 2), (-1, 2), (0, 1.5), (0.1, 1.6), (0.5, 1.5), (0.1, 2)],
                                                "effect of perceived adequacy of hours worked lookup", self, 0, 2.5),
                              LookupUncertainty('hearne2', [(-1, 1), (-1, 1), (0, 0.9), (0.1, 1), (0.5, 1.5), (1, 1.5)],
                                                "effect of energy levels on hours worked lookup", self, 0, 1.5),
                              LookupUncertainty('hearne2', [(-1, 1), (-1, 1), (0, 0.9), (0.1, 1), (0.5, 1.5), (1, 1.5)],
                                                "effect of high energy on further recovery lookup", self, 0, 1.25),
                              LookupUncertainty('hearne2', [(-2, 2), (-1, 1), (0, 100), (20, 120), (0.5, 1.5), (0.5, 2)],
                                                "effect of hours worked on energy recovery lookup", self, 0, 1.5),
                              LookupUncertainty('approximation', [(-0.5, 0.35), (3, 5), (1, 10), (0.2, 0.4), (0, 120)],
                                                "effect of hours worked on energy drain lookup", self, 0, 3),
                              LookupUncertainty('hearne1', [(0, 1), (0, 0.15), (1, 1.5), (0.75, 1.25)],
                                                "effect of low energy on further depletion lookup", self, 0, 1)]

        self._delete_lookup_uncertainties()


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)
    model = Burnout(r'./models/burnout', "burnout")

    # run policy with old cases
    results = perform_experiments(model, 100)
    lines(results, 'Energy Level', density=Density.BOXPLOT)
    plt.show()
