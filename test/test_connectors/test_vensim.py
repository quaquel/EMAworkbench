"""Unit tests for Vensim and vensimDLLwrapper."""

# Created on Jul 17, 2014
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

import os
import unittest

from ema_workbench.connectors.vensim import VensimModel, load_model
from ema_workbench.em_framework import (
    RealParameter,
    TimeSeriesOutcome,
    perform_experiments,
)
from ema_workbench.util import ema_logging

__test__ = True

if os.name != "nt":
    __test__ = False

class VensimTest(unittest.TestCase):
    def test_be_quiet(self):
        pass

    def test_load_model(self):
        pass

    def test_read_cin_file(self):
        pass

    def test_set_value(self):
        pass

    def test_run_simulation(self):
        model_file = r"../models/model.vpm"
        load_model(model_file)

    def test_get_data(self):
        pass


class VensimMSITest(unittest.TestCase):
    def test_vensim_model(self):
        # instantiate a model
        model = VensimModel("simple_model", wd="../models", model_file="model.vpm")
        # specify outcomes
        model.outcomes = [TimeSeriesOutcome("a")]

        # specify your uncertainties
        model.uncertainties = [RealParameter("x11", 0, 2.5),
                               RealParameter("x12", -2.5, 2.5)]


        nr_runs = 10
        experiments, outcomes = perform_experiments(model, nr_runs)

        self.assertEqual(experiments.shape[0], nr_runs)
        self.assertIn("TIME", outcomes.keys())
        self.assertIn(model.outcomes[0].name, outcomes.keys())


if __name__ == "__main__":
    if os.name == "nt":
        ema_logging.log_to_stderr(ema_logging.INFO)
        unittest.main()
