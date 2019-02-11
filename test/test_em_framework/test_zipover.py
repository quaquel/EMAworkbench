

import unittest.mock as mock
import unittest
import numpy as np
import math

from ema_workbench.em_framework import Model, SequentialEvaluator
from ema_workbench.em_framework.parameters import (RealParameter,
												   CategoricalParameter,
                                                   Constant)
from ema_workbench.em_framework.outcomes import ScalarOutcome
from ema_workbench.util.ema_exceptions import EMAError

def fake_problem(
		b=0.42,  # decay rate for P in lake (0.42 = irreversible)
		q=2.0,  # recycling exponent
		mean=0.02,  # mean of natural inflows
		stdev=0.001,  # future utility discount rate
		delta=0.98,  # standard deviation of natural inflows
		alpha=0.4,  # utility from pollution
		nsamples=100,  # Monte Carlo sampling of natural inflows
		myears=1,  # the runtime of the simulation model
		c1=0.25,
		c2=0.25,
		r1=0.5,
		r2=0.5,
		w1=0.5
):
	# Results are not important here, other than that there are 4 of them.
	return 0, 0, 0, 0


class ZipOverTestCase(unittest.TestCase):

	def test_zip_over(self):

		# instantiate the model
		lake_model = Model('fakeproblem', function=fake_problem)
		lake_model.time_horizon = 100

		# specify uncertainties
		lake_model.uncertainties = [RealParameter('b', 0.1, 0.45),
									RealParameter('q', 2.0, 4.5),
									RealParameter('mean', 0.01, 0.05),
									RealParameter('stdev', 0.001, 0.005),
									RealParameter('delta', 0.93, 0.99)]

		# set levers, one for each time step
		lake_model.levers = [RealParameter("c1", -2, 2),
                         RealParameter("c2", -2, 2),
                         RealParameter("r1", 0, 2),
                         RealParameter("r2", 0, 2),
                         CategoricalParameter("w1", np.linspace(0, 1, 10))
                         ]

		# specify outcomes
		lake_model.outcomes = [ScalarOutcome('max_P' ,),
							   ScalarOutcome('utility'),
							   ScalarOutcome('inertia'),
							   ScalarOutcome('reliability')]

		# override some of the defaults of the model
		lake_model.constants = [Constant('alpha', 0.41),
								Constant('nsamples', 150)]

		# generate some random policies by sampling over levers
		n_scenarios = 10
		n_policies = 4

		with SequentialEvaluator(lake_model) as evaluator:
			results, outcomes = evaluator.perform_experiments(n_scenarios, n_policies)

		assert len(results) == 40

		with SequentialEvaluator(lake_model) as evaluator:
			results, outcomes = evaluator.perform_experiments(10, 10,
															  zip_over={'scenarios', 'policies'})
		assert len(results) == 10

		with self.assertRaises(EMAError):
			# mismatch in number of scenarios and policies.
			with SequentialEvaluator(lake_model) as evaluator:
				results, outcomes = evaluator.perform_experiments(n_scenarios, n_policies,
																  zip_over={'scenarios',
																			'policies'})

		with self.assertRaises(EMAError):
			# only one model, so zip_over fails
			with SequentialEvaluator(lake_model) as evaluator:
				results, outcomes = evaluator.perform_experiments(10, 10,
																  zip_over={'scenarios',
																			'policies',
																			'models'})

