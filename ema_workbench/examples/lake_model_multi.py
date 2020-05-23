'''
An example of the lake problem using the ema workbench.

The model itself is adapted from the Rhodium example by Dave Hadka,
see https://gist.github.com/dhadka/a8d7095c98130d8f73bc

'''
import math

import numpy as np
from scipy.optimize import brentq

from ema_workbench import (SplitModel, RealParameter, ScalarOutcome, Constant,
                           ema_logging, MultiprocessingEvaluator)
from ema_workbench.em_framework.evaluators import MC


def lake_problem_setup(
    b=0.42,          # decay rate for P in lake (0.42 = irreversible)
    q=2.0,           # recycling exponent
    mean=0.02,       # mean of natural inflows
    stdev=0.001,     # future utility discount rate
    delta=0.98,      # standard deviation of natural inflows
    alpha=0.4,       # utility from pollution
    num_variants=100,    # Monte Carlo sampling of natural inflows
    iterations = 100,
        **kwargs):
    try:
        decisions = [kwargs[str(i)] for i in range(100)]
    except KeyError:
        decisions = [0, ] * 100
    print ("Setting Up")
    nvars = len(decisions)
    return {'X': np.zeros((nvars,)),
            'average_daily_P':  np.zeros((nvars,)),
            'reliability' : 0.0,
            'b': b,
            'decisions': np.array(decisions),
            'q': q,
            'stdev': stdev,
            'delta': delta,
            'alpha': alpha,
            'num_variants': num_variants,
            'Pcrit': brentq(lambda x: x**q / (1 + x**q) - b * x, 0.01, 1.5),
            'nvars': len(decisions),
            'mean': mean,
            'iterations': iterations,
            'num_variants': num_variants,
            }


def lake_problem_update(state,
                        iteration
                        ):
    X= state["X"]
    b = state["b"]
    q = state["q"]
    decisions = state["decisions"]
    natural_inflows = state["natural_inflows"]
    nsamples = state["iterations"]
    average_daily_P = state["average_daily_P"]
    t = iteration
    X[t] = (1 - b) * X[t - 1] + X[t - 1]**q / (1 + X[t - 1]**q) + \
                decisions[t - 1] + natural_inflows[t - 1]
    average_daily_P[t] += X[t] / float(nsamples)


def lake_problem_variant_setup(state):
    X= state["X"]
    stdev=state["stdev"]
    mean=state["mean"]
    num_variants = state["num_variants"]
    X[0] = 0.0
    state["natural_inflows"] = np.random.lognormal(
            math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
            math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
            size=num_variants)

def lake_problem_variant_report(state, report):
    X = state["X"]
    Pcrit = state["Pcrit"]
    num_variants = state["num_variants"]

    iterations = state["iterations"]
    state["reliability"] += np.sum(X < Pcrit) / float(iterations * num_variants)

def lake_problem_report(state):
    print(state)
    average_daily_P=state["average_daily_P"]
    alpha=state["alpha"]
    delta=state["delta"]
    num_variants=state["num_variants"]
    decisions=state["decisions"]
    reliability=state["reliability"]
    max_P = np.max(average_daily_P)
    utility = np.sum(alpha * decisions * np.power(delta, np.arange(num_variants)))
    inertia = np.sum(np.absolute(np.diff(decisions)) < 0.02) / float(num_variants - 1)

    return max_P, utility, inertia, reliability


if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)
    print ("in main")

    # instantiate the model
    lake_model = SplitModel('lakeproblem',
                       update=lake_problem_update,
                       setup=lake_problem_setup,
                       report=lake_problem_report,
                       iterations=100,
                       variant_report = lake_problem_variant_report,
                       variant_setup = lake_problem_variant_setup)
    lake_model.time_horizon = 100

    # specify uncertainties
    lake_model.uncertainties = [RealParameter('b', 0.1, 0.45),
                                RealParameter('q', 2.0, 4.5),
                                RealParameter('mean', 0.01, 0.05),
                                RealParameter('stdev', 0.001, 0.005),
                                RealParameter('delta', 0.93, 0.99)]

    # set levers, one for each time step
    lake_model.levers = [RealParameter(str(i), 0, 0.1) for i in
                         range(lake_model.time_horizon)]

    # specify outcomes
    lake_model.outcomes = [ScalarOutcome('max_P',),
                           ScalarOutcome('utility'),
                           ScalarOutcome('inertia'),
                           ScalarOutcome('reliability')]

    # override some of the defaults of the model
    lake_model.constants = [Constant('alpha', 0.41),
                            Constant('nsamples', 150)]

    # generate some random policies by sampling over levers
    n_scenarios = 1000
    n_policies = 4
    results=lake_model.run_experiment({})
    print(results)

#    with MultiprocessingEvaluator(lake_model) as evaluator:
#        res = evaluator.perform_experiments(n_scenarios, n_policies,
 #                                           levers_sampling=MC)

#        experiments, outcomes = res
#        print(experiments.shape)
#        print(list(outcomes.keys()))
#        print(list(outcomes.values()))
