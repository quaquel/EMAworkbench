'''
This example replicates Quinn, J.D., Reed, P.M., Keller, K. (2017)
Direct policy search for robust multi-objective management of deeply
uncertain socio-ecological tipping points. Environmental Modelling &
Software 92, 125-141.

It also show cases how the workbench can be used to apply the MORDM extension
suggested by Watson, A.A., Kasprzyk, J.R. (2017) Incorporating deeply uncertain
factors into the many objective search process. Environmental Modelling &
Software 89, 159-171.

'''
import math
import numpy as np

from scipy.optimize import brentq

from ema_workbench import (Model, RealParameter, ScalarOutcome, Constant,
                           ema_logging, MultiprocessingEvaluator,
                           CategoricalParameter, Scenario)

# Created on 1 Jun 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


def get_antropogenic_release(xt, c1, c2, r1, r2, w1):
    '''

    Parameters
    ----------
    xt : float
         polution in lake at time t
    c1 : float
         center rbf 1
    c2 : float
         center rbf 2
    r1 : float
         ratius rbf 1
    r2 : float
         ratius rbf 2
    w1 : float
         weight of rbf 1

    note:: w2 = 1 - w1

    '''

    rule = w1 * (abs(xt - c1) / r1)**3 + (1 - w1) * (abs(xt - c2) / r2)**3
    at1 = max(rule, 0.01)
    at = min(at1, 0.1)

    return at


def lake_problem(
    b=0.42,          # decay rate for P in lake (0.42 = irreversible)
    q=2.0,           # recycling exponent
    mean=0.02,       # mean of natural inflows
    stdev=0.001,     # future utility discount rate
    delta=0.98,      # standard deviation of natural inflows
    alpha=0.4,       # utility from pollution
    nsamples=100,    # Monte Carlo sampling of natural inflows
    myears=1,         # the runtime of the simulation model
    c1=0.25,
    c2=0.25,
    r1=0.5,
    r2=0.5,
        w1=0.5):

    Pcrit = brentq(lambda x: x**q / (1 + x**q) - b * x, 0.01, 1.5)

    X = np.zeros((myears,))
    average_daily_P = np.zeros((myears,))
    reliability = 0.0
    inertia = 0
    utility = 0

    for _ in range(nsamples):
        X[0] = 0.0
        decision = 0.1

        decisions = np.zeros(myears,)
        decisions[0] = decision

        natural_inflows = np.random.lognormal(
            math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
            math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
            size=myears)

        for t in range(1, myears):

            # here we use the decision rule
            decision = get_antropogenic_release(X[t - 1], c1, c2, r1, r2, w1)
            decisions[t] = decision

            X[t] = (1 - b) * X[t - 1] + X[t - 1]**q / \
                (1 + X[t - 1]**q) + decision + natural_inflows[t - 1]
            average_daily_P[t] += X[t] / nsamples

        reliability += np.sum(X < Pcrit) / (nsamples * myears)
        inertia += np.sum(np.absolute(np.diff(decisions) <
                                      0.02)) / (nsamples * myears)
        utility += np.sum(alpha * decisions * np.power(delta,
                                                       np.arange(myears))) / nsamples
    max_P = np.max(average_daily_P)

    return max_P, utility, inertia, reliability


if __name__ == '__main__':

    ema_logging.log_to_stderr(ema_logging.INFO)

    # instantiate the model
    lake_model = Model('lakeproblem', function=lake_problem)
    # specify uncertainties
    lake_model.uncertainties = [RealParameter('b', 0.1, 0.45),
                                RealParameter('q', 2.0, 4.5),
                                RealParameter('mean', 0.01, 0.05),
                                RealParameter('stdev', 0.001, 0.005),
                                RealParameter('delta', 0.93, 0.99)]

    # set levers
    lake_model.levers = [RealParameter("c1", -2, 2),
                         RealParameter("c2", -2, 2),
                         RealParameter("r1", 0, 2),
                         RealParameter("r2", 0, 2),
                         CategoricalParameter("w1", np.linspace(0, 1, 10))
                         ]
    # specify outcomes
    lake_model.outcomes = [ScalarOutcome('max_P',
                                         kind=ScalarOutcome.MINIMIZE),  # @UndefinedVariable
                           ScalarOutcome('utility',
                                         kind=ScalarOutcome.MAXIMIZE),  # @UndefinedVariable
                           ScalarOutcome('inertia',
                                         kind=ScalarOutcome.MAXIMIZE),  # @UndefinedVariable
                           ScalarOutcome('reliability',
                                         kind=ScalarOutcome.MAXIMIZE)]  # @UndefinedVariable

    # override some of the defaults of the model
    lake_model.constants = [Constant('alpha', 0.41),
                            Constant('nsamples', 100),
                            Constant('myears', 100)]

    # reference is optional, but can be used to implement search for
    # various user specified scenarios along the lines suggested by
    # Watson and Kasprzyk (2017)
    reference = Scenario('reference', b=0.4, q=2, mean=0.02, stdev=0.01)

    with MultiprocessingEvaluator(lake_model) as evaluator:
        evaluator.optimize(searchover='levers', nfe=100000,
                           epsilons=[0.1, ] * len(lake_model.outcomes),
                           reference=reference)
