'''


'''
from __future__ import (unicode_literals, print_function, absolute_import,
                                        division)

import pandas as pd
import warnings

from .outcomes import AbstractOutcome
from .parameters import (IntegerParameter, RealParameter, CategoricalParameter,
                         Scenario, Policy)
from .samplers import determine_parameters
from .util import determine_objects

try:
    from platypus import EpsNSGAII  # @UnresolvedImport
    from platypus import Problem as PlatypusProblem
    import platypus
except ImportError:
    warnings.warn("platypus based optimization not available", ImportWarning)
    class PlatypusProblem(object):
        def __init__(self, *args, **kwargs):
            pass 
    EpsNSGAII = None
    platypus = None

# Created on 5 Jun 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = []


class Problem(PlatypusProblem):
    '''small extension to Platypus problem object, includes information on
    the names of the decision variables, the names of the outcomes,
    and the type of search'''
    
    def __init__(self, searchover, parameters, parameter_names,
                 outcome_names):
        super(Problem, self).__init__(len(parameters), len(outcome_names) , 
                                      nconstrs=0)
        assert len(parameters) == len(parameter_names)
        assert searchover in ('levers', 'uncertainties', 'robust')
        
        self.searchover = searchover
        self.parameters = parameters
        self.parameter_names = parameter_names
        self.outcome_names = outcome_names


class RobustProblem(Problem):
    '''small extension to Problem object for robust optimization, adds the 
    scenarios and the robustness functions'''
    
    
    def __init__(self, parameters, parameter_names,
                 outcome_names, scenarios, robustness_functions):
        super(RobustProblem, self).__init__('robust', parameters, 
                                            parameter_names, outcome_names)
        assert len(robustness_functions) == len(outcome_names)        
        self.scenarios = scenarios
        self.robustness_functions = robustness_functions


def to_problem(model, searchover):
    '''helper function to create Problem object
    
    Parameters
    ----------
    model : AbstractModel instance
    searchover : str
    
    Returns
    -------
    Problem instance
    
    '''
    _type_mapping = {RealParameter: platypus.Real,
                     IntegerParameter: platypus.Integer,
                     CategoricalParameter: platypus.Permutation}
    
    # extract the levers and the outcomes
    decision_variables = determine_parameters(model, searchover, union=True)
    dvnames = [dv.name for dv in decision_variables]

    outcomes = determine_objects(model,'outcomes')
    outcomes = [outcome for outcome in outcomes if 
                outcome.kind != AbstractOutcome.INFO]
    outcome_names = [outcome.name for outcome in outcomes]
    
    
    problem = Problem(searchover, decision_variables, dvnames,
                      outcome_names)
    problem.types = to_platypus_types(decision_variables)
    problem.directions = [outcome.kind for outcome in outcomes]

    return problem



def to_robust_problem(model, scenarios, robustness_functions):
    '''helper function to create RobustProblem object
    
    Parameters
    ----------
    model : AbstractModel instance
    robustness_functions : iterable of ScalarOutcomes
    
    Returns
    -------
    RobustProblem instance
    
    '''

    
    # extract the levers and the outcomes
    decision_variables = determine_parameters(model, 'levers', union=True)
    dvnames = [dv.name for dv in decision_variables]

    outcomes = robustness_functions
    outcome_names = [outcome.name for outcome in outcomes]
    
    problem = RobustProblem(decision_variables, dvnames, outcome_names, 
                            scenarios, robustness_functions)
    
    problem.types = to_platypus_types(decision_variables)
    problem.directions = [outcome.kind for outcome in outcomes]

    return problem


def to_platypus_types(decision_variables):
    
    _type_mapping = {RealParameter: platypus.Real,
                     IntegerParameter: platypus.Integer,
                     CategoricalParameter: platypus.Permutation}
    types = []
    for dv in decision_variables:
        klass = _type_mapping[type(dv)]
        
        if not isinstance(dv, CategoricalParameter):
            decision_variable = klass(dv.lower_bound, dv.upper_bound)
        else:
            decision_variable = klass(dv.categories)
        
        types.append(decision_variable)
    return types


def to_dataframe(optimizer, dvnames, outcome_names):
    '''helper function to turn results of optimization into a pandas DataFrame
    
    Parameters
    ----------
    optimizer : platypus algorithm instance
    dvnames : list of str
    outcome_names : list of str
    
    Returns
    -------
    pandas DataFrame
    
    
    '''
    
    solutions = []
    for solution in platypus.unique(platypus.nondominated(optimizer.result)):
        decision_vars = dict(zip(dvnames, solution.variables))
        decision_out = dict(zip(outcome_names, solution.objectives))
        
        result = decision_vars.copy()
        result.update(decision_out)
        
        solutions.append(result)

    results = pd.DataFrame(solutions, columns=dvnames+outcome_names)
    return results


def process_uncertainties(jobs):
    scenarios = []
    policies = 0
    
    for i, platypus_job in enumerate(jobs):
        variables = dict(zip(platypus_job.solution.problem.parameter_names, 
                             platypus_job.solution.variables))
        name = str(i)
        job = Scenario(name=name, **variables)
        scenarios.append(job)
        
    return scenarios, policies

def process_levers(jobs):
    policies = []
    scenarios = 0
    
    for i, platypus_job in enumerate(jobs):
        variables = dict(zip(platypus_job.solution.problem.parameter_names, 
                             platypus_job.solution.variables))
        name = str(i)
        job = Policy(name=name, **variables)
        policies.append(job)
    
    return scenarios, policies


def process_robust(jobs):
    _, policies = process_levers(jobs)
    scenarios = jobs[0].solution.problem.scenarios
    
    return scenarios, policies


def evaluate(jobs_collection, experiments, outcomes, problem):
    
    searchover = problem.searchover
    outcome_names = problem.outcome_names
    
    if searchover=='levers':
        column = 'policy'
    else:
        column = 'scenario_id'
    
    for entry, job in jobs_collection:
        logical = experiments[column] == entry.name
        job_outcomes = [outcomes[key][logical][0] for key in outcome_names]
        
        job.solution.problem.function = lambda x: job_outcomes
        job.solution.evaluate()


def evaluate_robust(jobs_collection, experiments, outcomes,
                     problem):
    robustness_functions = problem.robustness_functions
    
    for entry, job in jobs_collection:
        logical = experiments['policy'] == entry.name

        job_outcomes = []
        for rf in robustness_functions:
            data = [outcomes[var_name][logical] for var_name in 
                    rf.variable_name]
            job_outcomes.append(rf.function(*data))
        
        job.solution.problem.function = lambda x: job_outcomes
        job.solution.evaluate()