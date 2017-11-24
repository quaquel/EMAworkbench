'''


'''
from __future__ import (unicode_literals, print_function, absolute_import,
                                        division)

import os
import pandas as pd
import warnings

from .outcomes import AbstractOutcome
from .parameters import (IntegerParameter, RealParameter, CategoricalParameter,
                         Scenario, Policy)
from .samplers import determine_parameters
from .util import determine_objects

try:
    from platypus import EpsNSGAII, Hypervolume  # @UnresolvedImport
    from platypus import Problem as PlatypusProblem
    import platypus
except ImportError:
    warnings.warn("platypus based optimization not available", ImportWarning)
    class PlatypusProblem(object):
        constraints = []
        
        def __init__(self, *args, **kwargs):
            pass 
    EpsNSGAII = None
    platypus = None


# Created on 5 Jun 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ["Problem", "Robust_Problem", "EpsilonProgress", "HyperVolume",
           "Convergence", "ArchiveLogger"]

class Problem(PlatypusProblem):
    '''small extension to Platypus problem object, includes information on
    the names of the decision variables, the names of the outcomes,
    and the type of search'''
    
    def __init__(self, searchover, parameters, parameter_names,
                 outcome_names, constraint_names, reference=None):
        super(Problem, self).__init__(len(parameters), len(outcome_names) , 
                                      nconstrs=len(constraint_names))
        assert len(parameters) == len(parameter_names)
        assert searchover in ('levers', 'uncertainties', 'robust')
        
        if searchover=='levers':
            assert not reference or isinstance(reference, Scenario) 
        elif searchover=='uncertainties':
            assert not reference or isinstance(reference, Policy) 
        else:
            assert not reference
        
        self.searchover = searchover
        self.parameters = parameters
        self.parameter_names = parameter_names
        self.outcome_names = outcome_names
        self.constraint_names = constraint_names
        self.reference = reference if reference else 0 


class RobustProblem(Problem):
    '''small extension to Problem object for robust optimization, adds the 
    scenarios and the robustness functions'''
    
    
    def __init__(self, parameters, parameter_names,
                 outcome_names, scenarios, robustness_functions, 
                 constraint_names):
        super(RobustProblem, self).__init__('robust', parameters, 
                                            parameter_names, outcome_names,
                                            constraint_names)
        assert len(robustness_functions) == len(outcome_names)        
        self.scenarios = scenarios
        self.robustness_functions = robustness_functions


def to_problem(model, searchover, reference=None):
    '''helper function to create Problem object
    
    Parameters
    ----------
    model : AbstractModel instance
    searchover : str
    reference : Policy or Scenario instance, optional
                overwrite the default scenario in case of searching over 
                levers, or default policy in case of searching over 
                uncertainties
    
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
    
    constraint_names = [c.name for c in model.constraints]
    
    problem = Problem(searchover, decision_variables, dvnames,
                      outcome_names, constraint_names, reference=reference)
    problem.types = to_platypus_types(decision_variables)
    problem.directions = [outcome.kind for outcome in outcomes]
    problem.constraints[:] = "==0"

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
    
    constraints = determine_parameters(model, 'constraints', union=True)
    constraint_names = [c.name for c in constraints]
    
    problem = RobustProblem(decision_variables, dvnames, outcome_names, 
                            scenarios, robustness_functions, constraint_names)
    
    problem.types = to_platypus_types(decision_variables)
    problem.directions = [outcome.kind for outcome in outcomes]
    problem.constraints[:] = "==0"

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
    problem = jobs[0].solution.problem
    scenarios = []
    
    for i, platypus_job in enumerate(jobs):
        variables = dict(zip(platypus_job.solution.problem.parameter_names, 
                             platypus_job.solution.variables))
        name = str(i)
        job = Scenario(name=name, **variables)
        scenarios.append(job)
    
    policies = problem.reference
        
    return scenarios, policies

def process_levers(jobs):
    problem = jobs[0].solution.problem
    policies = []
    
    for i, platypus_job in enumerate(jobs):
        variables = dict(zip(platypus_job.solution.problem.parameter_names, 
                             platypus_job.solution.variables))
        name = str(i)
        job = Policy(name=name, **variables)
        policies.append(job)
    
    scenarios = problem.reference
    
    return scenarios, policies


def process_robust(jobs):
    _, policies = process_levers(jobs)
    scenarios = jobs[0].solution.problem.scenarios
    
    return scenarios, policies


def evaluate(jobs_collection, experiments, outcomes, constraints, problem):
    
    searchover = problem.searchover
    outcome_names = problem.outcome_names
    constraint_names = problem.constraint_names
    
    if searchover=='levers':
        column = 'policy'
    else:
        column = 'scenario_id'
    
    for entry, job in jobs_collection:
        logical = experiments[column] == entry.name
        job_outcomes = [outcomes[key][logical][0] for key in outcome_names]
        job_constraints = [constraints.loc[logical, c].values[0] for c in 
                           constraint_names]
        
        if constraint_names:
            job.solution.problem.function = lambda _: (job_outcomes, 
                                                       job_constraints)
        else:
            job.solution.problem.function = lambda _: job_outcomes
        job.solution.evaluate()


def evaluate_robust(jobs_collection, experiments, outcomes,
                    constraints, problem):
    robustness_functions = problem.robustness_functions
    constraint_names = problem.constraint_names
    
    for entry, job in jobs_collection:
        logical = experiments['policy'] == entry.name

        job_outcomes = []
        for rf in robustness_functions:
            data = [outcomes[var_name][logical] for var_name in 
                    rf.variable_name]
            score = rf.function(*data)
            job_outcomes.append(score)
        
        job_constraints = [constraints.loc[logical, c][0] for c in constraint_names]
        
        if job_constraints:
            job.solution.problem.function = lambda _: (job_outcomes, job_constraints)
        else:
            job.solution.problem.function = lambda _: job_outcomes
        
#         job.solution.problem.function = lambda x: job_outcomes
        job.solution.evaluate()
        
class AbstractConvergenceMetric(object):
    '''base convergence metric class'''
    
    def __init__(self, name):
        self.name = name
        self.results = []
    
    def __call__(self, optimizer):
        raise NotImplementedError


class EpsilonProgress(AbstractConvergenceMetric):
    '''epsilon progress convergence metric class'''
    def __init__(self):
        super(EpsilonProgress, self).__init__("epsilon_progress")
    
    def __call__(self, optimizer):
        self.results.append(optimizer.algorithm.archive.improvements)

    
class HyperVolume(AbstractConvergenceMetric):
    '''Hypervolume convergence metric class
    
    Parameters
    ---------
    minimum : numpy array
    maximum : numpy array
    
    
    '''
    
    def __init__(self, minimum, maximum):
        super(HyperVolume, self).__init__("hypervolume")
        self.hypervolume_func = Hypervolume(minimum=minimum, maximum=maximum)
        
    def __call__(self, optimizer):
        self.results.append(self.hypervolume_func.calculate(optimizer.algorithm.archive))

class ArchiveLogger(AbstractConvergenceMetric):
    '''Helper class to write the archive to disk at each iteration
    
    Parameters
    ----------
    directory : str
    decision_varnames : list of str
    outcome_varnames : list of str
    base_filename : str, optional
    
    TODO:: put it in a tarbal instead of dedicated directory
    
    '''
    
    def __init__(self, directory, decision_varnames, 
                 outcome_varnames, base_filename='archive'):
        super(ArchiveLogger, self).__init__('archive_logger')
        self.directory = os.path.abspath(directory)
        self.base = base_filename
        self.decision_varnames = decision_varnames
        self.outcome_varnames = outcome_varnames
        self.index = 0
        
    def __call__(self, optimizer):
        self.index += 1
        
        fn = os.path.join(self.directory, '{}_{}.csv'.format(self.base, self.index))
        
        archive = to_dataframe(optimizer, self.decision_varnames, 
                               self.outcome_varnames)
        archive.to_csv(fn)
        

class Convergence(object):
    '''helper class for tracking convergence of optimization'''
    
    valid_metrics = set(["hypervolume", "epsilon_progress", "archive_logger"])
    
    def __init__(self, metrics):
        if metrics is None:
            metrics = []
        
        self.metrics = metrics
        
        for metric in metrics:
            assert metric.name in self.valid_metrics
        
    def __call__(self, optimizer):
        for metric in self.metrics:
            metric(optimizer)
            
    def to_dataframe(self):
        progress = {metric.name:metric.results for metric in 
                    self.metrics if metric.results}
        return pd.DataFrame.from_dict(progress)
