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
                 outcome_names, constraints, reference=None):
        if constraints is None:
            constraints = []

        super(Problem, self).__init__(len(parameters), len(outcome_names) , 
                                      nconstrs=len(constraints))
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
        self.ema_constraints = constraints
        self.constraint_names = [c.name for c in constraints]
        self.reference = reference if reference else 0 


class RobustProblem(Problem):
    '''small extension to Problem object for robust optimization, adds the 
    scenarios and the robustness functions'''

    def __init__(self, parameters, parameter_names,
                 outcome_names, scenarios, robustness_functions, 
                 constraints):
        super(RobustProblem, self).__init__('robust', parameters, 
                                            parameter_names, outcome_names,
                                            constraints)
        assert len(robustness_functions) == len(outcome_names)        
        self.scenarios = scenarios
        self.robustness_functions = robustness_functions


def to_problem(model, searchover, reference=None, constraints=None):
    '''helper function to create Problem object
    
    Parameters
    ----------
    model : AbstractModel instance
    searchover : str
    reference : Policy or Scenario instance, optional
                overwrite the default scenario in case of searching over 
                levers, or default policy in case of searching over 
                uncertainties
    constraints : list, optional
    
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
                      outcome_names, constraints, reference=reference)
    problem.types = to_platypus_types(decision_variables)
    problem.directions = [outcome.kind for outcome in outcomes]
    problem.constraints[:] = "==0"

    return problem


def to_robust_problem(model, scenarios, robustness_functions, constraints=None):
    '''helper function to create RobustProblem object

    Parameters
    ----------
    model : AbstractModel instance
    scenarios : collection
    robustness_functions : iterable of ScalarOutcomes
    constraints : list, optional


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
                            scenarios, robustness_functions, constraints)

    problem.types = to_platypus_types(decision_variables)
    problem.directions = [outcome.kind for outcome in outcomes]
    problem.constraints[:] = "==0"

    return problem


def to_platypus_types(decision_variables):
    '''helper function for mapping from workbench parameter types to
    platypus parameter types'''
    #TODO:: should categorical not be platypus.Subset, with size == 1?
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
    '''helper function to map jobs generated by platypus to Scenario objects

    Parameters
    ----------
    jobs : collection

    Returns
    -------
    scenarios, policies

    '''
    problem = jobs[0].solution.problem
    scenarios = []

    for i, platypus_job in enumerate(jobs):
        variables = transform_variables(problem, 
                                        platypus_job.solution.variables)
        variables = dict(zip(platypus_job.solution.problem.parameter_names, 
                             variables))
        name = str(i)
        job = Scenario(name=name, **variables)
        scenarios.append(job)

    policies = problem.reference

    return scenarios, policies

def process_levers(jobs):
    '''helper function to map jobs generated by platypus to Policy objects

    Parameters
    ----------
    jobs : collection

    Returns
    -------
    scenarios, policies
    
    '''
    problem = jobs[0].solution.problem
    policies = []
    
    for i, platypus_job in enumerate(jobs):
        variables = transform_variables(problem, 
                                        platypus_job.solution.variables)
        variables = dict(zip(platypus_job.solution.problem.parameter_names, 
                             variables))
        name = str(i)
        job = Policy(name=name, **variables)
        policies.append(job)
    
    scenarios = problem.reference
    
    return scenarios, policies


def process_robust(jobs):
    '''Helper function to process robust optimization jobs
    
    Parameters
    ----------
    jobs : collection

    Returns
    -------
    scenarios, policies
    
    '''
    _, policies = process_levers(jobs)
    scenarios = jobs[0].solution.problem.scenarios

    return scenarios, policies


def transform_variables(problem, variables):
    '''helper function for transforming platypus variables'''
    
    converted_vars = []
    for type, var in zip(problem.types, variables):  # @ReservedAssignment
        if isinstance(type, platypus.Integer):
            var = type.decode(var)
        converted_vars.append(var)
    return converted_vars


def evaluate(jobs_collection, experiments, outcomes, problem):
    '''Helper function for mapping the results from perform_experiments back
    to what platypus needs'''

    searchover = problem.searchover
    outcome_names = problem.outcome_names
    constraints = problem.ema_constraints
    
    if searchover=='levers':
        column = 'policy'
    else:
        column = 'scenario_id'
    
    for entry, job in jobs_collection:
        logical = experiments[column] == entry.name
        job_outcomes = {key:outcomes[key][logical][0] for key in outcome_names}
        
        # TODO:: only retain uncertainties
        job_experiment = experiments[logical]

        job_constraints = _evaluate_constraints(job_experiment, job_outcomes, 
                                                constraints)
        job_outcomes = [outcomes[key][logical][0] for key in outcome_names]        
        
        if constraints:
            job.solution.problem.function = lambda _: (job_outcomes, 
                                                       job_constraints)
        else:
            job.solution.problem.function = lambda _: job_outcomes
        job.solution.evaluate()


def evaluate_robust(jobs_collection, experiments, outcomes, problem):
    '''Helper function for mapping the results from perform_experiments back
    to what Platypus needs'''

    robustness_functions = problem.robustness_functions
    constraints = problem.ema_constraints
    
    for entry, job in jobs_collection:
        logical = experiments['policy'] == entry.name
        job_outcomes = {key:value[logical] for key, value in outcomes.items()}

        job_outcomes_dict = {}
        job_outcomes = []
        for rf in robustness_functions:
            data = [outcomes[var_name][logical] for var_name in 
                    rf.variable_name]
            score = rf.function(*data)
            job_outcomes_dict[rf.name] = score
            job_outcomes.append(score)

        # TODO:: only retain levers
        job_experiment = experiments[logical][0]
        job_constraints = _evaluate_constraints(job_experiment, 
                                                job_outcomes_dict,
                                                constraints)

        if job_constraints:
            job.solution.problem.function = lambda _: (job_outcomes, 
                                                       job_constraints)
        else:
            job.solution.problem.function = lambda _: job_outcomes
        
        job.solution.evaluate()


def _evaluate_constraints(job_experiment, job_outcomes, constraints):
    '''Helper function for evaluating the constraints for a given job'''
    job_constraints = []
    for constraint in constraints:
        data = [job_experiment[var] for var in constraint.parameter_names]
        data += [job_outcomes[var] for var in constraint.outcome_names]
        constraint_value = constraint.process(data)
        job_constraints.append(constraint_value)
    return job_constraints


class AbstractConvergenceMetric(object):
    '''base convergence metric class'''

    def __init__(self, name):
        super(AbstractConvergenceMetric, self).__init__()
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
