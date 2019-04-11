'''


'''
import copy
import functools
import os
import pandas as pd
import random
import warnings

import numpy as np

from .outcomes import AbstractOutcome
from .parameters import (IntegerParameter, RealParameter, CategoricalParameter,
                         BooleanParameter, Scenario, Policy)
from .samplers import determine_parameters
from .util import determine_objects
from ..util import get_module_logger, EMAError, temporary_filter, INFO
from . import callbacks, evaluators

try:
    from platypus import (
        EpsNSGAII,
        Hypervolume,
        Variator,
        Real,
        Integer,
        Subset,
        EpsilonProgressContinuation,
        RandomGenerator,
        TournamentSelector,
        NSGAII,
        EpsilonBoxArchive,
        Multimethod,
        GAOperator,
        SBX,
        PM,
        PCX,
        DifferentialEvolution,
        UNDX,
        SPX,
        UM)  # @UnresolvedImport
    from platypus import Problem as PlatypusProblem

    import platypus
except ImportError:
    warnings.warn("platypus based optimization not available", ImportWarning)

    class PlatypusProblem(object):
        constraints = []

        def __init__(self, *args, **kwargs):
            pass

    class Variator(object):
        def __init__(self, *args, **kwargs):
            pass

    class RandomGenerator(object):
        def __call__(self, *args, **kwargs):
            pass

    class TournamentSelector(object):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            pass

    class EpsilonProgressContinuation(object):
        pass

    EpsNSGAII = None
    platypus = None
    Real = Integer = Subset = None

# Created on 5 Jun 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ["Problem", "RobustProblem", "EpsilonProgress", "HyperVolume",
           "Convergence", "ArchiveLogger"]
_logger = get_module_logger(__name__)


class Problem(PlatypusProblem):
    '''small extension to Platypus problem object, includes information on
    the names of the decision variables, the names of the outcomes,
    and the type of search'''

    @property
    def parameter_names(self):
        return [e.name for e in self.parameters]

    def __init__(self, searchover, parameters,
                 outcome_names, constraints, reference=None):
        if constraints is None:
            constraints = []

        super(Problem, self).__init__(len(parameters), len(outcome_names),
                                      nconstrs=len(constraints))
#         assert len(parameters) == len(parameter_names)
        assert searchover in ('levers', 'uncertainties', 'robust')

        if searchover == 'levers':
            assert not reference or isinstance(reference, Scenario)
        elif searchover == 'uncertainties':
            assert not reference or isinstance(reference, Policy)
        else:
            assert not reference

        self.searchover = searchover
        self.parameters = parameters
        self.outcome_names = outcome_names
        self.ema_constraints = constraints
        self.constraint_names = [c.name for c in constraints]
        self.reference = reference if reference else 0


class RobustProblem(Problem):
    '''small extension to Problem object for robust optimization, adds the
    scenarios and the robustness functions'''

    def __init__(self, parameters, outcome_names, scenarios,
                 robustness_functions, constraints):
        super(RobustProblem, self).__init__('robust', parameters,
                                            outcome_names,
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

    # extract the levers and the outcomes
    decision_variables = determine_parameters(model, searchover, union=True)

    outcomes = determine_objects(model, 'outcomes')
    outcomes = [outcome for outcome in outcomes if
                outcome.kind != AbstractOutcome.INFO]
    outcome_names = [outcome.name for outcome in outcomes]

    if not outcomes:
        raise EMAError(("no outcomes specified to optimize over, "
                        "all outcomes are of kind=INFO"))

    problem = Problem(searchover, decision_variables,
                      outcome_names, constraints, reference=reference)
    problem.types = to_platypus_types(decision_variables)
    problem.directions = [outcome.kind for outcome in outcomes]
    problem.constraints[:] = "==0"

    return problem


def to_robust_problem(
        model,
        scenarios,
        robustness_functions,
        constraints=None):
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

    outcomes = robustness_functions
    outcomes = [outcome for outcome in outcomes if
                outcome.kind != AbstractOutcome.INFO]
    outcome_names = [outcome.name for outcome in outcomes]

    if not outcomes:
        raise EMAError(("no outcomes specified to optimize over, "
                        "all outcomes are of kind=INFO"))

    problem = RobustProblem(decision_variables, outcome_names,
                            scenarios, robustness_functions, constraints)

    problem.types = to_platypus_types(decision_variables)
    problem.directions = [outcome.kind for outcome in outcomes]
    problem.constraints[:] = "==0"

    return problem


def to_platypus_types(decision_variables):
    '''helper function for mapping from workbench parameter types to
    platypus parameter types'''
    # TODO:: should categorical not be platypus.Subset, with size == 1?
    _type_mapping = {RealParameter: platypus.Real,
                     IntegerParameter: platypus.Integer,
                     CategoricalParameter: platypus.Subset,
                     BooleanParameter: platypus.Subset,
                     }

    types = []
    for dv in decision_variables:
        klass = _type_mapping[type(dv)]

        if not isinstance(dv, (CategoricalParameter, BooleanParameter)):
            decision_variable = klass(dv.lower_bound, dv.upper_bound)
        else:
            decision_variable = klass(dv.categories, 1)

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
        vars = transform_variables(solution.problem,  # @ReservedAssignment
                                   solution.variables)

        decision_vars = dict(zip(dvnames, vars))
        decision_out = dict(zip(outcome_names, solution.objectives))

        result = decision_vars.copy()
        result.update(decision_out)

        solutions.append(result)

    results = pd.DataFrame(solutions, columns=dvnames + outcome_names)
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

    jobs = _process(jobs, problem)
    for i, job in enumerate(jobs):
        name = str(i)
        scenario = Scenario(name=name, **job)
        scenarios.append(scenario)

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
    jobs = _process(jobs, problem)
    for i, job in enumerate(jobs):
        name = str(i)
        job = Policy(name=name, **job)
        policies.append(job)

    scenarios = problem.reference

    return scenarios, policies


def _process(jobs, problem):
    '''helper function to transform platypus job to dict with correct
    values for workbench'''

    processed_jobs = []
    for job in jobs:
        variables = transform_variables(problem,
                                        job.solution.variables)
        processed_job = {}
        for param, var in zip(problem.parameters, variables):
            try:
                var = var.value
            except AttributeError:
                pass
            processed_job[param.name] = var
        processed_jobs.append(processed_job)
    return processed_jobs


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
        var = type.decode(var)
        try:
            var = var[0]
        except TypeError:
            pass

        converted_vars.append(var)
    return converted_vars


def evaluate(jobs_collection, experiments, outcomes, problem):
    '''Helper function for mapping the results from perform_experiments back
    to what platypus needs'''

    searchover = problem.searchover
    outcome_names = problem.outcome_names
    constraints = problem.ema_constraints

    if searchover == 'levers':
        column = 'policy'
    else:
        column = 'scenario'

    for entry, job in jobs_collection:
        logical = experiments[column] == entry.name
        job_outcomes = {key: outcomes[key][logical][0]
                        for key in outcome_names}

        # TODO:: only retain uncertainties
        job_experiment = experiments[logical]

        data = {k: v[logical][0] for k, v in outcomes.items()}
        job_constraints = _evaluate_constraints(job_experiment, data,
                                                constraints)
        job_outcomes = [outcomes[key][logical][0] for key in outcome_names]

        if job_constraints:
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
        job_outcomes = {key: value[logical] for key, value in outcomes.items()}

        job_outcomes_dict = {}
        job_outcomes = []
        for rf in robustness_functions:
            data = [outcomes[var_name][logical] for var_name in
                    rf.variable_name]
            score = rf.function(*data)
            job_outcomes_dict[rf.name] = score
            job_outcomes.append(score)

        # TODO:: only retain levers
        job_experiment = experiments[logical].iloc[0]
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

    def reset(self):
        self.results = []


class EpsilonProgress(AbstractConvergenceMetric):
    '''epsilon progress convergence metric class'''

    def __init__(self):
        super(EpsilonProgress, self).__init__("epsilon_progress")

    def __call__(self, optimizer):
        self.results.append(optimizer.algorithm.archive.improvements)


class HyperVolume(AbstractConvergenceMetric):
    '''Hypervolume convergence metric class

    This metric is derived from a hyper-volume measure, which describes the
    multi-dimensional volume of space contained within the pareto front. When
    computed with minimum and maximums, it describes the ratio of dominated
    outcomes to all possible outcomes in the extent of the space.  Getting this
    number to be high or low is not necessarily important, as not all outcomes
    within the min-max range will be feasible.  But, having the hypervolume remain
    fairly stable over multiple generations of the evolutionary algorithm provides
    an indicator of convergence.

    Parameters
    ---------
    minimum : numpy array
    maximum : numpy array

    '''

    def __init__(self, minimum, maximum):
        super(HyperVolume, self).__init__("hypervolume")
        self.hypervolume_func = Hypervolume(minimum=minimum, maximum=maximum)

    def __call__(self, optimizer):
        self.results.append(self.hypervolume_func.calculate(
            optimizer.algorithm.archive))

    @classmethod
    def from_outcomes(cls, outcomes):
        ranges = [o.expected_range for o in outcomes if o.kind != o.INFO]
        minimum, maximum = np.asarray(list(zip(*ranges)))
        return cls(minimum, maximum)


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

        fn = os.path.join(
            self.directory, '{}_{}.csv'.format(self.base, self.index))

        archive = to_dataframe(optimizer, self.decision_varnames,
                               self.outcome_varnames)
        archive.to_csv(fn)


class OperatorProbabilities(AbstractConvergenceMetric):

    def __init__(self, name, index):
        super(OperatorProbabilities, self).__init__(name)
        self.index = index

    def __call__(self, optimizer):
        try:
            props = optimizer.algorithm.variator.probabilities
            self.results.append(props[self.index])
        except AttributeError:
            pass


class Convergence(object):
    '''helper class for tracking convergence of optimization'''

    valid_metrics = set(["hypervolume", "epsilon_progress", "archive_logger"])

    def __init__(self, metrics, max_nfe, convergence_freq=1000,
                 logging_freq=5):
        self.max_nfe = max_nfe
        self.generation = -1
        self.index = []
        self.last_check = 0

        if metrics is None:
            metrics = []

        self.metrics = metrics
        self.convergence_freq = convergence_freq
        self.logging_freq = logging_freq

        for metric in metrics:
            assert isinstance(metric, AbstractConvergenceMetric)
            metric.reset()

    def __call__(self, optimizer, ):
        nfe = optimizer.algorithm.nfe

        self.generation += 1

        if (nfe >= self.last_check + self.convergence_freq) or self.last_check == 0:
            self.index.append(nfe)
            self.last_check = nfe

            for metric in self.metrics:
                metric(optimizer)

        if self.generation % self.logging_freq == 0:
            _logger.info(
                "generation {}: {}/{} nfe".format(self.generation, nfe,
                                                  self.max_nfe))

    def to_dataframe(self):
        progress = {metric.name: metric.results for metric in
                    self.metrics if metric.results}

        progress = pd.DataFrame.from_dict(progress)

        if not progress.empty:
            progress['nfe'] = self.index

        return progress


class CombinedVariator(Variator):

    def __init__(self, crossover_prob=0.5, mutation_prob=1):
        super(CombinedVariator, self).__init__(2)
        self.SBX = platypus.SBX()
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

    def evolve(self, parents):
        child1 = copy.deepcopy(parents[0])
        child2 = copy.deepcopy(parents[1])
        problem = child1.problem

        # crossover
        # we will evolve the individual
        for i, type in enumerate(problem.types):  # @ReservedAssignment
            if random.random() <= self.crossover_prob:
                klass = type.__class__
                child1, child2 = self._crossover[klass](
                    self, child1, child2, i, type)
                child1.evaluated = False
                child2.evaluated = False

        # mutate
        for child in [child1, child2]:
            self.mutate(child)

        return [child1, child2]

    def mutate(self, child):
        problem = child.problem

        for i, type in enumerate(problem.types):  # @ReservedAssignment
            if random.random() <= self.mutation_prob:
                klass = type.__class__
                child = self._mutate[klass](self, child, i, type)
                child.evaluated = False

    def crossover_real(self, child1, child2, i, type):  # @ReservedAssignment
        # sbx
        x1 = float(child1.variables[i])
        x2 = float(child2.variables[i])
        lb = type.min_value
        ub = type.max_value

        x1, x2 = self.SBX.sbx_crossover(x1, x2, lb, ub)

        child1.variables[i] = x1
        child2.variables[i] = x2

        return child1, child2

    def crossover_integer(self, child1, child2, i, type):  # @ReservedAssignment
        # HUX()
        for j in range(type.nbits):
            if child1.variables[i][j] != child2.variables[i][j]:
                if bool(random.getrandbits(1)):
                    child1.variables[i][j] = not child1.variables[i][j]
                    child2.variables[i][j] = not child2.variables[i][j]
        return child1, child2

    def crossover_categorical(self, child1, child2, i, type):  # @ReservedAssignment
        # SSX()
        # can probably be implemented in a simple manner, since size
        # of subset is fixed to 1

        s1 = set(child1.variables[i])
        s2 = set(child2.variables[i])

        for j in range(type.size):
            if (child2.variables[i][j] not in s1) and \
               (child1.variables[i][j] not in s2) and \
               (random.random() < 0.5):
                temp = child1.variables[i][j]
                child1.variables[i][j] = child2.variables[i][j]
                child2.variables[i][j] = temp

        return child1, child2

    def mutate_real(self, child, i, type, distribution_index=20):  # @ReservedAssignment
        # PM
        x = child.variables[i]
        lower = type.min_value
        upper = type.max_value

        u = random.random()
        dx = upper - lower

        if u < 0.5:
            bl = (x - lower) / dx
            b = 2.0 * u + (1.0 - 2.0 * u) * \
                pow(1.0 - bl, distribution_index + 1.0)
            delta = pow(b, 1.0 / (distribution_index + 1.0)) - 1.0
        else:
            bu = (upper - x) / dx
            b = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * \
                pow(1.0 - bu, distribution_index + 1.0)
            delta = 1.0 - pow(b, 1.0 / (distribution_index + 1.0))

        x = x + delta * dx
        x = max(lower, min(x, upper))

        child.variables[i] = x
        return child

    def mutate_integer(self, child, i, type, probability=1):  # @ReservedAssignment
        # bitflip
        for j in range(type.nbits):
            if random.random() <= probability:
                child.variables[i][j] = not child.variables[i][j]
        return child

    def mutate_categorical(self, child, i, type):  # @ReservedAssignment
        # replace
        probability = 1 / type.size

        if random.random() <= probability:
            subset = child.variables[i]

            if len(subset) < len(type.elements):
                j = random.randrange(len(subset))

                nonmembers = list(set(type.elements) - set(subset))
                k = random.randrange(len(nonmembers))
                subset[j] = nonmembers[k]

            len(subset)

            child.variables[i] = subset

        return child

    _crossover = {Real: crossover_real,
                  Integer: crossover_integer,
                  Subset: crossover_categorical}

    _mutate = {Real: mutate_real,
               Integer: mutate_integer,
               Subset: mutate_categorical}


class CombinedMutator(CombinedVariator):
    '''Data type aware Uniform mutator

    Overwrites the mutator on the algorithm as used by adaptive time
    continuation.

    This is a dirty hack, mutator should be a keyword argument on
    epsilon-NSGAII. Would require separating out explicitly the algorithm
    kwargs and the AdaptiveTimeContinuation kwargs.

    '''

    mutation_prob = 1.0

    def evolve(self, parents):
        _logger.debug(parents)

        problem = parents[0].problem
        children = []

        for parent in parents:
            child = copy.deepcopy(parent)
            for i, type in enumerate(problem.types):  # @ReservedAssignment
                if random.random() <= self.mutation_prob:
                    klass = type.__class__
                    child = self._mutate[klass](self, child, i, type)
                    child.evaluated = False

            children.append(child)
        return children

    def mutate_categorical(self, child, i, type):  # @ReservedAssignment
        child.variables[i] = [random.choice(type.elements)]
        return child

    def mutate_integer(self, child, i, type):  # @ReservedAssignment
        child.variables[i] = type.encode(random.randint(type.min_value,
                                                        type.max_value))
        return child

    def mutate_real(self, child, i, type):  # @ReservedAssignment
        child.variables[i] = random.uniform(type.min_value, type.max_value)
        return child

    _mutate = {Real: mutate_real,
               Integer: mutate_integer,
               Subset: mutate_categorical}


def _optimize(problem, evaluator, algorithm, convergence, nfe,
              convergence_freq, logging_freq, **kwargs):

    klass = problem.types[0].__class__

    if all([isinstance(t, klass) for t in problem.types]):
        variator = None
    else:
        variator = CombinedVariator()
    mutator = CombinedMutator()

    optimizer = algorithm(problem, evaluator=evaluator, variator=variator,
                          log_frequency=500, **kwargs)
    optimizer.mutator = mutator

    convergence = Convergence(convergence, nfe,
                              convergence_freq=convergence_freq,
                              logging_freq=logging_freq)
    callback = functools.partial(convergence, optimizer)
    evaluator.callback = callback

    with temporary_filter(name=[callbacks.__name__,
                                evaluators.__name__], level=INFO):
        optimizer.run(nfe)

    results = to_dataframe(optimizer, problem.parameter_names,
                           problem.outcome_names)
    convergence = convergence.to_dataframe()

    message = "optimization completed, found {} solutions"
    _logger.info(message.format(len(optimizer.algorithm.archive)))

    if convergence.empty:
        return results
    else:
        return results, convergence


class BORGDefaultDescriptor(object):
    # this treats defaults as class level attributes!

    def __init__(self, default_function):
        self.default_function = default_function

    def __get__(self, instance, owner):
        return self.default_function(instance.problem.nvars)

    def __set_name__(self, owner, name):
        self.name = name


class GenerationalBorg(EpsilonProgressContinuation):
    '''A generational implementation of the BORG Framework

    This algorithm adopts Epsilon Progress Continuation, and Auto Adaptive
    Operator Selection, but embeds them within the NSGAII generational
    algorithm, rather than the steady state implementation used by the BORG
    algorithm.

    The parametrization of all operators is based on the default values as used
    in Borg 1.9.

    Note:: limited to RealParameters only.

    '''
    pm_p = BORGDefaultDescriptor(lambda x: 1 / x)
    pm_dist = 20

    sbx_prop = 1
    sbx_dist = 15

    de_rate = 0.1
    de_stepsize = 0.5

    um_p = BORGDefaultDescriptor(lambda x: x + 1)

    spx_nparents = 10
    spx_noffspring = 2
    spx_expansion = 0.3

    pcx_nparents = 10
    pcx_noffspring = 2
    pcx_eta = 0.1
    pcx_zeta = 0.1

    undx_nparents = 10
    undx_noffspring = 2
    undx_zeta = 0.5
    undx_eta = 0.35

    def __init__(self, problem, epsilons, population_size=100,
                 generator=RandomGenerator(), selector=TournamentSelector(2),
                 variator=None, **kwargs):
        self.problem = problem

        # Parameterization taken from
        # Borg: An Auto-Adaptive MOEA Framework - Hadka, Reed
        variators = [GAOperator(SBX(probability=self.sbx_prop,
                                    distribution_index=self.sbx_dist),
                                PM(probability=self.pm_p,
                                   distribution_index=self.pm_dist)),
                     GAOperator(PCX(nparents=self.pcx_nparents,
                                    noffspring=self.pcx_noffspring,
                                    eta=self.pcx_eta,
                                    zeta=self.pcx_zeta),
                                PM(probability=self.pm_p,
                                   distribution_index=self.pm_dist)),
                     GAOperator(DifferentialEvolution(crossover_rate=self.de_rate,
                                                      step_size=self.de_stepsize),
                                PM(probability=self.pm_p,
                                   distribution_index=self.pm_dist)),
                     GAOperator(UNDX(nparents=self.undx_nparents,
                                     noffspring=self.undx_noffspring,
                                     zeta=self.undx_zeta,
                                     eta=self.undx_eta),
                                PM(probability=self.pm_p,
                                   distribution_index=self.pm_dist)),
                     GAOperator(SPX(nparents=self.spx_nparents,
                                    noffspring=self.spx_noffspring,
                                    expansion=self.spx_expansion),
                                PM(probability=self.pm_p,
                                   distribution_index=self.pm_dist)),
                     UM(probability=self.um_p)]

        variator = Multimethod(self, variators)

        super(GenerationalBorg, self).__init__(
            NSGAII(problem,
                   population_size,
                   generator,
                   selector,
                   variator,
                   EpsilonBoxArchive(epsilons),
                   **kwargs))
