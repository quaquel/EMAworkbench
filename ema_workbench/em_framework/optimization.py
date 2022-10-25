"""


"""
import copy
import functools
import os
import random
import shutil
import tarfile
import warnings

import numpy as np
import pandas as pd


from . import callbacks, evaluators
from .points import Scenario, Policy
from .outcomes import AbstractOutcome
from .parameters import IntegerParameter, RealParameter, CategoricalParameter, BooleanParameter
from .samplers import determine_parameters
from .util import determine_objects, ProgressTrackingMixIn
from ..util import get_module_logger, EMAError, temporary_filter, INFO

try:
    from platypus import (
        EpsNSGAII,
        Hypervolume,
        EpsilonIndicator,
        GenerationalDistance,
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
        UM,
        Solution,
        InvertedGenerationalDistance,
        Spacing,
    )  # @UnresolvedImport
    from platypus import Problem as PlatypusProblem

    import platypus


except ImportError:
    warnings.warn("platypus based optimization not available", ImportWarning)

    class PlatypusProblem:
        constraints = []

        def __init__(self, *args, **kwargs):
            pass

    class Variator:
        def __init__(self, *args, **kwargs):
            pass

    class RandomGenerator:
        def __call__(self, *args, **kwargs):
            pass

    class TournamentSelector:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            pass

    class EpsilonProgressContinuation:
        pass

    EpsNSGAII = None
    platypus = None
    Real = Integer = Subset = None

# Created on 5 Jun 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = [
    "Problem",
    "RobustProblem",
    "EpsilonProgress",
    "Convergence",
    "ArchiveLogger",
    "OperatorProbabilities",
    "rebuild_platypus_population",
    "HypervolumeMetric",
    "GenerationalDistanceMetric",
    "SpacingMetric",
    "InvertedGenerationalDistanceMetric",
    "EpsilonIndicatorMetric",
    "epsilon_nondominated",
    "to_problem",
    "to_robust_problem",
]
_logger = get_module_logger(__name__)


class Problem(PlatypusProblem):
    """small extension to Platypus problem object, includes information on
    the names of the decision variables, the names of the outcomes,
    and the type of search"""

    @property
    def parameter_names(self):
        return [e.name for e in self.parameters]

    def __init__(self, searchover, parameters, outcome_names, constraints, reference=None):
        if constraints is None:
            constraints = []

        super().__init__(len(parameters), len(outcome_names), nconstrs=len(constraints))
        #         assert len(parameters) == len(parameter_names)
        assert searchover in ("levers", "uncertainties", "robust")

        if searchover == "levers":
            assert not reference or isinstance(reference, Scenario)
        elif searchover == "uncertainties":
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
    """small extension to Problem object for robust optimization, adds the
    scenarios and the robustness functions"""

    def __init__(self, parameters, outcome_names, scenarios, robustness_functions, constraints):
        super().__init__("robust", parameters, outcome_names, constraints)
        assert len(robustness_functions) == len(outcome_names)
        self.scenarios = scenarios
        self.robustness_functions = robustness_functions


def to_problem(model, searchover, reference=None, constraints=None):
    """helper function to create Problem object

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

    """

    # extract the levers and the outcomes
    decision_variables = determine_parameters(model, searchover, union=True)

    outcomes = determine_objects(model, "outcomes")
    outcomes = [outcome for outcome in outcomes if outcome.kind != AbstractOutcome.INFO]
    outcome_names = [outcome.name for outcome in outcomes]

    if not outcomes:
        raise EMAError("no outcomes specified to optimize over, " "all outcomes are of kind=INFO")

    problem = Problem(
        searchover, decision_variables, outcome_names, constraints, reference=reference
    )
    problem.types = to_platypus_types(decision_variables)
    problem.directions = [outcome.kind for outcome in outcomes]
    problem.constraints[:] = "==0"

    return problem


def to_robust_problem(model, scenarios, robustness_functions, constraints=None):
    """helper function to create RobustProblem object

    Parameters
    ----------
    model : AbstractModel instance
    scenarios : collection
    robustness_functions : iterable of ScalarOutcomes
    constraints : list, optional

    Returns
    -------
    RobustProblem instance

    """

    # extract the levers and the outcomes
    decision_variables = determine_parameters(model, "levers", union=True)

    outcomes = robustness_functions
    outcomes = [outcome for outcome in outcomes if outcome.kind != AbstractOutcome.INFO]
    outcome_names = [outcome.name for outcome in outcomes]

    if not outcomes:
        raise EMAError("no outcomes specified to optimize over, " "all outcomes are of kind=INFO")

    problem = RobustProblem(
        decision_variables, outcome_names, scenarios, robustness_functions, constraints
    )

    problem.types = to_platypus_types(decision_variables)
    problem.directions = [outcome.kind for outcome in outcomes]
    problem.constraints[:] = "==0"

    return problem


def to_platypus_types(decision_variables):
    """helper function for mapping from workbench parameter types to
    platypus parameter types"""
    # TODO:: should categorical not be platypus.Subset, with size == 1?
    _type_mapping = {
        RealParameter: platypus.Real,
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


def to_dataframe(solutions, dvnames, outcome_names):
    """helper function to turn a collection of platypus Solution instances
    into a pandas DataFrame
    Parameters
    ----------
    solutions : collection of Solution instances
    dvnames : list of str
    outcome_names : list of str

    Returns
    -------
    pandas DataFrame
    """

    results = []
    for solution in platypus.unique(solutions):
        vars = transform_variables(solution.problem, solution.variables)  # @ReservedAssignment

        decision_vars = dict(zip(dvnames, vars))
        decision_out = dict(zip(outcome_names, solution.objectives))

        result = decision_vars.copy()
        result.update(decision_out)

        results.append(result)

    results = pd.DataFrame(results, columns=dvnames + outcome_names)
    return results


def process_uncertainties(jobs):
    """helper function to map jobs generated by platypus to Scenario objects

    Parameters
    ----------
    jobs : collection

    Returns
    -------
    scenarios, policies

    """
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
    """helper function to map jobs generated by platypus to Policy objects

    Parameters
    ----------
    jobs : collection

    Returns
    -------
    scenarios, policies

    """
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
    """helper function to transform platypus job to dict with correct
    values for workbench"""

    processed_jobs = []
    for job in jobs:
        variables = transform_variables(problem, job.solution.variables)
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
    """Helper function to process robust optimization jobs

    Parameters
    ----------
    jobs : collection

    Returns
    -------
    scenarios, policies

    """
    _, policies = process_levers(jobs)
    scenarios = jobs[0].solution.problem.scenarios

    return scenarios, policies


def transform_variables(problem, variables):
    """helper function for transforming platypus variables"""

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
    """Helper function for mapping the results from perform_experiments back
    to what platypus needs"""

    searchover = problem.searchover
    outcome_names = problem.outcome_names
    constraints = problem.ema_constraints

    if searchover == "levers":
        column = "policy"
    else:
        column = "scenario"

    for entry, job in jobs_collection:
        logical = experiments[column] == entry.name

        job_outputs = {}
        for k, v in outcomes.items():
            job_outputs[k] = v[logical][0]

        # TODO:: only retain uncertainties
        job_experiment = experiments[logical]
        job_constraints = _evaluate_constraints(job_experiment, job_outputs, constraints)
        job_outcomes = [job_outputs[key] for key in outcome_names]

        if job_constraints:
            job.solution.problem.function = lambda _: (job_outcomes, job_constraints)
        else:
            job.solution.problem.function = lambda _: job_outcomes
        job.solution.evaluate()


def evaluate_robust(jobs_collection, experiments, outcomes, problem):
    """Helper function for mapping the results from perform_experiments back
    to what Platypus needs"""

    robustness_functions = problem.robustness_functions
    constraints = problem.ema_constraints

    for entry, job in jobs_collection:
        logical = experiments["policy"] == entry.name

        job_outcomes_dict = {}
        job_outcomes = []
        for rf in robustness_functions:
            data = [outcomes[var_name][logical] for var_name in rf.variable_name]
            score = rf.function(*data)
            job_outcomes_dict[rf.name] = score
            job_outcomes.append(score)

        # TODO:: only retain levers
        job_experiment = experiments[logical].iloc[0]
        job_constraints = _evaluate_constraints(job_experiment, job_outcomes_dict, constraints)

        if job_constraints:
            job.solution.problem.function = lambda _: (job_outcomes, job_constraints)
        else:
            job.solution.problem.function = lambda _: job_outcomes

        job.solution.evaluate()


def _evaluate_constraints(job_experiment, job_outcomes, constraints):
    """Helper function for evaluating the constraints for a given job"""
    job_constraints = []
    for constraint in constraints:
        data = [job_experiment[var] for var in constraint.parameter_names]
        data += [job_outcomes[var] for var in constraint.outcome_names]
        constraint_value = constraint.process(data)
        job_constraints.append(constraint_value)
    return job_constraints


class AbstractConvergenceMetric:
    """base convergence metric class"""

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.results = []

    def __call__(self, optimizer):
        raise NotImplementedError

    def reset(self):
        self.results = []

    def get_results(self):
        return self.results


class EpsilonProgress(AbstractConvergenceMetric):
    """epsilon progress convergence metric class"""

    def __init__(self):
        super().__init__("epsilon_progress")

    def __call__(self, optimizer):
        self.results.append(optimizer.algorithm.archive.improvements)


class MetricWrapper:
    f"""wrapper class for wrapping platypus indicators

    Parameters
    ----------
    reference_set : DataFrame
    problem : PlatypusProblem instance
    kwargs : dict
             any additional keyword arguments to be passed
             on to the wrapper platypus indicator class

    Notes
    -----
    this class relies on multi-inheritance and careful consideration
    of the MRO to conveniently wrap the convergence metrics provided
    by platypus.

    """

    def __init__(self, reference_set, problem, **kwargs):
        self.problem = problem
        reference_set = rebuild_platypus_population(reference_set, self.problem)
        super().__init__(reference_set=reference_set, **kwargs)

    def calculate(self, archive):
        solutions = rebuild_platypus_population(archive, self.problem)
        return super().calculate(solutions)


class HypervolumeMetric(MetricWrapper, Hypervolume):
    """Hypervolume metric

    Parameters
    ----------
    reference_set : DataFrame
    problem : PlatypusProblem instance


    this is a thin wrapper around Hypervolume as provided
    by platypus to make it easier to use in conjunction with the
    workbench.

    """

    pass


class GenerationalDistanceMetric(MetricWrapper, GenerationalDistance):
    """GenerationalDistance metric

    Parameters
    ----------
    reference_set : DataFrame
    problem : PlatypusProblem instance
    d : int, default=1
        the power in the intergenerational distance function


    This is a thin wrapper around GenerationalDistance as provided
    by platypus to make it easier to use in conjunction with the
    workbench.

    see https://link.springer.com/content/pdf/10.1007/978-3-319-15892-1_8.pdf
    for more information

    """

    pass


class InvertedGenerationalDistanceMetric(MetricWrapper, InvertedGenerationalDistance):
    """InvertedGenerationalDistance metric

    Parameters
    ----------
    reference_set : DataFrame
    problem : PlatypusProblem instance
    d : int, default=1
        the power in the inverted intergenerational distance function


    This is a thin wrapper around InvertedGenerationalDistance as provided
    by platypus to make it easier to use in conjunction with the
    workbench.

    see https://link.springer.com/content/pdf/10.1007/978-3-319-15892-1_8.pdf
    for more information

    """

    pass


class EpsilonIndicatorMetric(MetricWrapper, EpsilonIndicator):
    """EpsilonIndicator metric

    Parameters
    ----------
    reference_set : DataFrame
    problem : PlatypusProblem instance


    this is a thin wrapper around EpsilonIndicator as provided
    by platypus to make it easier to use in conjunction with the
    workbench.

    """

    pass


class SpacingMetric(MetricWrapper, Spacing):
    """Spacing metric

    Parameters
    ----------
    problem : PlatypusProblem instance


    this is a thin wrapper around Spacing as provided
    by platypus to make it easier to use in conjunction with the
    workbench.

    """

    def __init__(self, problem):
        self.problem = problem


class HyperVolume(AbstractConvergenceMetric):
    """Hypervolume convergence metric class

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


    This class is deprecated. Use ArchiveLogger instead and calculate hypervolume
    in post using HypervolumeMetric as also shown in the directed search tutorial.

    """

    def __init__(self, minimum, maximum):
        super().__init__("hypervolume")
        warnings.warn(
            "HyperVolume is deprecated, use ArchiveLogger and HypervolumeMetric instead",
            warnings.DeprecationWarning,
        )
        self.hypervolume_func = Hypervolume(minimum=minimum, maximum=maximum)

    def __call__(self, optimizer):
        self.results.append(self.hypervolume_func.calculate(optimizer.algorithm.archive))

    @classmethod
    def from_outcomes(cls, outcomes):
        ranges = [o.expected_range for o in outcomes if o.kind != o.INFO]
        minimum, maximum = np.asarray(list(zip(*ranges)))
        return cls(minimum, maximum)


class ArchiveLogger(AbstractConvergenceMetric):
    """Helper class to write the archive to disk at each iteration

    Parameters
    ----------
    directory : str
    decision_varnames : list of str
    outcome_varnames : list of str
    base_filename : str, optional
    """

    def __init__(
        self, directory, decision_varnames, outcome_varnames, base_filename="archives.tar.gz"
    ):
        super().__init__("archive_logger")

        # FIXME how to handle case where directory already exists
        self.directory = os.path.abspath(directory)
        self.temp = os.path.join(self.directory, "tmp")
        os.mkdir(self.temp)

        self.base = base_filename
        self.decision_varnames = decision_varnames
        self.outcome_varnames = outcome_varnames
        self.tarfilename = os.path.join(self.directory, base_filename)

        # self.index = 0

    def __call__(self, optimizer):
        archive = to_dataframe(optimizer.result, self.decision_varnames, self.outcome_varnames)
        archive.to_csv(os.path.join(self.temp, f"{optimizer.nfe}.csv"))

    def reset(self):
        # FIXME what needs to go here?
        pass

    def get_results(self):
        with tarfile.open(self.tarfilename, "w:gz") as z:
            z.add(self.temp, arcname=os.path.basename(self.temp))

        shutil.rmtree(self.temp)
        return None

    @classmethod
    def load_archives(cls, filename):
        """load the archives stored with the ArchiveLogger

        Parameters
        ----------
        filename : str
                   relative path to file

        Returns
        -------
        dict with nfe as key and dataframe as vlaue
        """

        archives = {}
        with tarfile.open(os.path.abspath(filename)) as fh:
            for entry in fh.getmembers():
                if entry.name.endswith("csv"):
                    key = entry.name.split("/")[1][:-4]
                    archives[int(key)] = pd.read_csv(fh.extractfile(entry))
        return archives


class OperatorProbabilities(AbstractConvergenceMetric):
    """OperatorProbabiliy convergence tracker for use with
    auto adaptive operator selection.

    Parameters
    ----------
    name : str
    index : int


    State of the art MOEAs like Borg (and GenerationalBorg provided by the workbench)
    use autoadaptive operator selection. The algorithm has multiple different evolutionary
    operators. Over the run, it tracks how well each operator is doing in producing fitter
    offspring. The probability of the algorithm using a given evolutionary operator is
    proportional to how well this operator has been doing in producing fitter offspring in
    recent generations. This class can be used to track these probabilities over the
    run of the algorithm.

    """

    def __init__(self, name, index):
        super().__init__(name)
        self.index = index

    def __call__(self, optimizer):
        try:
            props = optimizer.algorithm.variator.probabilities
            self.results.append(props[self.index])
        except AttributeError:
            pass


def epsilon_nondominated(results, epsilons, problem):
    """Merge the list of results into a single set of
    non dominated results using the provided epsilon values

    Parameters
    ----------
    results : list of DataFrames
    epsilons : epsilon values for each objective
    problem : PlatypusProblem instance

    Returns
    -------
    DataFrame
    Notes
    -----
    this is a platypus based alternative to pareto.py (https://github.com/matthewjwoodruff/pareto.py)
    """
    if problem.nobjs != len(epsilons):
        ValueError(
            f"the number of epsilon values ({len(epsilons)}) must match the number of objectives {problem.nobjs}"
        )

    results = pd.concat(results, ignore_index=True)
    solutions = rebuild_platypus_population(results, problem)
    archive = EpsilonBoxArchive(epsilons)
    archive += solutions

    return to_dataframe(archive, problem.parameter_names, problem.outcome_names)


class Convergence(ProgressTrackingMixIn):
    """helper class for tracking convergence of optimization"""

    valid_metrics = {"hypervolume", "epsilon_progress", "archive_logger"}

    def __init__(self, metrics, max_nfe, convergence_freq=1000, logging_freq=5, log_progress=False):
        super().__init__(
            max_nfe,
            logging_freq,
            _logger,
            log_progress=log_progress,
            log_func=lambda self: f"generation" f" {self.generation}, {self.i}/{self.max_nfe}",
        )

        self.max_nfe = max_nfe
        self.generation = -1
        self.index = []
        self.last_check = 0

        if metrics is None:
            metrics = []

        self.metrics = metrics
        self.convergence_freq = convergence_freq
        self.logging_freq = logging_freq

        # TODO what is the point of this code?
        for metric in metrics:
            assert isinstance(metric, AbstractConvergenceMetric)
            metric.reset()

    def __call__(self, optimizer, force=False):
        """Stores convergences information given specified convergence
        frequency.

        Parameters
        ----------
        optimizer : platypus optimizer instance
        force : boolean, optional
                if True, convergence information will always be stored
                if False, converge information will be stored if the
                the number of nfe since the last time of storing is equal to
                or higher then convergence_freq


        the primary use case for force is to force convergence frequency information
        to be stored once the stopping condition of the optimizer has been reached
        so that the final convergence information is kept.

        """
        nfe = optimizer.nfe
        super().__call__(nfe - self.i)

        self.generation += 1

        if (nfe >= self.last_check + self.convergence_freq) or (self.last_check == 0) or force:
            self.index.append(nfe)
            self.last_check = nfe

            for metric in self.metrics:
                metric(optimizer)

    def to_dataframe(self):
        progress = {
            metric.name: result for metric in self.metrics if (result := metric.get_results())
        }

        progress = pd.DataFrame.from_dict(progress)

        if not progress.empty:
            progress["nfe"] = self.index

        return progress


def rebuild_platypus_population(archive, problem):
    """rebuild a population of platypus Solution instances

    Parameters
    ----------
    archive : DataFrame
    problem : PlatypusProblem instance

    Returns
    -------
    list of platypus Solutions

    """
    solutions = []
    for row in archive.itertuples():
        decision_variables = [getattr(row, attr) for attr in problem.parameter_names]
        objectives = [getattr(row, attr) for attr in problem.outcome_names]

        solution = Solution(problem)
        solution.variables = decision_variables
        solution.objectives = objectives
        solutions.append(solution)
    return solutions


class CombinedVariator(Variator):
    def __init__(self, crossover_prob=0.5, mutation_prob=1):
        super().__init__(2)
        self.SBX = platypus.SBX()
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

    def evolve(self, parents):
        child1 = copy.deepcopy(parents[0])
        child2 = copy.deepcopy(parents[1])
        problem = child1.problem

        # crossover
        # we will evolve the individual
        for i, kind in enumerate(problem.types):  # @ReservedAssignment
            if random.random() <= self.crossover_prob:
                klass = kind.__class__
                child1, child2 = self._crossover[klass](self, child1, child2, i, kind)
                child1.evaluated = False
                child2.evaluated = False

        # mutate
        for child in [child1, child2]:
            self.mutate(child)

        return [child1, child2]

    def mutate(self, child):
        problem = child.problem

        for i, kind in enumerate(problem.types):  # @ReservedAssignment
            if random.random() <= self.mutation_prob:
                klass = kind.__class__
                child = self._mutate[klass](self, child, i, kind)
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
            if (
                (child2.variables[i][j] not in s1)
                and (child1.variables[i][j] not in s2)
                and (random.random() < 0.5)
            ):
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
            b = 2.0 * u + (1.0 - 2.0 * u) * pow(1.0 - bl, distribution_index + 1.0)
            delta = pow(b, 1.0 / (distribution_index + 1.0)) - 1.0
        else:
            bu = (upper - x) / dx
            b = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * pow(1.0 - bu, distribution_index + 1.0)
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

    _crossover = {
        Real: crossover_real,
        Integer: crossover_integer,
        Subset: crossover_categorical,
    }

    _mutate = {
        Real: mutate_real,
        Integer: mutate_integer,
        Subset: mutate_categorical,
    }


def _optimize(
    problem,
    evaluator,
    algorithm,
    convergence,
    nfe,
    convergence_freq,
    logging_freq,
    variator=None,
    **kwargs,
):
    klass = problem.types[0].__class__

    try:
        eps_values = kwargs["epsilons"]
    except KeyError:
        pass
    else:
        if len(eps_values) != len(problem.outcome_names):
            raise EMAError("number of epsilon values does not match number " "of outcomes")

    if variator is None:
        if all(isinstance(t, klass) for t in problem.types):
            variator = None
        else:
            variator = CombinedVariator()
    # mutator = CombinedMutator()

    optimizer = algorithm(
        problem, evaluator=evaluator, variator=variator, log_frequency=500, **kwargs
    )
    # optimizer.mutator = mutator

    convergence = Convergence(
        convergence, nfe, convergence_freq=convergence_freq, logging_freq=logging_freq
    )
    callback = functools.partial(convergence, optimizer)
    evaluator.callback = callback

    with temporary_filter(name=[callbacks.__name__, evaluators.__name__], level=INFO):
        optimizer.run(nfe)

    convergence(optimizer, force=True)

    # convergence.pbar.__exit__(None, None, None)

    results = to_dataframe(optimizer.result, problem.parameter_names, problem.outcome_names)
    convergence = convergence.to_dataframe()

    message = "optimization completed, found {} solutions"
    _logger.info(message.format(len(optimizer.archive)))

    if convergence.empty:
        return results
    else:
        return results, convergence


class BORGDefaultDescriptor:
    # this treats defaults as class level attributes!

    def __init__(self, default_function):
        self.default_function = default_function

    def __get__(self, instance, owner):
        return self.default_function(instance.problem.nvars)

    def __set_name__(self, owner, name):
        self.name = name


class GenerationalBorg(EpsilonProgressContinuation):
    """A generational implementation of the BORG Framework

    This algorithm adopts Epsilon Progress Continuation, and Auto Adaptive
    Operator Selection, but embeds them within the NSGAII generational
    algorithm, rather than the steady state implementation used by the BORG
    algorithm.

    The parametrization of all operators is based on the default values as used
    in Borg 1.9.

    Note:: limited to RealParameters only.

    """

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

    def __init__(
        self,
        problem,
        epsilons,
        population_size=100,
        generator=RandomGenerator(),
        selector=TournamentSelector(2),
        variator=None,
        **kwargs,
    ):
        self.problem = problem

        # Parameterization taken from
        # Borg: An Auto-Adaptive MOEA Framework - Hadka, Reed
        variators = [
            GAOperator(
                SBX(probability=self.sbx_prop, distribution_index=self.sbx_dist),
                PM(probability=self.pm_p, distribution_index=self.pm_dist),
            ),
            GAOperator(
                PCX(
                    nparents=self.pcx_nparents,
                    noffspring=self.pcx_noffspring,
                    eta=self.pcx_eta,
                    zeta=self.pcx_zeta,
                ),
                PM(probability=self.pm_p, distribution_index=self.pm_dist),
            ),
            GAOperator(
                DifferentialEvolution(crossover_rate=self.de_rate, step_size=self.de_stepsize),
                PM(probability=self.pm_p, distribution_index=self.pm_dist),
            ),
            GAOperator(
                UNDX(
                    nparents=self.undx_nparents,
                    noffspring=self.undx_noffspring,
                    zeta=self.undx_zeta,
                    eta=self.undx_eta,
                ),
                PM(probability=self.pm_p, distribution_index=self.pm_dist),
            ),
            GAOperator(
                SPX(
                    nparents=self.spx_nparents,
                    noffspring=self.spx_noffspring,
                    expansion=self.spx_expansion,
                ),
                PM(probability=self.pm_p, distribution_index=self.pm_dist),
            ),
            UM(probability=self.um_p),
        ]

        variator = Multimethod(self, variators)

        super().__init__(
            NSGAII(
                problem,
                population_size,
                generator,
                selector,
                variator,
                EpsilonBoxArchive(epsilons),
                **kwargs,
            )
        )
