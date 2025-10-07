"""Wrapper around platypus-opt."""

import copy
import io
import os
import random
import tarfile
import time
from collections.abc import Iterable
from typing import Literal

import pandas as pd
import platypus
from platypus import (
    NSGAII,
    PCX,
    PM,
    SBX,
    SPX,
    UM,
    UNDX,
    DifferentialEvolution,
    EpsilonBoxArchive,
    GAOperator,
    InjectedPopulation,
    Integer,
    Multimethod,
    RandomGenerator,
    Real,
    Solution,
    Subset,
    TournamentSelector,
    Variator,
)
from platypus import Problem as PlatypusProblem

from ..util import INFO, EMAError, get_module_logger, temporary_filter
from . import callbacks, evaluators
from .model import AbstractModel
from .outcomes import Constraint, Outcome, ScalarOutcome
from .parameters import (
    BooleanParameter,
    CategoricalParameter,
    IntegerParameter,
    Parameter,
    RealParameter,
)
from .points import Sample
from .util import ProgressTrackingMixIn, determine_objects

# Created on 5 Jun 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = [
    "Problem",
    "RobustProblem",
    "epsilon_nondominated",
    "rebuild_platypus_population",
    "to_problem",
    "to_robust_problem",
]
_logger = get_module_logger(__name__)


class Problem(PlatypusProblem):
    """Small extension to Platypus problem object.

    Includes information on the names of the decision variables, the names of the outcomes,
    and the type of search.
    """

    @property
    def parameter_names(self) -> list[str]:
        """Getter for parameter names."""
        return [e.name for e in self.decision_variables]

    @property
    def outcome_names(self) -> list[str]:
        """Getter for outcome names."""
        return [e.name for e in self.objectives]

    def __init__(
        self,
        searchover: Literal["levers", "uncertainties", "robust"],
        decision_variables: list[Parameter],
        objectives: list[ScalarOutcome],
        constraints: list[Constraint] | None,
        reference: Sample | None = None,
    ):
        """Init."""
        if constraints is None:
            constraints = []

        super().__init__(
            len(decision_variables), len(objectives), nconstrs=len(constraints)
        )

        if searchover == "robust" and reference is not None:
            raise ValueError("you cannot use a single reference for robust search")
        for obj in objectives:
            if obj.kind == obj.INFO:
                raise ValueError(f"you need to specify the direction for objective {obj.name}, cannot be INFO")

        self.searchover = searchover
        self.decision_variables = decision_variables
        self.objectives = objectives

        self.ema_constraints = constraints
        self.constraint_names = [c.name for c in constraints]
        self.reference = reference if reference else 0

        self.types[:] = to_platypus_types(decision_variables)
        self.directions[:] = [outcome.kind for outcome in objectives]
        self.constraints[:] = "==0"


class RobustProblem(Problem):
    """Small extension to Problem object for robust optimization.

    adds the scenarios and the robustness functions
    """

    def __init__(
        self,
        decision_variables: list[Parameter],
        objectives: list[ScalarOutcome],
        scenarios: Iterable[Sample]|int,
        constraints: list[Constraint] | None=None,
    ):
        """Init."""
        # fixme, we should be able to get rid of robust problem all together?
        for objective in objectives:
            if objective.function is None:
                raise ValueError(f"no robustness function defined for {objective.name}")

        super().__init__("robust", decision_variables, objectives, constraints)
        self.scenarios = scenarios


def to_problem(
    model: AbstractModel,
    searchover: str,
    reference: Sample | None = None,
    constraints=None,
):
    """Helper function to create Problem object.

    Parameters
    ----------
    model : AbstractModel instance
    searchover : str
    reference : Sample instance, optional
                overwrite the default scenario in case of searching over
                levers, or default policy in case of searching over
                uncertainties
    constraints : list, optional

    Returns
    -------
    Problem instance

    """
    # extract the levers and the outcomes
    decision_variables = determine_objects(model, searchover, union=True)

    outcomes = determine_objects(model, "outcomes")
    outcomes = [outcome for outcome in outcomes if outcome.kind != Outcome.INFO]

    if not outcomes:
        raise EMAError(
            "No outcomes specified to optimize over, all outcomes are of kind=INFO"
        )

    problem = Problem(
        searchover, decision_variables, outcomes, constraints, reference=reference
    )

    return problem


def to_robust_problem(model, scenarios, objectives, constraints=None):
    """Helper function to create RobustProblem object.

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
    decision_variables = determine_objects(model, "levers", union=True)

    outcomes = objectives
    outcomes = [outcome for outcome in outcomes if outcome.kind != Outcome.INFO]

    if not outcomes:
        raise EMAError(
            "No outcomes specified to optimize over, all outcomes are of kind=INFO"
        )

    problem = RobustProblem(
        decision_variables, objectives, scenarios, constraints
    )

    return problem


def to_platypus_types(decision_variables):
    """Helper function for mapping from workbench parameter types to platypus parameter types."""
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

        if not isinstance(dv, (CategoricalParameter | BooleanParameter)):
            decision_variable = klass(dv.lower_bound, dv.upper_bound)
        else:
            decision_variable = klass(dv.categories, 1)

        types.append(decision_variable)
    return types


def to_dataframe(
    solutions: list[platypus.Solution], dvnames: list[str], outcome_names: list[str]
):
    """Helper function to turn a collection of platypus Solution instances into a pandas DataFrame.

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
        decision_vars = Sample._from_platypus_solution(solution)

        decision_out = dict(zip(outcome_names, solution.objectives))

        result = decision_vars.copy()
        result.update(decision_out)

        results.append(result)

    results = pd.DataFrame(results, columns=dvnames + outcome_names)
    return results


def process_jobs(jobs, searchover):
    """Helper function to map jobs generated by platypus to Sample instances.

    Parameters
    ----------
    jobs : collection

    Returns
    -------
    scenarios, policies

    """
    problem = jobs[0].solution.problem
    samples = [Sample._from_platypus_solution(job.solution) for job in jobs]

    references = problem.reference
    match searchover:
        case "levers":
            return references, samples
        case "uncertainties":
            return samples, references
        case "robust":
            return jobs[0].solution.problem.scenarios, samples
        case _:
            raise NotImplementedError(
                f"unknown value for searchover, got {searchover} should be one of 'levers', 'uncertainties', or 'robust'"
            )


# def _process(jobs, problem):
#     """Helper function to transform platypus job to dict with correct values for workbench."""
#     processed_jobs = []
#     for job in jobs:
#         variables = transform_variables(problem, job.solution.variables)
#         processed_job = {}
#         for param, var in zip(problem.parameters, variables):
#             try:
#                 var = var.value
#             except AttributeError:
#                 pass
#             processed_job[param.name] = var
#         processed_jobs.append(processed_job)
#     return processed_jobs


def evaluate(jobs_collection, experiments, outcomes, problem):
    """Helper function for mapping the results from perform_experiments back to what platypus needs."""
    searchover = problem.searchover
    outcome_names = problem.outcome_names
    constraints = problem.ema_constraints

    column = "policy" if searchover == "levers" else "scenario"

    for entry, job in jobs_collection:
        logical = experiments[column] == entry.name

        job_outputs = {}
        for k, v in outcomes.items():
            job_outputs[k] = v[logical][0]

        # TODO:: only retain uncertainties
        job_experiment = experiments[logical]
        job_constraints = _evaluate_constraints(
            job_experiment, job_outputs, constraints
        )
        job_outcomes = [job_outputs[key] for key in outcome_names]

        if job_constraints:
            job.solution.problem.function = (
                lambda _, job_outcomes=job_outcomes, job_constraints=job_constraints: (
                    job_outcomes,
                    job_constraints,
                )
            )
        else:
            job.solution.problem.function = (
                lambda _, job_outcomes=job_outcomes: job_outcomes
            )
        job.solution.evaluate()


def evaluate_robust(jobs_collection, experiments, outcomes, problem):
    """Helper function for mapping the results from perform_experiments back to what Platypus needs."""
    robustness_functions = problem.objectives
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
        job_constraints = _evaluate_constraints(
            job_experiment, job_outcomes_dict, constraints
        )

        if job_constraints:
            job.solution.problem.function = (
                lambda _, job_outcomes=job_outcomes, job_constraints=job_constraints: (
                    job_outcomes,
                    job_constraints,
                )
            )
        else:
            job.solution.problem.function = (
                lambda _, job_outcomes=job_outcomes: job_outcomes
            )

        job.solution.evaluate()


def _evaluate_constraints(job_experiment, job_outcomes, constraints):
    """Helper function for evaluating the constraints for a given job."""
    job_constraints = []
    for constraint in constraints:
        data = [job_experiment[var] for var in constraint.parameter_names]
        data += [job_outcomes[var] for var in constraint.outcome_names]
        constraint_value = constraint.process(data)
        job_constraints.append(constraint_value)
    return job_constraints


class ProgressBarExtension(platypus.extensions.FixedFrequencyExtension):
    """Small platypus extension showing a progress bar."""

    def __init__(self, total_nfe: int, frequency: int = 100):
        """Init."""
        super().__init__(frequency=frequency)
        self.progress_tracker = ProgressTrackingMixIn(
            total_nfe,
            frequency,
            _logger,
            log_func=lambda self: f"generation"
            f" {self.generation}, {self.i}/{self.max_nfe}",
        )

    def do_action(self, algorithm):
        """Update the progress bar."""
        nfe = algorithm.nfe
        self.progress_tracker(nfe - self.progress_tracker.i)


class ArchiveStorageExtension(platypus.extensions.FixedFrequencyExtension):
    """Extension that stores the archive to a tarball at a fixed frequency.

    Parameters
    ----------
    directory : str
    decision_variable_names : list of the names of the decision variables
    outcome_names : list of names of the outcomes of interest
    filename : the name of the tarball
    frequency : int
        The frequency the action occurs.
    by_nfe : bool
        If :code:`True`, the frequency is given in number of function
        evaluations.  If :code:`False`, the frequency is given in the number
        of iterations.

    Raises
    ------
    FileExistsError if tarfile already exists.

    """

    def __init__(
        self,
        decision_variable_names: list[str],
        outcome_names: list[str],
        directory: str | None = None,
        filename: str | None = None,
        frequency: int = 1000,
        by_nfe: bool = True,
    ):
        super().__init__(frequency=frequency, by_nfe=by_nfe)
        self.decision_variable_names = decision_variable_names
        self.outcome_names = outcome_names
        self.temp = os.path.join(directory, "tmp")
        self.tar_filename = os.path.join(os.path.abspath(directory), filename)

        if os.path.exists(self.tar_filename):
            raise FileExistsError(
                f"File {self.tar_filename} for storing the archives already exists."
            )

    def do_action(self, algorithm: platypus.algorithms.AbstractGeneticAlgorithm):
        """Add the current archive to the tarball."""
        # broadens the algorithms in platypus we can support automagically
        try:
            data = algorithm.archive
        except AttributeError:
            data = algorithm.result

        # fixme, this opens and closes the tarball everytime
        #   can't we open in in the init and have a clean way to close it
        #   on any exit?
        with tarfile.open(self.tar_filename, "a") as f:
            archive = to_dataframe(
                data, self.decision_variable_names, self.outcome_names
            )
            stream = io.BytesIO()
            archive.to_csv(stream, encoding="UTF-8", index=False)
            stream.seek(0)
            tarinfo = tarfile.TarInfo(f"{algorithm.nfe}.csv")
            tarinfo.size = len(stream.getbuffer())
            tarinfo.mtime = time.time()
            f.addfile(tarinfo, stream)


def epsilon_nondominated(results, epsilons, problem):
    """Merge the list of results into a single set of non dominated results using the provided epsilon values.

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
        raise ValueError(
            f"The number of epsilon values ({len(epsilons)}) must match the number of objectives {problem.nobjs}"
        )

    results = pd.concat(results, ignore_index=True)
    solutions = rebuild_platypus_population(results, problem)
    archive = EpsilonBoxArchive(epsilons)
    archive += solutions

    return to_dataframe(archive, problem.parameter_names, problem.outcome_names)


def rebuild_platypus_population(archive, problem):
    """Rebuild a population of platypus Solution instances.

    Parameters
    ----------
    archive : DataFrame
    problem : PlatypusProblem instance

    Returns
    -------
    list of platypus Solutions

    """
    expected_columns = problem.nvars + problem.nobjs
    actual_columns = len(archive.columns)

    if actual_columns != expected_columns:
        raise EMAError(
            f"The number of columns in the archive ({actual_columns}) does not match the "
            f"expected number of decision variables and objectives ({expected_columns})."
        )

    solutions = []
    for row in archive.itertuples():
        try:
            decision_variables = [
                getattr(row, attr) for attr in problem.parameter_names
            ]
        except AttributeError as e:
            missing_parameters = [
                attr for attr in problem.parameter_names if not hasattr(row, attr)
            ]
            raise EMAError(
                f"Parameter names {missing_parameters} not found in archive"
            ) from e

        try:
            objectives = [getattr(row, attr) for attr in problem.outcome_names]
        except AttributeError as e:
            missing_outcomes = [
                attr for attr in problem.outcome_names if not hasattr(row, attr)
            ]
            raise EMAError(
                f"Outcome names {missing_outcomes} not found in archive'"
            ) from e

        solution = Solution(problem)
        solution.variables[:] = [
            platypus_type.encode(value)
            for platypus_type, value in zip(problem.types, decision_variables)
        ]
        solution.objectives[:] = objectives
        solutions.append(solution)
    return solutions


class CombinedVariator(Variator):
    """Combined variator."""

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
            if child1.variables[i][j] != child2.variables[i][j]:  # noqa: SIM102
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
            b = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * pow(
                1.0 - bu, distribution_index + 1.0
            )
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
    problem: Problem,
    evaluator: "BaseEvaluator",  # noqa: F821
    algorithm: type[platypus.algorithms.AbstractGeneticAlgorithm],
    nfe: int,
    convergence_freq: int,
    logging_freq: int,
    variator: Variator = None,
    initial_population: Iterable[Sample] | None = None,
    filename: str | None = None,
    directory: str | None = None,
    **kwargs,
):
    """Helper function for optimization."""
    klass = problem.types[0].__class__

    try:
        eps_values = kwargs["epsilons"]
    except KeyError:
        pass
    else:
        if len(eps_values) != len(problem.outcome_names):
            raise ValueError(
                "Number of epsilon values does not match number of outcomes"
            )

    if variator is None:
        if all(isinstance(t, klass) for t in problem.types):
            variator = None
        else:
            variator = CombinedVariator()

    generator = (
        RandomGenerator()
        if initial_population is None
        else InjectedPopulation(
            [sample._to_platypus_solution(problem) for sample in initial_population]
        )
    )

    optimizer = algorithm(
        problem,
        evaluator=evaluator,
        variator=variator,
        log_frequency=500,
        generator=generator,
        **kwargs,
    )
    storage = ArchiveStorageExtension(
        problem.parameter_names,
        problem.outcome_names,
        directory=directory,
        filename=filename,
        frequency=convergence_freq,
        by_nfe=True,
    )
    progress_bar = ProgressBarExtension(nfe, frequency=logging_freq)
    optimizer.add_extension(storage)
    optimizer.add_extension(progress_bar)

    with temporary_filter(name=[callbacks.__name__, evaluators.__name__], level=INFO):
        optimizer.run(nfe)

    storage.do_action(
        optimizer
    )  # ensure last archive is included in the convergence information
    progress_bar.progress_tracker.pbar.__exit__(
        None, None, None
    )  # ensure progress bar is closed correctly

    try:
        data = optimizer.archive
    except AttributeError:
        data = optimizer.result

    results = to_dataframe(data, problem.parameter_names, problem.outcome_names)

    _logger.info(f"optimization completed, found {len(data)} solutions")

    return results


class BORGDefaultDescriptor:
    """Descriptor used by Borg."""

    # this treats defaults as class level attributes!

    def __init__(self, default_function):
        self.default_function = default_function

    def __get__(self, instance, owner):
        return self.default_function(instance.problem.nvars)

    def __set_name__(self, owner, name):
        self.name = name


class GenerationalBorg(NSGAII):
    """A generational implementation of the BORG Framework.

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

    um_p = BORGDefaultDescriptor(lambda x: 1 / x)

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
        problem: Problem,
        epsilons: list[float],
        population_size: int = 100,
        generator: platypus.Generator = RandomGenerator(),  # noqa: B008
        selector: platypus.Selector = TournamentSelector(2),  # noqa: B008
        **kwargs,
    ):
        """Init."""
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
                DifferentialEvolution(
                    crossover_rate=self.de_rate, step_size=self.de_stepsize
                ),
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
            problem,
            population_size,
            generator,
            selector,
            variator,
            EpsilonBoxArchive(epsilons),
            **kwargs,
        )
        self.add_extension(platypus.extensions.EpsilonProgressContinuationExtension())
