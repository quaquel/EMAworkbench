"""Wrapper around platypus-opt."""

import contextlib
import copy
import io
import os
import random
import tarfile
import time
from collections.abc import Iterable
from typing import Literal

import numpy as np
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
from .outcomes import Constraint, ScalarOutcome
from .parameters import (
    BooleanParameter,
    CategoricalParameter,
    IntegerParameter,
    Parameter,
    RealParameter,
)
from .points import Sample
from .util import ProgressTrackingMixIn

# Created on 5 Jun 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = [
    "GenerationalBorg",
    "Problem",
    "epsilon_nondominated",
    "load_archives",
    "rebuild_platypus_population",
]
_logger = get_module_logger(__name__)


SeedLike = int | float | str | bytes | bytearray  # seedlike for stdlib random.seed


class Problem(PlatypusProblem):
    """Small extension to Platypus problem object.

    Includes the decision variables, outcomes, and constraints,
    any reference Sample(s), and the type of search.

    """

    @property
    def parameter_names(self) -> list[str]:
        """Getter for parameter names."""
        return [e.name for e in self.decision_variables]

    @property
    def outcome_names(self) -> list[str]:
        """Getter for outcome names."""
        return [e.name for e in self.objectives]

    @property
    def constraint_names(self) -> list[str]:
        """Getter for constraint names."""
        return [c.name for c in self.ema_constraints]

    def __init__(
        self,
        searchover: Literal["levers", "uncertainties", "robust"],
        decision_variables: list[Parameter],
        objectives: list[ScalarOutcome],
        constraints: list[Constraint] | None = None,
        reference: Sample | Iterable[Sample] | int | None = None,
    ):
        """Init."""
        if constraints is None:
            constraints = []
        if reference is None:
            reference = 1

        super().__init__(
            len(decision_variables), len(objectives), nconstrs=len(constraints)
        )

        # fixme we can probably get rid of 'robust'
        #    just flip to robust if reference is an iterable
        #    handle most value error checks inside optimize and robust_optimize instead of here
        if (searchover == "robust") and (
            (reference == 1) or isinstance(reference, Sample)
        ):
            raise ValueError(
                "you cannot use a no or a  single reference scenario for robust optimization"
            )
        for obj in objectives:
            if obj.kind == obj.INFO:
                raise ValueError(
                    f"you need to specify the direction for objective {obj.name}, cannot be INFO"
                )

        self.searchover = searchover
        self.decision_variables = decision_variables
        self.objectives = objectives

        self.ema_constraints = constraints
        self.reference = reference

        self.types[:] = to_platypus_types(decision_variables)
        self.directions[:] = [outcome.kind for outcome in objectives]
        self.constraints[:] = "==0"


def to_platypus_types(decision_variables: Iterable[Parameter]) -> list[platypus.Type]:
    """Helper function for mapping from workbench parameter types to platypus parameter types."""
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
    solutions: Iterable[platypus.Solution], dvnames: list[str], outcome_names: list[str]
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


def process_jobs(jobs: list[platypus.core.EvaluateSolution]):
    """Helper function to map jobs generated by platypus to Sample instances.

    Parameters
    ----------
    jobs : collection

    Returns
    -------
    scenarios, policies

    """
    problem = jobs[0].solution.problem
    searchover = problem.searchover
    references = problem.reference

    samples = [Sample._from_platypus_solution(job.solution) for job in jobs]
    match searchover:
        case "levers":
            return references, samples
        case "uncertainties":
            return samples, references
        case "robust":
            return references, samples
        case _:
            raise ValueError(
                f"unknown value for searchover, got {searchover} should be one of 'levers', 'uncertainties', or 'robust'"
            )


def evaluate(
    jobs_collection: Iterable[tuple[Sample, platypus.core.EvaluateSolution]],
    experiments: pd.DataFrame,
    outcomes: dict[str, np.ndarray],
    problem: Problem,
):
    """Helper function for mapping the results from perform_experiments back to what platypus needs."""
    searchover = problem.searchover
    outcome_names = problem.outcome_names
    constraints = problem.ema_constraints

    column = "scenario" if searchover == "uncertainties" else "policy"

    for sample, job in jobs_collection:
        logical = experiments[column] == sample.name

        job_outputs = {}
        for k, v in outcomes.items():
            job_outputs[k] = v[logical]

        # TODO:: only retain decision variables
        job_experiment = experiments[logical]

        if searchover == "levers" or searchover == "uncertainties":
            job_outputs = {k: v[0] for k, v in job_outputs.items()}
        else:
            robustness_scores = {}
            for obj in problem.objectives:
                data = [outcomes[var_name] for var_name in obj.variable_name]
                score = obj.function(*data)
                robustness_scores[obj.name] = score
            job_outputs = robustness_scores
            job_experiment = job_experiment.iloc[
                0
            ]  # we only need a single row with the levers here

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


def _evaluate_constraints(
    job_experiment: pd.Series,
    job_outcomes: dict[str, float | int],
    constraints: list[Constraint],
):
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


class RuntimeConvergenceTracking(platypus.extensions.FixedFrequencyExtension):
    """Platypus Extension for tracking runtime convergence information.

    This extension tracks runtime information that cannot be retrieved from the archives that are stored. Specifically,
    it automatically tries to track epsilon progress and the operator probabilities in case of a MultiMethod
    variator.

    """

    def __init__(
        self,
        frequency: int = 1000,
        by_nfe: bool = True,
    ):
        super().__init__(frequency=frequency, by_nfe=by_nfe)
        self.data = []
        self.attributes_to_try = ["nfe"]

    def do_action(self, algorithm: platypus.algorithms.AbstractGeneticAlgorithm):
        """Retrieve the runtime convergence information."""
        runtime_info = {}
        runtime_info["nfe"] = algorithm.nfe

        with contextlib.suppress(AttributeError):
            runtime_info["epsilon_progress"] = algorithm.archive.improvements

        variator = algorithm.variator
        if isinstance(variator, Multimethod):
            for method, prob in zip(variator.variators, variator.probabilities):
                if isinstance(method, GAOperator):
                    method = method.variation  # noqa: PLW2901

                runtime_info[method.__class__.__name__] = prob

        self.data.append(runtime_info)

    def to_dataframe(self):
        return pd.DataFrame(self.data)


def load_archives(path_to_file: str) -> list[tuple[int, pd.DataFrame]]:
    """Returns a list of stored archives.

    Each entry in the list is a tuple. The first element is the number of
    nfe, the second is the archive at that number of nfe.

    Parameters
    ----------
    path_to_file : the path to the archive

    """
    with tarfile.open(path_to_file, "r") as archive:
        content = archive.getnames()
        archives = []
        for fn in content:
            f = archive.extractfile(fn)
            data = pd.read_csv(f)
            nfe = int(fn.split(".")[0])
            archives.append((nfe, data))

    return archives


def epsilon_nondominated(
    results: list[pd.DataFrame], epsilons: list[float], problem: Problem
) -> pd.DataFrame:
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


def rebuild_platypus_population(archive: pd.DataFrame, problem: Problem):
    """Rebuild a population of platypus Solution instances.

    Parameters
    ----------
    archive : DataFrame
    problem : PlatypusProblem instance

    Returns
    -------
    list of platypus Solutions

    """
    # fixme, might this be easier via Sample._to_platypus_solution?
    #   we can just turn each row into a Sample instance directly and then go to a Solution instance
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

    def evolve(self, parents: list[Solution]) -> tuple[Solution, Solution]:
        """Evolve the provided parents."""
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

        return child1, child2

    def mutate(self, child: Solution):
        problem = child.problem

        for i, kind in enumerate(problem.types):  # @ReservedAssignment
            if random.random() <= self.mutation_prob:
                klass = kind.__class__
                child = self._mutate[klass](self, child, i, kind)
                child.evaluated = False

    def crossover_real(
        self, child1: Solution, child2: Solution, i: int, type: platypus.Real
    ) -> tuple[Solution, Solution]:  # @ReservedAssignment
        # sbx
        x1 = float(child1.variables[i])
        x2 = float(child2.variables[i])
        lb = type.min_value
        ub = type.max_value

        x1, x2 = self.SBX.sbx_crossover(x1, x2, lb, ub)

        child1.variables[i] = x1
        child2.variables[i] = x2

        return child1, child2

    def crossover_integer(
        self, child1: Solution, child2: Solution, i: int, type: platypus.Integer
    ) -> tuple[Solution, Solution]:  # @ReservedAssignment
        # HUX()
        for j in range(type.nbits):
            if child1.variables[i][j] != child2.variables[i][j]:  # noqa: SIM102
                if bool(random.getrandbits(1)):
                    child1.variables[i][j] = not child1.variables[i][j]
                    child2.variables[i][j] = not child2.variables[i][j]
        return child1, child2

    def crossover_categorical(
        self, child1: Solution, child2: Solution, i: int, type: platypus.Subset
    ) -> tuple[Solution, Solution]:  # @ReservedAssignment
        # SSX()
        # Implemented in a simplified manner, since size of subset is 1

        if (child2.variables[i] != child1.variables[i]) and (random.random() < 0.5):
            temp = child1.variables[i]
            child1.variables[i] = child2.variables[i]
            child2.variables[i] = temp

        return child1, child2

    def mutate_real(
        self, child: Solution, i: int, type: platypus.Real, distribution_index: int = 20
    ) -> Solution:  # @ReservedAssignment
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

    def mutate_integer(
        self, child: Solution, i: int, type: platypus.Integer, probability: float = 1
    ) -> Solution:  # @ReservedAssignment
        # bitflip
        for j in range(type.nbits):
            if random.random() <= probability:
                child.variables[i][j] = not child.variables[i][j]
        return child

    def mutate_categorical(
        self, child: Solution, i: int, type: platypus.Subset
    ) -> Solution:  # @ReservedAssignment
        # replace, again simplified because len(subset) is 1
        non_members = [
            entry for entry in type.elements if entry.value != child.variables[i]
        ]
        new_value = random.choice(non_members)
        child.variables[i] = new_value.value

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
    rng: None | SeedLike = None,
    **kwargs,
):
    """Helper function for optimization."""
    klass = problem.types[0].__class__

    random.seed(rng)

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
    runtime_convergence_info = RuntimeConvergenceTracking(frequency=convergence_freq)
    optimizer.add_extension(storage)
    optimizer.add_extension(progress_bar)
    optimizer.add_extension(runtime_convergence_info)

    with temporary_filter(name=[callbacks.__name__, evaluators.__name__], level=INFO):
        optimizer.run(nfe)

    storage.do_action(
        optimizer
    )  # ensure last archive is included in the convergence information
    runtime_convergence_info.do_action(
        optimizer
    )  # ensure the last convergence information is added as well
    progress_bar.progress_tracker.pbar.__exit__(
        None, None, None
    )  # ensure progress bar is closed correctly

    try:
        data = optimizer.archive
    except AttributeError:
        data = optimizer.result

    runtime_convergence = runtime_convergence_info.to_dataframe()

    results = to_dataframe(data, problem.parameter_names, problem.outcome_names)

    _logger.info(f"optimization completed, found {len(data)} solutions")

    return results, runtime_convergence


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

    pm_p = None
    pm_dist = 20

    sbx_prop = 1
    sbx_dist = 15

    de_rate = 0.1
    de_stepsize = 0.5

    um_p = None

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
        self.pm_p = 1 / problem.nvars
        self.um_p = 1 / problem.nvars

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

        kwargs["variator"] = Multimethod(self, variators)
        super().__init__(
            problem,
            population_size=population_size,
            generator=generator,
            selector=selector,
            archive=EpsilonBoxArchive(epsilons),
            **kwargs,
        )
        self.add_extension(platypus.extensions.EpsilonProgressContinuationExtension())
