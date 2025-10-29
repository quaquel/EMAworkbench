"""collection of evaluators for performing experiments, optimization, and robust optimization."""

import abc
import enum
import numbers
import os
from collections.abc import Callable, Iterable
from typing import Literal, overload

import numpy as np
import pandas as pd
import platypus
from platypus import AbstractGeneticAlgorithm, EpsNSGAII, Variator

from ema_workbench.em_framework.samplers import AbstractSampler

from ..util import EMAError, get_module_logger
from .callbacks import AbstractCallback, DefaultCallback
from .experiment_runner import ExperimentRunner
from .model import AbstractModel
from .optimization import (
    Problem,
    _optimize,
    evaluate,
    process_jobs,
)
from .outcomes import Constraint, Outcome, ScalarOutcome
from .parameters import Parameter
from .points import Experiment, Sample, SampleCollection, experiment_generator
from .salib_samplers import FASTSampler, MorrisSampler, SobolSampler
from .samplers import (
    FullFactorialSampler,
    LHSSampler,
    MonteCarloSampler,
)
from .util import determine_objects

# Created on 5 Mar 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = [
    "BaseEvaluator",
    "Samplers",
    "SequentialEvaluator",
    "optimize",
    "perform_experiments",
]

_logger = get_module_logger(__name__)

SeedLike = int | float | str | bytes | bytearray  # seedlike for stdlib random.seed


class Samplers(enum.Enum):
    """Enum for different kinds of samplers."""

    ## TODO:: have samplers register themselves on class instantiation
    ## TODO:: should not be defined here

    MC = MonteCarloSampler()
    LHS = LHSSampler()
    FF = FullFactorialSampler()
    SOBOL = SobolSampler()
    FAST = FASTSampler()
    MORRIS = MorrisSampler()


SamplerTypes = Literal[
    Samplers.MC,
    Samplers.LHS,
    Samplers.FF,
    Samplers.SOBOL,
    Samplers.FAST,
    Samplers.MORRIS,
]


class BaseEvaluator(abc.ABC):
    """evaluator for experiments using a multiprocessing pool.

    Parameters
    ----------
    msis : collection of models

    Raises
    ------
    ValueError

    """

    reporting_frequency = 3

    def __init__(self, msis: AbstractModel | list[AbstractModel]):
        super().__init__()

        if isinstance(msis, AbstractModel):
            msis = [msis]
        else:
            for entry in msis:
                if not isinstance(entry, AbstractModel):
                    raise TypeError(
                        f"{entry} should be an AbstractModel instance, but is a {entry.__class__} instance"
                    )

        self._msis = msis
        self.callback = None

    def __enter__(self):  # noqa: D105
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: D105
        self.finalize()

        if exc_type is not None:
            return False

    @abc.abstractmethod
    def initialize(self):
        """Initialize the evaluator."""

    @abc.abstractmethod
    def finalize(self):
        """Finalize the evaluator."""

    @abc.abstractmethod
    def evaluate_experiments(
        self, experiments: Iterable[Experiment], callback: Callable, **kwargs
    ):
        """Used by ema_workbench."""

    def evaluate_all(self, jobs: list[platypus.core.EvaluateSolution], **kwargs):
        """Makes ema_workbench evaluators compatible with platypus evaluators."""
        try:
            problem = jobs[0].solution.problem
        except IndexError:
            # no jobs to evaluate
            return jobs

        searchover = problem.searchover

        scenarios, policies = process_jobs(jobs)

        match searchover:
            case "levers" | "robust":
                jobs_collection = zip(policies, jobs)
            case "uncertainties":
                jobs_collection = zip(scenarios, jobs)
            case _:
                raise NotImplementedError()

        # overwrite the default 10 progress reports  with 5 reports
        callback = perform_experiments(
            self._msis,
            evaluator=self,
            reporting_frequency=self.reporting_frequency,
            scenarios=scenarios,
            policies=policies,
            return_callback=True,
            log_progress=True,
        )

        experiments, outcomes = callback.get_results()

        evaluate(jobs_collection, experiments, outcomes, problem)

        return jobs

    def perform_experiments(
        self,
        scenarios: int | Iterable[Sample] | Sample = 0,
        policies: int | Iterable[Sample] | Sample = 0,
        reporting_interval: int | None = None,
        reporting_frequency: int | None = 10,
        uncertainty_union: bool = False,
        lever_union: bool = False,
        outcome_union: bool = False,
        uncertainty_sampling: AbstractSampler | SamplerTypes = Samplers.LHS,
        uncertainty_sampling_kwargs: dict | None = None,
        lever_sampling: AbstractSampler | SamplerTypes = Samplers.LHS,
        lever_sampling_kwargs: dict | None = None,
        callback: type[AbstractCallback] | None = None,
        combine: Literal["full_factorial", "sample", "cycle"] = "full_factorial",
        **kwargs,
    ):
        """Convenience method for performing experiments.

        A call to this method is forwarded to :func:perform_experiments, with evaluator and
        models arguments added in.

        """
        return perform_experiments(
            self._msis.copy(),
            scenarios=scenarios,
            policies=policies,
            evaluator=self,
            reporting_interval=reporting_interval,
            reporting_frequency=reporting_frequency,
            uncertainty_union=uncertainty_union,
            lever_union=lever_union,
            outcome_union=outcome_union,
            uncertainty_sampling=uncertainty_sampling,
            uncertainty_sampling_kwargs=uncertainty_sampling_kwargs,
            lever_sampling=lever_sampling,
            lever_sampling_kwargs=lever_sampling_kwargs,
            callback=callback,
            combine=combine,
            **kwargs,
        )

    def optimize(
        self,
        algorithm: type[AbstractGeneticAlgorithm] = EpsNSGAII,
        nfe: int = 10000,
        searchover: str = "levers",
        reference: Sample | None = None,
        constraints: Iterable[Constraint] | None = None,
        convergence_freq: int = 1000,
        logging_freq: int = 5,
        variator: type[Variator] | None = None,
        rng: SeedLike | Iterable[SeedLike] | None = None,
        initial_population: Iterable[Sample] | None = None,
        filename: str | None = None,
        directory: str | None = None,
        **kwargs,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convenience method for outcome optimization.

        A call to this method is forwarded to :func:optimize, with evaluator and models
        arguments added in.

        """
        if len(self._msis) > 1:
            raise NotImplementedError(
                "Optimization over multiple models is not yet supported"
            )
        model = self._msis[0]

        return optimize(
            model,
            algorithm=algorithm,
            nfe=int(nfe),
            searchover=searchover,
            evaluator=self,
            reference=reference,
            constraints=constraints,
            convergence_freq=convergence_freq,
            logging_freq=logging_freq,
            variator=variator,
            rng=rng,
            filename=filename,
            directory=directory,
            initial_population=initial_population,
            **kwargs,
        )

    def robust_optimize(
        self,
        robustness_functions: list[ScalarOutcome],
        scenarios: int | Iterable[Sample],
        algorithm: type[AbstractGeneticAlgorithm] = EpsNSGAII,
        nfe: int = 10000,
        convergence_freq: int = 1000,
        logging_freq: int = 5,
        rng: SeedLike| Iterable[SeedLike] | None = None,
        **kwargs,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convenience method for robust optimization.

        A call to this method is forwarded to :func:robust_optimize, with evaluator and models
        arguments added in.

        """
        if len(self._msis) > 1:
            raise NotImplementedError(
                "Optimization over multiple models is not yet supported"
            )
        model = self._msis[0]

        return robust_optimize(
            model,
            robustness_functions,
            scenarios,
            evaluator=self,
            algorithm=algorithm,
            nfe=nfe,
            convergence_freq=convergence_freq,
            logging_freq=logging_freq,
            rng=rng,
            **kwargs,
        )


class SequentialEvaluator(BaseEvaluator):
    """Sequential evaluator."""

    def initialize(self):
        """Initializer."""

    def finalize(self):
        """Finalizer."""

    def evaluate_experiments(
        self, experiments: Iterable[Experiment], callback: Callable, **kwargs
    ):
        """Evaluate experiments."""
        _logger.info("performing experiments sequentially")

        # TODO:: replace with context manager
        cwd = os.getcwd()
        runner = ExperimentRunner(self._msis)

        for experiment in experiments:
            outcomes = runner.run_experiment(experiment)
            callback(experiment, outcomes)
        runner.cleanup()
        os.chdir(cwd)


def perform_experiments(
    models: AbstractModel | list[AbstractModel],
    scenarios: int | Iterable[Sample] | Sample = 0,
    policies: int | Iterable[Sample] | Sample = 0,
    evaluator: BaseEvaluator | None = None,
    reporting_interval: int | None = None,
    reporting_frequency: int | None = 10,
    uncertainty_union: bool = False,
    lever_union: bool = False,
    outcome_union: bool = False,
    uncertainty_sampling: AbstractSampler | SamplerTypes = Samplers.LHS,
    uncertainty_sampling_kwargs: dict | None = None,
    lever_sampling: AbstractSampler | SamplerTypes = Samplers.LHS,
    lever_sampling_kwargs: dict | None = None,
    callback: type[AbstractCallback] | None = None,
    return_callback: bool = False,
    combine: Literal["full_factorial", "sample", "cycle"] = "full_factorial",
    log_progress: bool = False,
    **kwargs,
) -> DefaultCallback | tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Sample uncertainties and levers, and perform the resulting experiments on each of the models.

    Parameters
    ----------
    models : one or more AbstractModel instances
    scenarios : int or iterable of Sample instances, optional
    policies :  int or iterable of Sample instances, optional
    evaluator : Evaluator instance, optional
    reporting_interval : int, optional
    reporting_frequency: int, optional
    uncertainty_union : boolean, optional
    lever_union : boolean, optional
    outcome_union : boolean, optional
    uncertainty_sampling : {LHS, MC, FF, SOBOL, MORRIS, FAST}, optional
    uncertainty_sampling_kwargs : dict, optional
    lever_sampling : {LHS, MC, FF, SOBOL, MORRIS, FAST}, optional TODO:: update doc
    lever_sampling_kwargs : dict, optional
    callback  : Callback instance, optional
    return_callback : boolean, optional
    log_progress : bool, optional
    combine : {'factorial', 'sample'}, optional
              how to combine uncertainties and levers?
              In case of 'factorial', both are sampled separately using their
              respective samplers. Next the resulting designs are combined in a
              full factorial manner.
              In case of 'sample', both are sampled separately and
              then combined by cycling over the shortest of the the two sets
              of designs until the longest set of designs is exhausted.

    Additional keyword arguments are passed on to evaluate_experiments of the evaluator

    Returns
    -------
    tuple
        the experiments as a dataframe, and a dict
        with the name of an outcome as key, and the associated values
        as numpy array. Experiments and outcomes are aligned on index.


    """
    # TODO:: break up in to helper functions
    #        unreadable in this form

    if not scenarios and not policies:
        raise EMAError(
            "no experiments possible since both scenarios and policies are 0"
        )

    scenarios, uncertainties, n_scenarios = _setup(
        scenarios,
        uncertainty_sampling,
        uncertainty_sampling_kwargs,
        models,
        parameter_type="uncertainties",
        union=uncertainty_union,
    )
    policies, levers, n_policies = _setup(
        policies,
        lever_sampling,
        lever_sampling_kwargs,
        models,
        union=lever_union,
        parameter_type="levers",
    )

    try:
        n_models = len(models)
    except TypeError:
        n_models = 1
        models = [
            models,
        ]

    outcomes = determine_objects(models, "outcomes", union=outcome_union)

    nr_of_exp = -1
    match combine:
        case "full_factorial":
            nr_of_exp = n_models * n_scenarios * n_policies

            # TODO:: change to 0 policies / 0 scenarios is sampling set to 0 for
            # it
            _logger.info(
                f"performing {n_scenarios} scenarios * {n_policies} policies * {n_models} model(s) = "
                f"{nr_of_exp} experiments"
            )
        case "sample":
            nr_of_exp = n_models * max(n_scenarios, n_policies)
            # TODO:: change to 0 policies / 0 scenarios is sampling set to 0 for
            # it
            _logger.info(
                f"performing max({n_scenarios} scenarios, {n_policies} policies) * {n_models} model(s) = "
                f"{nr_of_exp} experiments"
            )
        case "cycle":
            nr_of_exp = n_models * max(n_scenarios, n_policies)
            # TODO:: change to 0 policies / 0 scenarios is sampling set to 0 for
            # it
            _logger.info(
                f"performing max({n_scenarios} scenarios, {n_policies} policies) * {n_models} model(s) = "
                f"{nr_of_exp} experiments"
            )
        case _:
            raise ValueError(
                f'unknown value for combine, got {combine}, should be one of "sample" or "factorial"'
            )

    callback = setup_callback(
        callback,
        uncertainties,
        levers,
        outcomes,
        nr_of_exp,
        reporting_interval,
        reporting_frequency,
        log_progress,
    )

    if not evaluator:
        evaluator = SequentialEvaluator(models)

    experiments = experiment_generator(models, scenarios, policies, combine=combine)

    evaluator.evaluate_experiments(experiments, callback, **kwargs)

    if callback.i != nr_of_exp:
        raise EMAError(
            f"Some fatal error has occurred while running the experiments, not all runs have completed. Expected {nr_of_exp}, got {callback.i}"
        )

    _logger.info("experiments finished")

    if return_callback:
        return callback

    results = callback.get_results()
    return results


def setup_callback(
    callback,
    uncertainties,
    levers,
    outcomes,
    nr_of_exp,
    reporting_interval,
    reporting_frequency,
    log_progress,
):
    if not callback:
        callback = DefaultCallback(
            uncertainties,
            levers,
            outcomes,
            nr_of_exp,
            reporting_interval=reporting_interval,
            reporting_frequency=reporting_frequency,
            log_progress=log_progress,
        )
    else:
        callback = callback(
            uncertainties,
            levers,
            outcomes,
            nr_of_exp,
            reporting_interval=reporting_interval,
            reporting_frequency=reporting_frequency,
            log_progress=log_progress,
        )
    return callback


def _setup(
    samples: int | Iterable[Sample] | Sample | SampleCollection | None,
    sampler: AbstractSampler | SamplerTypes | None,
    sampler_kwargs: dict | None,
    models: Iterable[AbstractModel],
    union: bool = True,
    parameter_type: Literal["uncertainties", "levers"] = "uncertainties",
) -> tuple[Iterable[Sample], list[Parameter], int]:
    """Helper function.

    Parameters
    ----------
    samples : int | Iterable[Sample] | Sample | SampleCollection | None
    sampler: AbstractSampler | SamplerTypes| None
             sampler to use, only relevant if samples is an int
    sampler_kwargs: dict
                    kwargs for sampler, only relevant if sampler is not None
    models : Iterable[AbstractModel]
             models to consider
    union: bool
            only relevant if len(model)s > 1, how to handle the parameters across multiple models
            if true, use union, if false use intersection.
    parameter_type: Literal["uncertainties", "levers"]
            which parameters to consider on models.

    """
    # todo fix sampler type hints by adding Literal[all fields of sampler enum].

    if sampler_kwargs is None:
        sampler_kwargs = {}

    if not samples:
        samples = [Sample("None")]
        parameters = []
        n_samples = 1
    elif isinstance(samples, numbers.Integral):
        if not isinstance(sampler, AbstractSampler):
            sampler = sampler.value
        parameters = determine_objects(models, parameter_type, union=union)
        samples = sampler.generate_samples(parameters, samples, **sampler_kwargs)
        parameters = samples.parameters
        n_samples = len(samples)
    else:
        try:
            parameters = samples.parameters
            n_samples = len(samples)
        except AttributeError:
            parameters = determine_objects(models, parameter_type, union=True)
            if isinstance(samples, Sample):
                samples = [samples]

            parameters = [p for p in parameters if p.name in samples[0]]
            n_samples = len(samples)
    return samples, parameters, n_samples


@overload
def optimize(
    model: AbstractModel,
    algorithm: type[AbstractGeneticAlgorithm] = EpsNSGAII,
    nfe: int = 10000,
    searchover: str = "levers",
    evaluator: BaseEvaluator | None = None,
    reference: Sample | None = None,
    constraints: Iterable[Constraint] | None = None,
    convergence_freq: int = 1000,
    logging_freq: int = 5,
    variator: Variator = None,
    rng: Iterable[SeedLike] | None = None,
    initial_population: Iterable[Sample] | None = None,
    filename: str | None = None,
    directory: str | None = None,
    **kwargs,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]: ...


@overload
def optimize(
    model: AbstractModel,
    algorithm: type[AbstractGeneticAlgorithm] = EpsNSGAII,
    nfe: int = 10000,
    searchover: str = "levers",
    evaluator: BaseEvaluator | None = None,
    reference: Sample | None = None,
    constraints: Iterable[Constraint] | None = None,
    convergence_freq: int = 1000,
    logging_freq: int = 5,
    variator: Variator = None,
    rng: SeedLike | None = None,
    initial_population: Iterable[Sample] | None = None,
    filename: str | None = None,
    directory: str | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]: ...


def optimize(
    model: AbstractModel,
    algorithm: type[AbstractGeneticAlgorithm] = EpsNSGAII,
    nfe: int = 10000,
    searchover: str = "levers",
    evaluator: BaseEvaluator | None = None,
    reference: Sample | None = None,
    constraints: Iterable[Constraint] | None = None,
    convergence_freq: int = 1000,
    logging_freq: int = 5,
    variator: Variator = None,
    rng: SeedLike | Iterable[SeedLike] | None = None,
    initial_population: Iterable[Sample] | None = None,
    filename: str | None = None,
    directory: str | None = None,
    **kwargs,
):
    """Optimize the model.

    Parameters
    ----------
    model : 1 or more Model instances
    algorithm : a valid Platypus optimization algorithm
    nfe : int
    searchover : {'uncertainties', 'levers'}
    evaluator : evaluator instance
    reference : Sample instance, optional
                overwrite the default scenario in case of searching over
                levers, or default policy in case of searching over
                uncertainties
    constraints : list, optional
    convergence_freq :  int
                        nfe between convergence check
    logging_freq : int
                   number of generations between logging of progress
    variator : platypus GAOperator instance, optional
               if None, it falls back on the defaults in platypus-opts
               which is SBX with PM
    rng : seed for initializing the global python random number generator as used by platypus-opt
          because platypus-opt uses the global random number generator, full reproducibility cannot
          be guaranteed in case of threading.
          If seed is an iterable, the optimization is run for each seed. The seed is added to the filename
          to ensure separate tarballs for each run of the optimization.
    initial_population : A collection of samples with which to initialize the optimization algorithm
    filename : filename for storing intermediate archives/results
               Every convergence_freq, the archive or results at that moment are added to this file.
               The file itself will be a tarball.
    directory : directory in which to store tarball of intermediate archives/results.
    kwargs : any additional arguments will be passed on to algorithm

    Returns
    -------
    pandas DataFrame with pareto front
    pandas DataFrame with runtime convergence information

    Raises
    ------
    ValueError if searchover is not one of 'uncertainties' or 'levers'

    """
    if searchover not in ("levers", "uncertainties"):
        raise ValueError(
            f"Searchover should be one of 'levers' or 'uncertainties', not {searchover}"
        )

    decision_variables = model.levers if searchover == "levers" else model.uncertainties
    problem = Problem(
        searchover, decision_variables, model.outcomes, constraints, reference=reference
    )

    # solve the optimization problem
    if not evaluator:
        evaluator = SequentialEvaluator(model)

    if isinstance(rng, Iterable):
        all_results = []
        for entry in rng:
            new_filename = f"{entry}_{filename}"

            result = _optimize(
                problem,
                evaluator,
                algorithm,
                nfe,
                convergence_freq,
                logging_freq,
                variator=variator,
                initial_population=initial_population,
                filename=new_filename,
                directory=directory,
                rng=entry,
                **kwargs,
            )
            all_results.append(result)
        return all_results
    else:
        return _optimize(
            problem,
            evaluator,
            algorithm,
            nfe,
            convergence_freq,
            logging_freq,
            variator=variator,
            filename=filename,
            directory=directory,
            initial_population=initial_population,
            rng=rng,
            **kwargs,
        )

@overload
def robust_optimize(
    model: AbstractModel,
    robustness_functions: list[ScalarOutcome],
    scenarios: int | Iterable[Sample],
    algorithm: type[AbstractGeneticAlgorithm] = EpsNSGAII,
    nfe: int = 10000,
    evaluator: BaseEvaluator | None = None,
    constraints: Iterable[Constraint] | None = None,
    convergence_freq: int = 1000,
    logging_freq: int = 5,
    variator: Variator = None,
    rng: SeedLike | None = None,
    initial_population: Iterable[Sample] | None = None,
    filename: str | None = None,
    directory: str | None = None,
    **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]: ...

@overload
def robust_optimize(
    model: AbstractModel,
    robustness_functions: list[ScalarOutcome],
    scenarios: int | Iterable[Sample],
    algorithm: type[AbstractGeneticAlgorithm] = EpsNSGAII,
    nfe: int = 10000,
    evaluator: BaseEvaluator | None = None,
    constraints: Iterable[Constraint] | None = None,
    convergence_freq: int = 1000,
    logging_freq: int = 5,
    variator: Variator = None,
    rng: Iterable[SeedLike] | None = None,
    initial_population: Iterable[Sample] | None = None,
    filename: str | None = None,
    directory: str | None = None,
    **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]: ...

def robust_optimize(
    model: AbstractModel,
    robustness_functions: list[ScalarOutcome],
    scenarios: int | Iterable[Sample],
    algorithm: type[AbstractGeneticAlgorithm] = EpsNSGAII,
    nfe: int = 10000,
    evaluator: BaseEvaluator | None = None,
    constraints: Iterable[Constraint] | None = None,
    convergence_freq: int = 1000,
    logging_freq: int = 5,
    variator: Variator = None,
    rng: SeedLike | Iterable[SeedLike] | None = None,
    initial_population: Iterable[Sample] | None = None,
    filename: str | None = None,
    directory: str | None = None,
    **kwargs
):
    """Perform robust optimization.

    Parameters
    ----------
    model : model instance
    robustness_functions : collection of ScalarOutcomes
    scenarios : int, or collection
    kwargs : any additional arguments will be passed on to algorithm

    Raises
    ------
    AssertionError if robustness_function is not a ScalarOutcome,
    if robustness_function.kind is INFO, or
    if robustness_function.function is None

    robustness functions are scalar outcomes, kind should be MINIMIZE or
    MAXIMIZE, function is the robustness function you want to use.

    """
    for rf in robustness_functions:
        assert isinstance(rf, ScalarOutcome)
        assert rf.kind != Outcome.INFO
        assert rf.function is not None

    problem = Problem(
        "robust", model.levers, robustness_functions, constraints, reference=scenarios
    )

    if not evaluator:
        evaluator = SequentialEvaluator(model)

    # solve the optimization problem
    if isinstance(rng, Iterable):
        all_results = []
        for entry in rng:
            new_filename = f"{entry}_{filename}"

            result = _optimize(
                problem,
                evaluator,
                algorithm,
                nfe,
                convergence_freq,
                logging_freq,
                variator=variator,
                initial_population=initial_population,
                filename=new_filename,
                directory=directory,
                rng=entry,
                **kwargs,
            )
            all_results.append(result)
        return all_results
    else:
        return _optimize(
            problem,
            evaluator,
            algorithm,
            nfe,
            convergence_freq,
            logging_freq,
            variator=variator,
            filename=filename,
            directory=directory,
            initial_population=initial_population,
            rng=rng,
            **kwargs,
        )
