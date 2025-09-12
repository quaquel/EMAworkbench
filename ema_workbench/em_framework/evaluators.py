"""collection of evaluators for performing experiments, optimization, and robust optimization."""
import abc
import enum
import numbers
import os
from collections.abc import Callable, Iterable

from ema_workbench.em_framework.samplers import AbstractSampler

from ..util import EMAError, get_module_logger
from .callbacks import DefaultCallback, AbstractCallback
from .experiment_runner import ExperimentRunner
from .model import AbstractModel
from .optimization import (
    EpsNSGAII,
    _optimize,
    evaluate,
    evaluate_robust,
    process_levers,
    process_robust,
    process_uncertainties,
    to_problem,
    to_robust_problem,
)
from .outcomes import AbstractOutcome, ScalarOutcome
from .points import Experiment, Policy, Scenario, experiment_generator
from .salib_samplers import FASTSampler, MorrisSampler, SobolSampler
from .samplers import (
    DesignIterator,
    FullFactorialSampler,
    LHSSampler,
    MonteCarloSampler,
    sample_levers,
    sample_uncertainties,
)
from .util import determine_objects

# Created on 5 Mar 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = [
    "Samplers",
    "SequentialEvaluator",
    "optimize",
    "perform_experiments",
]

_logger = get_module_logger(__name__)


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

    def __init__(self, msis:AbstractModel|list[AbstractModel]):
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

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
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
        self, experiments:Iterable[Experiment], callback:Callable, **kwargs
    ):
        """Used by ema_workbench."""


    def evaluate_all(self, jobs, **kwargs):
        """Makes ema_workbench evaluators compatible with platypus evaluators."""
        self.callback()

        try:
            problem = jobs[0].solution.problem
        except IndexError:
            # no jobs to evaluate
            return jobs

        searchover = problem.searchover

        if searchover == "levers":
            scenarios, policies = process_levers(jobs)
            jobs_collection = zip(policies, jobs)
        elif searchover == "uncertainties":
            scenarios, policies = process_uncertainties(jobs)
            jobs_collection = zip(scenarios, jobs)
        elif searchover == "robust":
            scenarios, policies = process_robust(jobs)
            jobs_collection = zip(policies, jobs)
        else:
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

        if searchover in ("levers", "uncertainties"):
            evaluate(jobs_collection, experiments, outcomes, problem)
        else:
            evaluate_robust(jobs_collection, experiments, outcomes, problem)

        return jobs

    def perform_experiments(
            self,
            scenarios: int | DesignIterator | Scenario = 0,
            policies: int | DesignIterator | Policy = 0,
            reporting_interval: int | None = None,
            reporting_frequency: int | None = 10,
            uncertainty_union: bool = False,
            lever_union: bool = False,
            outcome_union: bool = False,
            uncertainty_sampling: AbstractSampler = Samplers.LHS,
            uncertainty_sampling_kwargs: dict | None = None,
            lever_sampling: AbstractSampler = Samplers.LHS,
            lever_sampling_kwargs: dict | None = None,
            callback: type[AbstractCallback] | None = None,
            combine="factorial",
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
        algorithm=EpsNSGAII,
        nfe=10000,
        searchover="levers",
        reference=None,
        constraints=None,
        convergence_freq=1000,
        logging_freq=5,
        variator=None,
        **kwargs,
    ):
        """Convenience method for outcome optimization.

        A call to this method is forwarded to :func:optimize, with evaluator and models
        arguments added in.

        """
        return optimize(
            self._msis,
            algorithm=algorithm,
            nfe=int(nfe),
            searchover=searchover,
            evaluator=self,
            reference=reference,
            constraints=constraints,
            convergence_freq=convergence_freq,
            logging_freq=logging_freq,
            variator=variator,
            **kwargs,
        )

    def robust_optimize(
        self,
        robustness_functions,
        scenarios,
        algorithm=EpsNSGAII,
        nfe=10000,
        convergence_freq=1000,
        logging_freq=5,
        **kwargs,
    ):
        """Convenience method for robust optimization.

        A call to this method is forwarded to :func:robust_optimize, with evaluator and models
        arguments added in.

        """
        return robust_optimize(
            self._msis,
            robustness_functions,
            scenarios,
            self,
            algorithm=algorithm,
            nfe=nfe,
            convergence_freq=convergence_freq,
            logging_freq=logging_freq,
            **kwargs,
        )


class SequentialEvaluator(BaseEvaluator):
    """Sequential evaluator."""

    def initialize(self):
        """Initializer."""


    def finalize(self):
        """Finalizer."""


    def evaluate_experiments(self, experiments:Iterable[Experiment], callback:Callable, **kwargs):
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
    models: AbstractModel|list[AbstractModel],
    scenarios:int|DesignIterator|Scenario=0,
    policies:int|DesignIterator|Policy=0,
    evaluator:BaseEvaluator|None=None,
    reporting_interval:int|None=None,
    reporting_frequency:int|None=10,
    uncertainty_union:bool=False,
    lever_union:bool=False,
    outcome_union:bool=False,
    uncertainty_sampling:AbstractSampler=Samplers.LHS,
    uncertainty_sampling_kwargs:dict|None=None,
    lever_sampling:AbstractSampler=Samplers.LHS,
    lever_sampling_kwargs:dict|None=None,
    callback:type[AbstractCallback]|None=None,
    return_callback:bool=False,
    combine:str="factorial",
    log_progress:bool=False,
    **kwargs,
):
    """Sample uncertainties and levers, and perform the resulting experiments on each of the models.

    Parameters
    ----------
    models : one or more AbstractModel instances
    scenarios : int or collection of Scenario instances, optional
    policies :  int or collection of Policy instances, optional
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
    # unreadable in this form

    if not scenarios and not policies:
        raise EMAError(
            "no experiments possible since both scenarios and policies are 0"
        )
    if uncertainty_sampling_kwargs is None:
        uncertainty_sampling_kwargs = {}
    if lever_sampling_kwargs is None:
        lever_sampling_kwargs = {}
    uncertainty_sampling_kwargs["uncertainty_union"] = uncertainty_union
    lever_sampling_kwargs["lever_union"] = lever_union

    scenarios, uncertainties, n_scenarios = setup_scenarios(
        scenarios, uncertainty_sampling, uncertainty_sampling_kwargs, models
    )
    policies, levers, n_policies = setup_policies(
        policies, lever_sampling, lever_sampling_kwargs, models
    )

    try:
        n_models = len(models)
    except TypeError:
        n_models = 1
        models = [models,]

    outcomes = determine_objects(models, "outcomes", union=outcome_union)


    match combine:
        case "factorial":

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
        case _:
            raise ValueError(f"unknown value for combine, got {combine}, should be one of \"zipover\" or \"factorial\"")

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

    evaluator.evaluate_experiments(experiments, callback,**kwargs)

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


def setup_policies(policies:int|DesignIterator|Policy, sampler:AbstractSampler|None, lever_sampling_kwargs, models):
    # todo fix sampler type hints by adding Literal[all fields of sampler enum]

    if not policies:
        policies = [Policy("None")]
        levers = []
        n_policies = 1
    elif isinstance(policies, numbers.Integral):
        if not isinstance(sampler, AbstractSampler):
            sampler = sampler.value

        policies = sample_levers(models, policies, sampler=sampler,  **lever_sampling_kwargs)
        levers = policies.parameters
        n_policies = policies.n
    else:
        try:
            levers = policies.parameters
            n_policies = policies.n
        except AttributeError:
            levers = determine_objects(models, "levers", union=True)
            if isinstance(policies, Policy):
                policies = [policies]

            levers = [l for l in levers if l.name in policies[0]] #noqa: E741
            n_policies = len(policies)
    return policies, levers, n_policies


def setup_scenarios(scenarios:int|DesignIterator|Scenario, sampler:AbstractSampler|None, uncertainty_sampling_kwargs, models):
    # todo fix sampler type hints by adding Literal[all fields of sampler enum]

    if not scenarios:
        scenarios = [Scenario("None")]
        uncertainties = []
        n_scenarios = 1
    elif isinstance(scenarios, numbers.Integral):
        if not isinstance(sampler, AbstractSampler):
            sampler = sampler.value
        scenarios = sample_uncertainties(
            models, scenarios, sampler=sampler, **uncertainty_sampling_kwargs
        )
        uncertainties = scenarios.parameters
        n_scenarios = scenarios.n
    else:
        try:
            uncertainties = scenarios.parameters
            n_scenarios = scenarios.n
        except AttributeError:
            uncertainties = determine_objects(models, "uncertainties", union=True)
            if isinstance(scenarios, Scenario):
                scenarios = [scenarios]

            uncertainties = [u for u in uncertainties if u.name in scenarios[0]]
            n_scenarios = len(scenarios)
    return scenarios, uncertainties, n_scenarios


def optimize(
    models,
    algorithm=EpsNSGAII,
    nfe=10000,
    searchover="levers",
    evaluator=None,
    reference=None,
    convergence=None,
    constraints=None,
    convergence_freq=1000,
    logging_freq=5,
    variator=None,
    **kwargs,
):
    """Optimize the model.

    Parameters
    ----------
    models : 1 or more Model instances
    algorithm : a valid Platypus optimization algorithm
    nfe : int
    searchover : {'uncertainties', 'levers'}
    evaluator : evaluator instance
    reference : Policy or Scenario instance, optional
                overwrite the default scenario in case of searching over
                levers, or default policy in case of searching over
                uncertainties
    convergence : function or collection of functions, optional
    constraints : list, optional
    convergence_freq :  int
                        nfe between convergence check
    logging_freq : int
                   number of generations between logging of progress
    variator : platypus GAOperator instance, optional
               if None, it falls back on the defaults in platypus-opts
               which is SBX with PM
    kwargs : any additional arguments will be passed on to algorithm

    Returns
    -------
    pandas DataFrame

    Raises
    ------
    EMAError if searchover is not one of 'uncertainties' or 'levers'
    NotImplementedError if len(models) > 1

    """
    if searchover not in ("levers", "uncertainties"):
        raise EMAError(
            f"Searchover should be one of 'levers' or 'uncertainties', not {searchover}"
        )

    try:
        if len(models) == 1:
            models = models[0]
        else:
            raise NotImplementedError(
                "Optimization over multiple models is not yet supported"
            )
    except TypeError:
        pass

    problem = to_problem(
        models, searchover, constraints=constraints, reference=reference
    )

    # solve the optimization problem
    if not evaluator:
        evaluator = SequentialEvaluator(models)

    return _optimize(
        problem,
        evaluator,
        algorithm,
        convergence,
        nfe,
        convergence_freq,
        logging_freq,
        variator=variator,
        **kwargs,
    )


def robust_optimize(
    model,
    robustness_functions,
    scenarios,
    evaluator=None,
    algorithm=EpsNSGAII,
    nfe=10000,
    convergence=None,
    constraints=None,
    convergence_freq=1000,
    logging_freq=5,
    **kwargs,
):
    """Perform robust optimization.

    Parameters
    ----------
    model : model instance
    robustness_functions : collection of ScalarOutcomes
    scenarios : int, or collection
    evaluator : Evaluator instance
    algorithm : platypus Algorithm instance
    nfe : int
    convergence : list
                  list of convergence metrics
    constraints : list
    convergence_freq :  int
                        nfe between convergence check
    logging_freq : int
                   number of generations between logging of progress
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
        assert rf.kind != AbstractOutcome.INFO
        assert rf.function is not None

    problem = to_robust_problem(
        model,
        scenarios,
        constraints=constraints,
        robustness_functions=robustness_functions,
    )

    # solve the optimization problem
    if not evaluator:
        evaluator = SequentialEvaluator(model)

    return _optimize(
        problem,
        evaluator,
        algorithm,
        convergence,
        int(nfe),
        convergence_freq,
        logging_freq,
        **kwargs,
    )
