"""A variety of samplers.

These different samplers implement basic sampling
techniques including Full Factorial sampling, Latin Hypercube sampling, and
Monte Carlo sampling.

"""

import abc
import itertools
import math
import numbers
from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import scipy.stats as stats
import scipy.stats.qmc as qmc

from ema_workbench.em_framework import util
from ema_workbench.em_framework.parameters import (
    BooleanParameter,
    CategoricalParameter,
    IntegerParameter,
    Parameter,
)
from ema_workbench.em_framework.points import Point, Policy, Scenario
from ema_workbench.util.ema_exceptions import EMAError

# Created on 16 aug. 2011
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = [
    "AbstractSampler",
    "DesignIterator",
    "FullFactorialSampler",
    "LHSSampler",
    "MonteCarloSampler",
    "determine_parameters",
    "sample_levers",
    "sample_parameters",
    "sample_uncertainties",
]


SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence
RNGLike = np.random.Generator | np.random.BitGenerator


class AbstractSampler(metaclass=abc.ABCMeta):
    """Abstract base class from which different samplers can be derived.

    In the simplest cases, only the sample method needs to be overwritten.
    generate_designs` is the only method called from outside. The
    other methods are used internally to generate the designs.

    """

    @abc.abstractmethod
    def generate_samples(
        self,
        parameters: list[Parameter],
        size: int,
        rng: SeedLike | RNGLike | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate n samples from the parameters.

        Parameters
        ----------
        parameters : collection
                     a collection of :class:`~parameters.RealParameter`,
                     :class:`~parameters.IntegerParameter`,
                     and :class:`~parameters.CategoricalParameter`
                     instances.
        size : int
               the number of samples to generate.
        rng: numpy random number generator
        kwargs : any additional keyword arguments

        Returns
        -------
        numpy array with samples

        """

    def _rescale(self, parameters: list[Parameter], samples) -> np.ndarray:
        """Rescale uniform samples using dist and process integers."""
        for j, p in enumerate(parameters):
            samples_j = samples[:, j]

            samples[:, j] = p.dist.ppf(samples_j)

            if isinstance(p, IntegerParameter):
                samples_j = np.floor(samples_j)
            samples[:, j] = samples_j
        return samples


class LHSSampler(AbstractSampler):
    """generates a Latin Hypercube sample over the parameters."""

    def generate_samples(
        self,
        parameters: list[Parameter],
        size: int,
        rng: SeedLike | RNGLike | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate samples using latin hypercube sampling.

        Parameters
        ----------
        parameters : collection
                     a collection of :class:`~parameters.RealParameter`,
                     :class:`~parameters.IntegerParameter`,
                     and :class:`~parameters.CategoricalParameter`
                     instances.
        size : int
               the number of samples to generate.
        rng: numpy random number generator
        kwargs : any additional keyword arguments

        Additional valid keyword arguments are

        scramble : bool, optional
        optimization : {None, "random-cd", "lloyd"}, optional
        strength : {1, 2}, optional


        Returns
        -------
        numpy array with samples

        """
        lhs = qmc.LatinHypercube(d=len(parameters), rng=rng, **kwargs)
        samples = lhs.random(size)
        samples = self._rescale(parameters, samples)
        return samples


class MonteCarloSampler(AbstractSampler):
    """Monte Carlo sampler for each of the parameters."""

    def generate_samples(
        self,
        parameters: list[Parameter],
        size: int,
        rng: SeedLike | RNGLike | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate samples using Monte Carlo sampling.

        Parameters
        ----------
        parameters : collection
                     a collection of :class:`~parameters.RealParameter`,
                     :class:`~parameters.IntegerParameter`,
                     and :class:`~parameters.CategoricalParameter`
                     instances.
        size : int
               the number of samples to generate.
        rng: numpy random number generator
        kwargs : any additional keyword arguments

        There are no additional valid keyword arguments for the Monte Carlo sampler.

        """
        samples = stats.uniform.rvs(size=(size, len(parameters)), random_state=rng)
        samples = self._rescale(parameters, samples)
        return samples


class FullFactorialSampler(AbstractSampler):
    """Generates a full factorial sample.

    If the parameter is non-categorical, the resolution is set the
    number of samples. If the parameter is categorical, the specified value
    for samples will be ignored and each category will be used instead.

    """

    def generate_samples(
        self,
        parameters: list[Parameter],
        size: int,
        rng: SeedLike | RNGLike | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate samples using full factorial sampling.

        Parameters
        ----------
        parameters : collection
                     a collection of :class:`~parameters.RealParameter`,
                     :class:`~parameters.IntegerParameter`,
                     and :class:`~parameters.CategoricalParameter`
                     instances.
        size : int
               the number of samples to generate.
        rng: numpy random number generator
        kwargs : any additional keyword arguments

        There are no additional valid keyword arguments for the Monte Carlo sampler.

        """
        samples = []
        for param in parameters:
            cats = param.resolution
            if not cats:
                cats = np.linspace(param.lower_bound, param.upper_bound, size)
                if isinstance(param, IntegerParameter):
                    cats = np.round(cats, 0)
                    cats = set(cats)
                    cats = (int(entry) for entry in cats)
                    cats = sorted(cats)
            samples.append(cats)

        samples = np.asarray(list(itertools.product(*samples)))

        return samples


def determine_parameters(models, attribute, union=True):
    """Determine the parameters over which to sample.

    Parameters
    ----------
    models : a collection of AbstractModel instances
    attribute : {'uncertainties', 'levers'}
    union : bool, optional
            in case of multiple models, sample over the union of
            levers, or over the intersection of the levers

    Returns
    -------
    collection of Parameter instances

    """
    return util.determine_objects(models, attribute, union=union)


def sample_parameters(
    parameters: list[Parameter],
    n_samples: int,
    sampler: AbstractSampler | None = None,
    kind=Point,
    **kwargs,
):
    """Generate cases by sampling over the parameters.

    Parameters
    ----------
    parameters : collection of AbstractParameter instances
    n_samples : int
    sampler : Sampler instance, optional
    kind : {Case, Scenario, Policy}, optional
            the class into which the samples are collected
    kwargs : any additional keyword arguments

    Returns
    -------
    generator yielding Case, Scenario, or Policy instances

    """
    if sampler is None:
        sampler = LHSSampler()
    samples = sampler.generate_samples(parameters, n_samples, **kwargs)

    return DesignIterator(samples, parameters, kind)


def sample_levers(
    models, n_samples: int, sampler: AbstractSampler | None = None, **kwargs
):
    """Generate policies by sampling over the levers.

    Parameters
    ----------
    models : a collection of AbstractModel instances
    n_samples : int
    union : bool, optional
            in case of multiple models, sample over the union of
            levers, or over the intersection of the levers
    sampler : Sampler instance, optional

    Returns
    -------
    generator yielding Policy instances

    """
    union = kwargs.pop("lever_union", True)

    if sampler is None:
        sampler = LHSSampler()
    levers = determine_parameters(models, "levers", union=union)

    if not levers:
        raise EMAError(
            "You are trying to sample policies, but no levers have been defined"
        )

    return sample_parameters(levers, n_samples, sampler, Policy, **kwargs)


def sample_uncertainties(
    models,
    n_samples: numbers.Integral,
    sampler: AbstractSampler | None = None,
    **kwargs,
):
    """Generate scenarios by sampling over the uncertainties.

    Parameters
    ----------
    models : a collection of AbstractModel instances
    n_samples : int
    sampler : Sampler instance, optional
    kwargs : any additional keyword arguments

    Returns
    -------
    generator yielding Scenario instances

    """
    union = kwargs.pop("uncertainty_union", True)

    if sampler is None:
        sampler = LHSSampler()
    uncertainties = determine_parameters(models, "uncertainties", union=union)

    if not uncertainties:
        raise EMAError(
            "You are trying to sample scenarios, but no uncertainties have been defined"
        )

    return sample_parameters(uncertainties, n_samples, sampler, Scenario, **kwargs)


def from_experiments(models, experiments):
    """Generate scenarios from an existing experiments DataFrame.

    Parameters
    ----------
    models : collection of AbstractModel instances
    experiments : DataFrame

    Returns
    -------
     generator
        yielding Scenario instances

    """
    # fixme

    policy_names = np.unique(experiments["policy"])
    model_names = np.unique(experiments["model"])

    # we sample ff over models and policies so we need to ensure
    # we only get the experiments for a single model policy combination
    logical = (experiments["model"] == model_names[0]) & (
        experiments["policy"] == policy_names[0]
    )

    experiments = experiments[logical]

    uncertainties = util.determine_objects(models, "uncertainties", union=True)
    samples = {unc.name: experiments[:, unc.name] for unc in uncertainties}

    scenarios = DesignIterator(samples, uncertainties, experiments.shape[0])
    scenarios.kind = Scenario

    return scenarios


class DesignIterator:
    """iterable for the experimental designs."""

    # the construction with a class and the generator ensures we can repeatedly iterate over the samples.

    def __init__(
        self, samples: np.ndarray, parameters: list[Parameter], kind: type[Point]
    ):
        self.samples = samples
        self.parameters = parameters
        self.kind = kind
        self.n = self.samples.shape[0]

    def __iter__(self) -> Iterable[Point]:
        """Return an iterator yielding Points instances."""
        return design_generator(self.samples, self.parameters, self.kind)

    def __str__(self):  # noqa: D105
        return f"ema_workbench.DesignIterator, {self.n} designs on {len(self.params)} parameters"

    def combine(
        self,
        other: "DesignIterator",
        combine: Literal["full_factorial", "sample", "cycle"],
        kind: type[Point]|None = None,
        rng: SeedLike | RNGLike | None = None,
    ) -> "DesignIterator":
        """Combine 2 design iterators into a new design iterator..

        Parameters
        ----------
        other : the iterator to combine with this one
        combine : how to combine the designs.
        kind : type[Point]|None
               the Point sublcass to use for the return iterator
               in case of None, it will use the same kind if both iterators have the same otherwise
               it will just fall back on using Point.
        rng : RNG or None, only relevant in case combine is "sample"

        Returns
        -------
        a new DesignIterator instance

        """
        combined_samples = None
        samples_1 = self.samples
        samples_2 = other.samples

        match combine:
            case "full_factorial":
                samples_1_repeated = np.repeat(
                    samples_1, repeats=samples_2.shape[0], axis=0
                )
                samples_2_tiled = np.tile(samples_2, (samples_1.shape[0], 1))
                combined_samples = np.hstack((samples_1_repeated, samples_2_tiled))
            case "sample" | "cycle":
                if samples_1.shape[0] == samples_2.shape[0]:
                    combined_samples = np.hstack((samples_1, samples_2))
                else:
                    longest, shortest = (
                        (samples_1, samples_2)
                        if samples_1.shape[0] > samples_2.shape[0]
                        else (samples_2, samples_1)
                    )
                    if combine == "sample":
                        rng = np.random.default_rng(rng)
                        indices = rng.integers(0, shortest.shape[0], longest.shape[0])
                        upsampled = shortest[indices]
                    else:
                        n = int(math.ceil(longest.shape[0] / shortest.shape[0]))
                        upsampled = np.tile(shortest, (n, 1))[0 : longest.shape[0], :]
                    combined_samples = np.hstack((longest, upsampled))
            case _:
                raise ValueError(
                    f"unknown value for combine, got {combine}, should be one of full_factorial, sample"
                )

        combined_parameters = self.parameters + other.parameters

        if kind is None:
            kind = self.kind if self.kind == other.kind else Point

        return DesignIterator(combined_samples, combined_parameters, kind)


def design_generator(samples: np.ndarray, params: list[Parameter], kind: type[Point]):
    """Return a generator yielding points instances.

    This generator iterates over the samples, and turns each row into a Point and ensures datatypes are correctly handled.

    Parameters
    ----------
    samples : The samples taken for the parameters
    params : the Parameter instances that have been sampled
    kind : the (sub)class of Point to use

    Yields
    ------
    Point


    """
    for sample in samples:
        design_dict = {}
        for param, value in zip(params, sample):
            if isinstance(param, IntegerParameter):
                value = int(value)  # noqa: PLW2901
            if isinstance(param, BooleanParameter):
                value = bool(value)  # noqa: PLW2901
            if isinstance(param, CategoricalParameter):
                # categorical parameter is an integer parameter, so
                # conversion to int is already done
                value = param.cat_for_index(value).value  # noqa: PLW2901

            design_dict[param.name] = value

        yield kind(**design_dict)
