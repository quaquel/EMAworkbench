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
    IntegerParameter,
    Parameter,
)
from ema_workbench.em_framework.points import Sample, SampleCollection


# Created on 16 aug. 2011
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = [
    "AbstractSampler",
    "FullFactorialSampler",
    "LHSSampler",
    "MonteCarloSampler",
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
    ) -> "SampleCollection":
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
    ) -> "SampleCollection":
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
        return SampleCollection(samples, parameters)


class MonteCarloSampler(AbstractSampler):
    """Monte Carlo sampler for each of the parameters."""

    def generate_samples(
        self,
        parameters: list[Parameter],
        size: int,
        rng: SeedLike | RNGLike | None = None,
        **kwargs,
    ) -> "SampleCollection":
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
        return SampleCollection(samples, parameters)


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
    ) -> "SampleCollection":
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

        return SampleCollection(samples, parameters)


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

    scenarios = SampleCollection(samples, uncertainties, experiments.shape[0])
    scenarios.kind = Sample

    return scenarios
