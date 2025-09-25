"""Samplers for working with SALib."""

# Created on 12 Jan 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

import abc
import warnings

import numpy as np

from .parameters import Parameter
from .points import SampleCollection
from .samplers import AbstractSampler

try:
    from SALib.sample import fast_sampler, morris, sobol
except ImportError:
    warnings.warn("SALib samplers not available", ImportWarning, stacklevel=2)
    sobol = morris = fast_sampler = None

__all__ = ["FASTSampler", "MorrisSampler", "SobolSampler", "get_SALib_problem"]


def get_SALib_problem(parameters: list[Parameter]):
    """Returns a dict with a problem specification as required by SALib."""
    bounds = [(0, 1)] * len(parameters)

    problem = {
        "num_vars": len(parameters),
        "names": [p.name for p in parameters],
        "bounds": bounds,
    }
    return problem


class SALibSampler(AbstractSampler):
    """Base wrapper class for SALib samplers."""

    def generate_samples(
        self,
        parameters: list[Parameter],
        size: int,
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> SampleCollection:
        """Generate samples.

        The main method of :class: `~sampler.Sampler` and its
        children. This will call the sample method for each of the
        uncertainties and return the resulting designs.

        Parameters
        ----------
        parameters : collection
                     a collection of Parameter instances
        size : int
               the number of samples to generate.
        rng: np.random.Generator|None

        Returns
        -------
        dict
            dict with the uncertainty.name as key, and the sample as value

        Notes
        -----
        Salib, at least for version 1.5.1, is not consistent in how it handles the seeding of random number
        generators. Sobol uses the moderns numpy.random.Generator approach, while other parts use the older approach.
        So, when controlling the seeding for SALIB samplers, please check the valid types for seed in salib and
        adapt the value passed to rng accordingly.

        """
        problem = get_SALib_problem(parameters)
        samples = self.sample(problem, size, rng=rng, **kwargs)
        samples = self._rescale(parameters, samples)
        return SampleCollection(samples, parameters)

    @abc.abstractmethod
    def sample(
        self, problem: dict, size: int, rng: np.random.Generator | None, **kwargs
    ) -> np.ndarray:
        """Call the underlying salib sampling method and return the samples.

        Any additional keyword arguments will be passed to the underlying salib sampling method

        Parameters
        ----------
        problem : a dictionary with the problem specification
        size : the number of samples to generate
        rng : a np.random.Generator, or something that can seed a rgn.
        kwargs : any additional keyword arguments

        Additional valid keyword arguments are
        parameters : collection
             a collection of Parameter instances
        size : int
               the number of samples to generate.
        rng: np.random.Generator|None

        """


class SobolSampler(SALibSampler):
    """Sampler generating a Sobol design using SALib."""

    def sample(
        self, problem: dict, size: int, rng: np.random.Generator | None, **kwargs
    ) -> np.ndarray:
        """Call the underlying salib sampling method and return the samples.

        Any additional keyword arguments will be passed to the underlying salib sampling method

        Parameters
        ----------
        problem : a dictionary with the problem specification
        size : the number of samples to generate
        rng : a np.random.Generator, or something that can seed a rgn.
        kwargs : any additional keyword arguments

        Additional valid keyword arguments are
        calc_second_order : bool, optional
            Calculate second-order sensitivities. Default is True.
        scramble : bool, optional
            If True, use LMS+shift scrambling. Otherwise, no scrambling is done.
            Default is True.
        skip_values : int, optional
            Number of points in Sobol' sequence to skip, ideally a value of base 2.
            It's recommended not to change this value and use `scramble` instead.
            `scramble` and `skip_values` can be used together.
            Default is 0.

        """
        return sobol.sample(problem, size, seed=rng, **kwargs)


class MorrisSampler(SALibSampler):
    """Sampler generating a morris design using SALib."""

    def sample(
        self, problem: dict, size: int, rng: np.random.Generator | None, **kwargs
    ) -> np.ndarray:
        """Call the underlying salib sampling method and return the samples.

        Any additional keyword arguments will be passed to the underlying salib sampling method

        Parameters
        ----------
        problem : a dictionary with the problem specification
        size : the number of samples to generate
        rng : a np.random.Generator, or something that can seed a rgn.
        kwargs : any additional keyword arguments

        Additional valid keyword arguments are
        num_levels : int
            The number of grid levels
        grid_jump : int
            The grid jump size
        optimal_trajectories : int, optional
            The number of optimal trajectories to sample (between 2 and N)
        local_optimization : bool, optional
            Flag whether to use local optimization according to Ruano et al. (2012)
            Speeds up the process tremendously for bigger N and num_levels.
            Stating this variable to be true causes the function to ignore gurobi.

        """
        return morris.sample(problem, size, seed=rng, **kwargs)


class FASTSampler(SALibSampler):
    """Sampler generating a Fourier Amplitude Sensitivity Test (FAST)."""

    def sample(
        self, problem: dict, size: int, rng: np.random.Generator | None, **kwargs
    ) -> np.ndarray:
        """Call the underlying salib sampling method and return the samples.

        Any additional keyword arguments will be passed to the underlying salib sampling method

        Parameters
        ----------
        problem : a dictionary with the problem specification
        size : the number of samples to generate
        rng : a np.random.Generator, or something that can seed a rgn.
        kwargs : any additional keyword arguments

        Additional valid keyword arguments are
        M : int (default: 4)
            The interference parameter, i.e., the number of harmonics to sum in the
            Fourier series decomposition (default 4)
        Fourier series decomposition

        """
        return fast_sampler.sample(problem, size, seed=rng, **kwargs)
