"""Samplers for working with SALib."""

import abc
import operator
import warnings

import numpy as np

from .parameters import IntegerParameter, Parameter
from .samplers import DesignIterator, AbstractSampler

try:
    from SALib.sample import fast_sampler, morris, sobol
except ImportError:
    warnings.warn("SALib samplers not available", ImportWarning, stacklevel=2)
    sobol = morris = fast_sampler = None

# Created on 12 Jan 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ["FASTSampler", "MorrisSampler", "SobolSampler", "get_SALib_problem"]


def get_SALib_problem(uncertainties):
    """Returns a dict with a problem specification as required by SALib."""
    # fixme include distribution information here so salib rescaling works
    # fixme or sample on uniform domain, and handle all rescaling in workbench...

    _warning = False
    uncertainties = sorted(uncertainties, key=operator.attrgetter("name"))
    bounds = []

    for u in uncertainties:
        lower = u.lower_bound
        upper = u.upper_bound
        if isinstance(u, IntegerParameter):
            upper += 1  # to deal with floorin in generate_samples

        bounds.append((lower, upper))

    problem = {
        "num_vars": len(uncertainties),
        "names": [unc.name for unc in uncertainties],
        "bounds": bounds,
    }
    return problem


class SALibSampler(AbstractSampler):
    """Base wrapper class for SALib samplers."""

    def generate_samples(self, parameters:list[Parameter], size:int, rng:np.random.Generator|None = None) -> np.ndarray:
        """Generate samples.

        The main method of :class: `~sampler.Sampler` and its
        children. This will call the sample method for each of the
        uncertainties and return the resulting designs.

        Parameters
        ----------
        uncertainties : collection
                        a collection of Parameter instances
        size : int
               the number of samples to generate.


        Returns:
        -------
        dict
            dict with the uncertainty.name as key, and the sample as value

        """
        problem = get_SALib_problem(parameters)
        samples = self.sample(problem, size)

        return samples

    @abc.abstractmethod
    def sample(self, problem:dict, size:int) -> np.ndarray:
        """Call the underlying salib sampling method and return the samples."""


class SobolSampler(SALibSampler):
    """Sampler generating a Sobol design using SALib.

    Parameters
    ----------
    second_order : bool, optional
                   indicates whether second order effects should be included

    """

    def __init__(self, second_order=True): # noqa: D107
        self.second_order = second_order
        self._warning = False

        super().__init__()

    def sample(self, problem:dict, size:int) -> np.ndarray:
        return sobol.sample(problem, size, calc_second_order=self.second_order)


class MorrisSampler(SALibSampler):
    """Sampler generating a morris design using SALib.

    Parameters
    ----------
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

    def __init__(self, num_levels=4, optimal_trajectories=None, local_optimization=True):  # noqa: D107
        super().__init__()
        self.num_levels = num_levels
        self.optimal_trajectories = optimal_trajectories
        self.local_optimization = local_optimization

    def sample(self, problem:dict, size:int) -> np.ndarray:
        return morris.sample(
            problem,
            size,
            self.num_levels,
            self.optimal_trajectories,
            self.local_optimization,
        )


class FASTSampler(SALibSampler):
    """Sampler generating a Fourier Amplitude Sensitivity Test (FAST).

    Parameters
    ----------
    m : int (default: 4)
        The interference parameter, i.e., the number of harmonics to sum in the
        Fourier series decomposition
    """

    def __init__(self, m=4):  # noqa: D107
        super().__init__()
        self.m = m

    def sample(self, problem:dict, size:int) -> np.ndarray:
        return fast_sampler.sample(problem, size, self.m)
