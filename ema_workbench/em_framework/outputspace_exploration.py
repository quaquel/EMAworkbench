""" Provides a genetic algorithm based on novelty search for output space exploration.

The algorithm is inspired by `Chérel et al (2015) <https://doi.org/10.1371/journal.pone.0138212>`_. In short,
from Chérel et al, we have taken the idea of the HitBox. Basically, this is an epsilon archive where one
keeps track of how many solutions have fallen into each grid cell. Next, tournament selection based on novelty is
used as the selective pressure. Novelty is defined as 1/nr. of solutions in same grid cell. This is then
combined with auto-adaptive population sizing as used in e-NSGAII. This replaces the use of adaptive Cauchy mutation
as used by Chérel et al. There is also an more sophisticated algorithm that adds auto-adaptive operator selection as
used in BORG.

The algorithm can be used in combination with the optimization functionality of the workbench.
Just pass an OutputSpaceExploration instance as algorithm to optimize.



"""

import functools
import math

from .optimization import BORGDefaultDescriptor
from ..util.ema_exceptions import EMAError

from platypus import (
    TournamentSelector,
    Archive,
    RandomGenerator,
    Dominance,
    AbstractGeneticAlgorithm,
    default_variator,
    AdaptiveTimeContinuation,
    GAOperator,
    SBX,
    PM,
    DifferentialEvolution,
    SPX,
    UM,
    PCX,
    UNDX,
    Multimethod,
)


__all__ = ["OutputSpaceExploration", "AutoAdaptiveOutputSpaceExploration"]


class Novelty(Dominance):
    """Compares to solutions based on their novelty

    Parameters
    ----------
    algorithm : platypus algorithm instance

    """

    def __init__(self, algorithm):
        super().__init__()
        self.algorithm = algorithm

    def compare(self, winner, candidate):
        """Compare two solutions.

        Returns -1 if the first solution dominates the second, 1 if the
        second solution dominates the first, or 0 if the two solutions are
        mutually non-dominated.

        Parameters
        ----------
        winner : Solution
            The first solution.
        candidate : Solution
            The second solution.
        """
        w_score = self.algorithm.archive.get_novelty_score(winner)
        c_score = self.algorithm.archive.get_novelty_score(candidate)

        if w_score > c_score:
            return -1
        if w_score == c_score:
            return 0
        else:
            return 1


class HitBox(Archive):
    """Hit Box archive

    Parameters:
    ----------
    grid_spec : list of tuples
                each tuple specifies the minimum value, maximum value, and epsilon

    This class implements both the hit box for calculating novelty
    as well as maintaining an archive of solutions. Per grid cell, the solution
    closest to the centre of the cell is kept. This archive thus functions
    very similar to an EpsilonArchive, including tracking epsilon progress.

    TODO: you actually only need the epsilons and if available
    TODO: any constraints

    """

    def __init__(self, grid_spec):
        super().__init__(None)
        self.archive = {}
        self.centroids = {}
        self.grid_counter = {}
        self.grid_spec = grid_spec
        self.improvements = 0
        self.overall_novelty = 0

    def add(self, solution):
        key = get_index_for_solution(solution, self.grid_spec)

        try:
            self.grid_counter[key] += 1
        except KeyError:
            self.grid_counter[key] = 1
            self.improvements += 1

            centroid = [
                self.grid_spec[i][0] + (entry + 0.5) * self.grid_spec[i][2]
                for i, entry in enumerate(key)
            ]
            self.centroids[key] = centroid
            self.archive[key] = solution
        else:
            centroid = self.centroids[key]

            distance_s = [(a - b) ** 2 for a, b in zip(solution.objectives, centroid)]
            distance_s = math.sqrt(sum(distance_s))

            distance_c = [(a - b) ** 2 for a, b in zip(self.archive[key].objectives, centroid)]
            distance_c = math.sqrt(sum(distance_c))

            if distance_s < distance_c:
                self.archive[key] = solution

        self._contents = list(self.archive.values())
        self.overall_novelty += 1 / self.grid_counter[key]

        return True

    def get_novelty_score(self, solution):
        key = get_index_for_solution(solution, self.grid_spec)
        return 1 / self.grid_counter[key]


class OutputSpaceExplorationAlgorithm(AbstractGeneticAlgorithm):
    def __init__(
        self,
        problem,
        grid_spec=None,
        population_size=100,
        generator=RandomGenerator(),
        variator=None,
        **kwargs,
    ):
        if problem.nobjs != len(grid_spec):
            raise EMAError(
                "the number of items in grid_spec does not match the number of objectives"
            )
        super().__init__(problem, population_size, generator=generator, **kwargs)
        self.archive = HitBox(grid_spec)
        self.selector = TournamentSelector(2, dominance=Novelty(self))
        self.algorithm = self  # hack for convergence

        self.variator = variator
        self.population = None

        self.comparator = Novelty(self)

    def step(self):
        if self.nfe == 0:
            self.initialize()
        else:
            self.iterate()

        self.result = self.archive

    def initialize(self):
        super().initialize()

        if self.archive is not None:
            self.archive += self.population

        if self.variator is None:
            self.variator = default_variator(self.problem)

    def iterate(self):
        offspring = []

        while len(offspring) < self.population_size:
            parents = self.selector.select(self.variator.arity, self.population)
            offspring.extend(self.variator.evolve(parents))

        self.evaluate_all(offspring)

        # do we want to compare, or just keep the new population
        # novelty changes.
        self.archive.extend(offspring)
        offspring.extend(self.population)
        offspring = sorted(offspring, key=functools.cmp_to_key(self.comparator))
        self.population = offspring[: self.population_size]


def get_index_for_solution(solution, grid_spec):
    """maps the objectives to the key for the grid cell
    into which this solution falls.

    Parameters
    ----------
    solution : platypus Solution instance
    grid_spec :

    Returns
    -------
    tuple

    """
    key = [
        get_bin_index(entry, grid_spec[i][0], grid_spec[i][2])
        for i, entry in enumerate(solution.objectives)
    ]
    key = tuple(key)

    return key


def get_bin_index(value, minumum_value, epsilon):
    """maps the value for a single objective to the index
    of the grid cell along that diemnsion

    Parameters
    ----------
    value
    minumum_value
    epsilon

    Returns
    -------
    int

    """
    return math.floor((value - minumum_value) / epsilon)


class OutputSpaceExploration(AdaptiveTimeContinuation):
    """Basic genetic algorithm for output space exploration using novelty
    search.

    Parameters
    ----------
    problem : a platypus Problem instance
    grid_spec : list of tuples
                with min, max, and epsilon for
                each outcome of interest
    population_size : int, optional


    The algorithm defines novelty using an epsilon-like grid in the output space.
    Novelty is one divided by the number of seen solutions in a given grid cell.
    Tournament selection using novelty is used to create offspring. Crossover
    is done using simulated binary crossover and mutation is done using polynomial
    mutation.

    The epsilon like grid structure for tracking novelty is implemented
    using an archive, the Hit Box. Per epsilon grid cell, a single solution closes
    to the centre of the cell is maintained. This makes the algorithm
    behave virtually identical to `ε-NSGAII <https://link.springer.com/chapter/10.1007/978-3-540-31880-4_27>`_.
    The archive is returned as results and epsilon progress is defined.

    To deal with a stalled search, adaptive time continuation, identical to
    ε-NSGAII is used.

    Notes
    -----
    Output space exploration relies on the optimization functionality of the
    workbench. Therefore, outcomes of kind INFO are ignored. For output
    space exploration the direction (i.e. minimize or maximize) does not matter.

    """

    def __init__(
        self,
        problem,
        grid_spec=None,
        population_size=100,
        generator=RandomGenerator(),
        variator=None,
        **kwargs,
    ):
        super().__init__(
            OutputSpaceExplorationAlgorithm(
                problem,
                grid_spec=grid_spec,
                population_size=population_size,
                generator=generator,
                variator=variator,
                **kwargs,
            )
        )


class AutoAdaptiveOutputSpaceExploration(AdaptiveTimeContinuation):
    """A combination of auto-adaptive operator selection with OutputSpaceExploration.

    The parametrization of all operators is based on the default values as used
    in Borg 1.9.


    Parameters
    ----------
    problem : a platypus Problem instance
    grid_spec : list of tuples
                with min, max, and epsilon for
                each outcome of interest
    population_size : int, optional


    Notes
    -----
    Limited to RealParameters only.

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
        grid_spec=None,
        population_size=100,
        generator=RandomGenerator(),
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
            OutputSpaceExplorationAlgorithm(
                problem,
                grid_spec=grid_spec,
                population_size=population_size,
                generator=generator,
                variator=variator,
                **kwargs,
            )
        )
