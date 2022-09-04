import functools
import math


from ema_workbench.em_framework.optimization import BORGDefaultDescriptor

from platypus import (
    TournamentSelector,
    Archive,
    RandomGenerator,
    Dominance,
    AbstractGeneticAlgorithm,
    GAOperator,
    DifferentialEvolution,
    PM,
    default_variator,
    AdaptiveTimeContinuation,
)


__all__ = ["OutputSpaceExploration"]


class Novelty(Dominance):
    """Comapres to solutions based on their novelty

    Parameters
    ----------
    algorithm : platypus algorithm instance

    """

    def __init__(self, algorithm):
        super(Novelty, self).__init__()
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
        super(HitBox, self).__init__(None)
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

            distance_c = [
                (a - b) ** 2 for a, b in zip(self.archive[key].objectives, centroid)
            ]
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

    de_rate = 0.1
    de_stepsize = 0.5

    pm_p = BORGDefaultDescriptor(lambda x: 1 / x)
    pm_dist = 20

    def __init__(
        self,
        problem,
        grid_spec=None,
        population_size=100,
        generator=RandomGenerator(),
        **kwargs
    ):
        super().__init__(problem, population_size, generator=generator, **kwargs)
        self.archive = HitBox(grid_spec)
        self.selector = TournamentSelector(2, dominance=Novelty(self))
        self.algorithm = self  # hack for convergence

        # what crossover and mutation to use?
        # wrap them in a GAOperator
        # self.variator = GAOperator(
        #     DifferentialEvolution(
        #         crossover_rate=self.de_rate, step_size=self.de_stepsize
        #     ),
        #     PM(probability=self.pm_p, distribution_index=self.pm_dist),
        # )
        self.variator = None
        self.population = None

        self.comparator = Novelty(self)

    def step(self):
        if self.nfe == 0:
            self.initialize()
        else:
            self.iterate()

        if self.archive is not None:
            self.result = self.archive
        else:
            self.result = self.population

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
        offspring.extend(self.population)
        self.archive.extend(offspring)
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
    """ maps the value for a single objective to the index
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
    problem
    grid_spec
    population_size
    nfe


    The algorithm defines novelty using an epsilon-like grid in the output space.
    Novelty is one divided by the number of seen solutions in a given grid cell.
    Tournament selection using novelty is used to create offspring. Crossover
    is done using simulated binary crossover and mutation is done using polynomial
    mutation.

    The epsilon like grid structure for tracking novelty is implemented
    using an archive, the Hit Box. per epsilon grid cell, a single solution closes
    to the centre of the cell is maintained. This makes the algorithm
    behave virtually identical to e-NSGAII. The archive is returned as results
    and epsilon progress is defined.

    To deal with a stalled search, adaptive time continuation, identical to
    e-NSGAII is used.

    """

    def __init__(
        self,
        problem,
        grid_spec=None,
        population_size=100,
        generator=RandomGenerator(),
        **kwargs
    ):
        super().__init__(
            OutputSpaceExplorationAlgorithm(
                problem,
                grid_spec=grid_spec,
                population_size=population_size,
                generator=generator,
                **kwargs
            )
        )
