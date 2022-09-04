
import functools
import math


from ema_workbench.em_framework.optimization import BORGDefaultDescriptor

from platypus import (TournamentSelector, Archive, RandomGenerator, NSGAII,
                      Variator, SBX, default_variator, Dominance, AbstractGeneticAlgorithm,
                      GAOperator, DifferentialEvolution, PM, default_variator,
                      AdaptiveTimeContinuation)


__all__ = ['OutputSpaceExplorationNSGAII']


class Novelty(Dominance):

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
    """


    Parameters:
    ----------
    grid_spec : list of tuples
                each tuple specifies the minimum value, maximum value, and epsilon

    TODO: you actually only need the epsilons and if available
    TODO: any constraints



    """

    def __init__(self, grid_spec):

        # domination is irrelevant
        # all you need is map a given output tuple to the right interval and
        # use this to get the key
        # so each key is a tuple with the index of the interval?
        # is that not a simply modulus operator?

        super(HitBox, self).__init__(None)
        self.archive = {}
        self.centroids = {}
        self.grid_counter = {}
        self.grid_spec = grid_spec
        self.improvements = 0
        self.overall_novelty = 0

    def add(self, solution):
        # we can start treating this
        # as a proper archive next to a hitbox
        # by keeping only a single point per hitbox
        # but how to select the point?
        # one idea would be to calculate the centroid for each
        # hit box and then just keep the closest to the centroid

        key = get_index_for_solution(solution, self.grid_spec)
        # self.grid[key].append(solution)

        try:
            self.grid_counter[key] += 1
        except KeyError:
            self.grid_counter[key] = 1
            self.improvements += 1


            # can we generate the centroid we need here?
            # basically take the key, the min, and the epsilon
            # basically the halfway point on each dimension....
            # so min + key * epsilon + 1/2 epsilon
            centroid = [self.grid_spec[i][0] + (entry+0.5)*self.grid_spec[i][2] for i, entry in enumerate(key)]
            self.centroids[key] = centroid
            self.archive[key] =  solution
        else:

            # solution is in an existing box
            # only keep solution closest to centroid
            centroid = self.centroids[key]


            distance_s = [(a - b) ** 2 for a, b in zip(solution.objectives, centroid)]
            distance_s = math.sqrt(sum(distance_s))

            distance_c = [(a - b) ** 2 for a, b in zip(self.archive[key].objectives, centroid)]
            distance_c = math.sqrt(sum(distance_c))

            if distance_s < distance_c:
                self.archive[key] = solution

        # probably _contents should be a property and
        # you just retrieve the unique solutions from the grid dict
        # matter of glueing all lists together
        self._contents = list(self.archive.values())

        # todo some novelty score
        # might have both net_novelty
        # and the equivalent of e_progress: new hitbox
        self.overall_novelty += 1/self.grid_counter[key]

        return True

    def get_novelty_score(self, solution):
        key = get_index_for_solution(solution, self.grid_spec)
        return 1/self.grid_counter[key]


class OutputSpaceExploration(AbstractGeneticAlgorithm):
    # TODO add adaptive time continuation to rebuild
    # TODO trace novelty
    # a population if novelty stalls
    # question is how to measure novelty stalling? e-progress might work
    # basically if no new grid boxes get filled, no new novelty
    # you could instead also do some kind of relative novelty
    # e.g $\sum_{i=1}^{n} \frac{1}{s_i}$ where $s_i$ is the score
    # in the hitbox
    de_rate = 0.1
    de_stepsize = 0.5

    pm_p = BORGDefaultDescriptor(lambda x: 1 / x)
    pm_dist = 20

    def __init__(self,
                 problem,
                 grid_spec=None,
                 population_size=100,
                 generator=RandomGenerator(),
                 **kwargs):
        super().__init__(problem, population_size, generator=generator, **kwargs)
        self.archive = HitBox(grid_spec)
        self.selector = TournamentSelector(2, dominance=Novelty(self))
        self.algorithm = self # hack for convergence

        # what crossover and mutation to use?
        # wrap them in a GAOperator
        # self.variator = GAOperator(
        #     DifferentialEvolution(
        #         crossover_rate=self.de_rate, step_size=self.de_stepsize
        #     ),
        #     PM(probability=self.pm_p, distribution_index=self.pm_dist),
        # )
        self.variator=None

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
        super(OutputSpaceExploration, self).initialize()

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
        self.population = offspring[:self.population_size]



# can be fully vectorized.... using numpy but that requires
# a numpy represenation of all solutions in a single numpy array
# which  might be expensive to generate



def get_index_for_solution(solution, grid_spec):
    key = [get_bin_index(entry, grid_spec[i][0], grid_spec[i][2]) for i, entry in
            enumerate(solution.objectives)]
    key = tuple(key)

    return key

def get_bin_index(value, minumum_value, epsilon):
    """

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


# for testing
# minimum_value = 10
# maximum_value = 20
# epsilon = 0.5
#
# x = np.arange(minimum_value, maximum_value + epsilon, epsilon)
#
# grid = {}
# for i, entry in enumerate(zip(x, x[1::])):
#     grid[i] = entry
#
# for value in np.random.uniform(minimum_value, maximum_value, 10):
#     index = get_bin_index(value, minimum_value, epsilon)
#     # print(value, grid[index])
#     # note that you will get an edge effect if value equals maximum_value
#     assert value > grid[index][0]
#     assert value <= grid[index][1]

# some custom algorithm
# probably based on NSGA2 and AdaptiveTimeContinuation, but without e-archiving
# even better, make a new e-archive, but let it keep track of all solutions per grid cells
# wont this give rise to memory problem
# GridSelector would then only need to know the archive and can derive our stuff from this directly
