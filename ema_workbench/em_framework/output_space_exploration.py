
import collections
import functools
import random
from math import floor


from platypus import (Selector, Archive, RandomGenerator, NSGAII,
                      Variator, SBX, default_variator)


__all__ = ['OutputSpaceExplorationNSGAII']


class GridSelector(Selector):

    def __init__(self, algorithm):
        super(Selector, self).__init__()
        self.algorithm = algorithm

    def select_one(self, population):
        # why not look up the grid cell for the population,
        # use this as fitness and thus basis for selection
        # at least you then are evolving only a population


        archive = self.algorithm.archive

        # select from the archive proportional to the counts in the
        # gridspace
        keys, weights = zip(*list(archive.grid_counter.items()))

        # select from the population proportional to the counts in the
        # gridspace
        # weights = []
        # for p in population:
        #     key = get_index_for_solution(p, archive.grid_spec)
        #     weight = archive.grid_counter[key]
        #     weights.append(weight)

        # make the weights inversely proportional
        # TODO might we not do this on the archive after
        # TODO adding entire population
        # inverse by division might work better
        # weights = [1+max(weights) - weight for weight in weights]

        # check plos paper
        # use SingleObjective
        # relabel archive to HitMap
        # do selection using tournament, but with fittness based on inverse rarity


        weights = [1/weight for weight in weights]

        key = random.choices(keys, weights)[0]
        return random.choice(archive.grid[key])

        # return random.choices(population, weights)[0]


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
        # can we use a some 'dominance' class for identifying which counters
        # need to be incremented

        # what structure do we use to track all counters (a numpy array of sorts?)
        # and have some way of getting the index of which grid cell to update?

        # or just go with some dict like thing with meaningfull keys?
        # in a way you do not need to really know the intervals that define a given box
        # you can just to a random.choice inversely proportional to the counts
        # you then use this key to get the solutions in that grid cell and just
        # uniformly select from them

        # domination is irrelevant
        # all you need is map a given output tuple to the right interval and
        # use this to get the key
        # so each key is a tuple with the index of the interval?
        # is that not a simply modulus operator?

        super(HitBox, self).__init__(None)
        self.grid = collections.defaultdict(list)
        self.grid_counter = collections.defaultdict(int)
        self.grid_spec = grid_spec
        self._contents = []

        # do you need to set up this entire structure upfront
        # just use a defaultdict and be done with it?
        # can we have some kind of linked dicts
        # grid_ids  = []
        # for entry in grid_spec:
        #     minv, maxv, eps = entry
        #     n = 1 +  (maxv-minv)/eps
        #     grid_ids.append(range(n))
        #
        # grid_cells = itertools.product(*grid_ids)
        # for entry in grid_cells:
        #     self.grid[entry] = []

    def add(self, solution):
        objectives = solution.objectives

        key = get_index_for_solution(solution, self.grid_spec)
        self.grid[key].append(solution)
        self.grid_counter[key] += 1

        # probably _contents should be a property and
        # you just retrieve the unique solutions from the grid dict
        # matter of glueing all lists together
        self._contents.append(solution) # some needless duplication
        return True


class OutputSpaceExploration(NSGAII):
    # TODO add adaptive time continuation to rebuild
    # a population if novelty stalls
    # question is how to measure novelty stalling? e-progress might work
    # basically if no new grid boxes get filled, no new novelty
    # you could instead also do some kind of relative novelty
    # e.g $\sum_{i=1}^{n} \frac{1}{s_i}$ where $s_i$ is the score
    # in the hitbox

    def __init__(self,
                 problem,
                 grid_spec,
                 population_size=100,
                 generator=RandomGenerator(),
                 **kwargs):
        super().__init__(
            problem,
            population_size,
            generator=generator,
            selector=GridSelector(self),
            **kwargs)
        self.archive = HitBox(grid_spec)

    # def iterate(self):
    #     offspring = []
    #
    #     while len(offspring) < self.population_size:
    #         parents = self.selector.select(self.variator.arity, self.population)
    #         offspring.extend(self.variator.evolve(parents))
    #
    #     self.evaluate_all(offspring)
    #
    #     offspring.extend(self.population)
    #     nondominated_sort(offspring)
    #     self.population = nondominated_truncate(offspring, self.population_size)
    #
    #     if self.archive is not None:
    #         self.archive.extend(self.population)

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
    return floor((value - minumum_value) / epsilon)


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
