'''
Created on 21 okt. 2012

Helper module with functions used by the model ensemble when perfomring
an optimization. 

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

'''
from __future__ import division
import numpy as np
import numpy.lib.recfunctions as recfunctions
import random 

from deap.tools import HallOfFame, isDominated

from deap import base
from deap import creator
from deap import tools

from expWorkbench import ema_logging
from expWorkbench import debug, info

import abc
from expWorkbench.ema_exceptions import EMAError

__all__ = ["mut_polynomial_bounded",
           "NSGA2StatisticsCallback",
           "generate_individual_outcome",
           "generate_individual_robust",
           "evaluate_population_outcome",
           "evaluate_population_robust",
           "closest_multiple_of_four"
           ]

#create a correct way of initializing the individual
def generate_individual_outcome(icls, attr_list, keys):
    '''
    Helper function for generating an individual in case of outcome 
    optimization
    
    :param icls: class of the individual
    :param attr_list: list of initializers for each attribute
    :param keys: the name of each attribute
    :returns: an instantiated individual
    
    '''
    ind = icls()
    for key, attr in zip(keys, attr_list):
        ind[key] = attr()
    return ind

#create a correct way of initializing the individual
def generate_individual_robust(icls, attr_list, keys):
    '''
    Helper function for generating an individual in case of robust optimization
    
    :param icls: class of the individual
    :param attr_list: list of initializers for each attribute
    :param keys: the name of each attribute
    :returns: an instantiated individual
    
    '''
    ind = generate_individual_outcome(icls, attr_list, keys)
    ind['name'] = make_name(ind) 
    return ind

def make_name(ind):
    keys  = sorted(ind.keys())
    try:
        keys.pop(keys.index('name'))
    except ValueError:
        ema_logging.debug("value error in make name, field 'name' not in list")
    
    name = ""
    for key in keys:
        name += " "+str(ind[key])
    return name
    

def evaluate_population_robust(population, ri, toolbox, ensemble, cases=None, **kwargs):
    '''
    Helper function for evaluating a population in case of robust optimization
    
    :param population: the population to evaluate
    :param ri: reporinting interval
    :param toolbox: deap toolbox instance
    :param ensemble: the ensemble instance running the optimization
    :param cases: the cases to use in the robust optimization
    
    '''
    ensemble._policies = [dict(member) for member in population]
    experiments, outcomes = ensemble.perform_experiments(cases,
                                                reporting_interval=ri, 
                                                **kwargs)
    
    for member in population:
        member_outcomes = {}
        for key, value in outcomes.items():
            logical = experiments["policy"] == member["name"]
            member_outcomes[key] = value[logical]
            
        member.fitness.values = toolbox.evaluate(member_outcomes)

def evaluate_population_outcome(population, ri, toolbox, ensemble):
    '''
    Helper function for evaluating a population in case of outcome optimization
    
    :param population: the population to evaluate
    :param ri: reporting interval
    :param toolbox: deap toolbox instance
    :param ensemble: the ensemble instance running the optimization
    
    '''
    
    cases = [dict(member) for member in population]
    experiments, outcomes = ensemble.perform_experiments(cases,
                                                reporting_interval=ri)

    # TODO:: model en policy moeten er wel in blijven, 
    # dit stelt je in staat om ook over policy en models heen te kijken
    # naar wat het optimimum is. Dus je moet aan experiments
    # standaard alle models en alle policies toevoegen en dan pas 
    # je index opvragen
    # Dit levert wel 2 extra geneste loops op... 
    
    experiments = recfunctions.drop_fields(experiments,\
                                           drop_names=['model', 'policy'], 
                                           asrecarray = True)    
    ordering = [entry[0] for entry in experiments.dtype.descr]
    
    experiments = experiments.tolist()
    indices = {tuple(experiments[i]):i for i in range(len(experiments))}
    
    # we need to map the outcomes of the experiments back to the 
    # correct individual
    for member in population:
        index = tuple([member[entry] for entry in ordering])
        associated_index = indices[index]
        
        member_outcomes = {}
        for key, value in outcomes.items():
            member_outcomes[key] = value[associated_index, :]
            
        member.fitness.values = toolbox.evaluate(member_outcomes)

def mut_polynomial_bounded(individual, eta, policy_levers, keys, indpb):
    """Polynomial mutation as implemented in original NSGA-II algorithm in
    C by Deb. Modified to cope with categories, next to continuous variables. 
    
    :param individual: Individual to be mutated.
    :param eta: Crowding degree of the mutation. A high eta will produce
                a mutant resembling its parent, while a small eta will
                produce a solution much more different.
    :param policy_levers:
    :param keys:
    :returns: A tuple of one individual.
    """
        
    for key in keys:
        if random.random() <= indpb:
            x = individual[key]

            type_allele = policy_levers[key]['type'] 
            value = policy_levers[key]["values"]
            if type_allele=='range float':
                xl = value[0]
                xu = value[1]
                delta_1 = (x - xl) / (xu - xl)
                delta_2 = (xu - x) / (xu - xl)
                rand = random.random()
                mut_pow = 1.0 / (eta + 1.)
    
                if rand < 0.5:
                    xy = 1.0 - delta_1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * xy**(eta + 1)
                    delta_q = val**mut_pow - 1.0
                else:
                    xy = 1.0 - delta_2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy**(eta + 1)
                    delta_q = 1.0 - val**mut_pow
    
                x = x + delta_q * (xu - xl)
                x = min(max(x, xl), xu)
                
                individual[key] = x
            elif type_allele=='range int':
                xl = value[0]
                xu = value[1]
                delta_1 = (x - xl) / (xu - xl)
                delta_2 = (xu - x) / (xu - xl)
                rand = random.random()
                mut_pow = 1.0 / (eta + 1.)
    
                if rand < 0.5:
                    xy = 1.0 - delta_1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * xy**(eta + 1)
                    delta_q = val**mut_pow - 1.0
                else:
                    xy = 1.0 - delta_2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy**(eta + 1)
                    delta_q = 1.0 - val**mut_pow
    
                x = x + delta_q * (xu - xl)
                x = min(max(x, xl), xu)
                
                individual[key] = int(round(x))
                
            elif type_allele=='list':
                individual[key] = random.choice(value)
    return individual,



def compare(ind1, ind2):
    '''
    Helper function for comparing to individuals. Returns True if all fields
    are the same, otherwise False.
    
    :param ind1: individual 1
    :param ind2: individual 2
    
    '''
    
    for key in ind1.keys():
        if ind1[key] != ind2[key]:
            return False

    return True

def closest_multiple_of_four(number):
    '''
    Helper function for transforming the population size to the closest
    multiple of four. Is necessary because of implementation issues of the 
    NSGA2 algorithm in deap. 
    
    '''
    
    return number - number % 4

def mut_uniform_int(individual, policy_levers, keys):
    """Mutate an individual by replacing attributes, with probability *indpb*,
    by a integer uniformly drawn between *low* and *up* inclusively.
    
    :param low: The lower bound of the range from wich to draw the new
                integer.
    :param up: The upper bound of the range from wich to draw the new
                integer.
    :param indpb: Probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    for i, entry in enumerate(policy_levers.iteritems()):
        if random.random() < 1/len(policy_levers.keys()):
            key, entry = entry
            values = entry['values']
            if entry['type'] == 'range float':
                individual[key] = random.uniform(values[0], values[1])
            elif entry['type'] == 'range int':
                individual[key] = random.randint(values[0], values[1])
            elif entry['type'] == 'list':
                individual[key] = random.choice(values)
            else:
                raise NotImplementedError("unknown type: {}".format(entry['type']))
    
    return individual,

class AbstractOptimizationAlgorithm(object):
    
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, evaluate_population, generate_individual, 
                 levers, reporting_interval, obj_function,
                 ensemble, crossover_rate, mutation_rate, weights,
                 pop_size):
        self.evaluate_population = evaluate_population
        self.levers = levers
        self.reporting_interval = reporting_interval
        self.ensemble = ensemble
        self.crossover_rate = crossover_rate 
        self.mutation_rate = mutation_rate 
        self.weights = weights
        self.obj_function = obj_function
        self.pop_size = pop_size
        
        #create a class for the individual
        creator.create("Fitness", base.Fitness, weights=self.weights)
        creator.create("Individual", dict, 
                       fitness=creator.Fitness) #@UndefinedVariable
        self.toolbox = base.Toolbox()
        self.levers = levers
    
        self.attr_list = []
        self.lever_names = []
        for key, value in levers.iteritems():
            lever_type = value['type']
            values = value['values']
            
            if lever_type=='list':
                self.toolbox.register(key, random.choice, values)
            else:
                if lever_type == 'range int':
                    self.toolbox.register(key, random.randint, 
                                          values[0], values[1])
                elif lever_type == 'range float':
                    self.toolbox.register(key, random.uniform, 
                                          values[0], values[1])
                else:
                    raise EMAError("unknown allele type: possible types are range and list")

            self.attr_list.append(getattr(self.toolbox, key))
            self.lever_names.append(key)

        # Structure initializers
        self.toolbox.register("individual", 
                         generate_individual, 
                         creator.Individual, #@UndefinedVariable
                         self.attr_list, keys=self.lever_names) 
        self.toolbox.register("population", tools.initRepeat, list, 
                         self.toolbox.individual)
    
        # Operator registering
        self.toolbox.register("evaluate", self.obj_function)
        
        self.get_population = self._first_get_population
        self.called = 0
        
        #some statistics logging
        self.stats_callback = NSGA2StatisticsCallback(algorithm=self)
    
    @abc.abstractmethod
    def _first_get_population(self):
        pass

    @abc.abstractmethod
    def _get_population(self):
        pass

class NSGA2(AbstractOptimizationAlgorithm):
    
    def __init__(self, weights, levers, generate_individual, obj_function,
                 pop_size, evaluate_population, nr_of_generations, 
                 crossover_rate,mutation_rate, reporting_interval,
                 ensemble):

        # generate population
        # for some stupid reason, DEAP demands a multiple of four for 
        # population size in case of NSGA-2 
        pop_size = closest_multiple_of_four(pop_size)
        info("population size restricted to %s " % (pop_size))
               
        super(NSGA2, self).__init__(evaluate_population, generate_individual, 
                 levers, reporting_interval, obj_function,
                 ensemble, crossover_rate, mutation_rate, weights,
                 pop_size)
        
        self.archive = ParetoFront(similar=compare)
        self.toolbox.register("crossover", tools.cxOnePoint)
        self.toolbox.register("mutate", mut_polynomial_bounded)
        self.toolbox.register("select", tools.selNSGA2)

    def _first_get_population(self):
        debug("Start of evolution")
        
        self.pop = self.toolbox.population(self.pop_size)
        
        # Evaluate the entire population
        self.evaluate_population(self.pop, self.reporting_interval, self.toolbox, 
                                 self.ensemble)

        # This is just to assign the crowding distance to the individuals
        tools.assignCrowdingDist(self.pop)        

        self.stats_callback(self.pop)
        self.stats_callback.log_stats(self.called)
        self.get_population = self._get_population
    
    def _get_population(self):
        self.called +=1
        pop_size = len(self.pop)
        a = self.pop[0:closest_multiple_of_four(len(self.pop))]
        
        offspring = tools.selTournamentDCD(a, len(self.pop))
        offspring = [self.toolbox.clone(ind) for ind in offspring]
        
        no_name=False
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # Apply crossover 
            if random.random() < self.crossover_rate:
                keys = sorted(child1.keys())
                
                try:
                    keys.pop(keys.index("name"))
                except ValueError:
                    no_name = True
                
                child1_temp = [child1[key] for key in keys]
                child2_temp = [child2[key] for key in keys]
                self.toolbox.crossover(child1_temp, child2_temp)

                if not no_name:
                    for child, child_temp in zip((child1, child2), 
                                             (child1_temp,child2_temp)):
                        name = ""
                        for key, value in zip(keys, child_temp):
                            child[key] = value
                            name += " "+str(child[key])
                        child['name'] = name 
                else:
                    for child, child_temp in zip((child1, child2), 
                                             (child1_temp,child2_temp)):
                        for key, value in zip(keys, child_temp):
                            child[key] = value
                
            #apply mutation
            self.toolbox.mutate(child1, self.mutation_rate, self.levers, self.lever_names, 0.05)
            self.toolbox.mutate(child2, self.mutation_rate, self.levers, self.lever_names, 0.05)

            for entry in (child1, child2):
                del entry.fitness.values
            
       
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        self.evaluate_population(invalid_ind, self.reporting_interval, self.toolbox, 
                                 self.ensemble)

        # Select the next generation population
        self.pop = self.toolbox.select(self.pop + offspring, pop_size)
        self.stats_callback(self.pop)
        self.stats_callback.log_stats(self.called)
        return self.pop

class epsNSGA2(NSGA2):

    def __init__(self, weights, levers, generate_individual, obj_function,
                 pop_size, evaluate_population, nr_of_generations, 
                 crossover_rate,mutation_rate, reporting_interval,
                 ensemble):
        super(epsNSGA2, self).__init__(weights, levers, generate_individual, obj_function,
                 pop_size, evaluate_population, nr_of_generations, 
                 crossover_rate,mutation_rate, reporting_interval,
                 ensemble)
        self.archive = EpsilonParetoFront(np.asarray([1e-3]*len(weights)))
        
        self.desired_labda = 4
    
    def _rebuild_population(self):
        desired_pop_size = self.desired_labda * len(self.archive.items)
        self.pop_size = desired_pop_size
        new_pop = [entry for entry in self.archive.items]
        
        while len(new_pop) < desired_pop_size:
            rand_i = random.randint(0, len(self.archive.items)-1)
            individual = self.archive.items[rand_i]
            mut_uniform_int(individual, self.levers, self.lever_names)
            
            # add to new_pop
            new_pop.append(individual)
        
        return new_pop
        
    def _get_population(self):

        archive_length = len(self.archive.items)
        ema_logging.info(archive_length)
        
        # TODO here a restart check is needed
        labda = self.pop_size/archive_length
        if np.abs(1-(labda/self.desired_labda)) > 0.25:
            self.called +=1
            new_pop = self._rebuild_population()
        
            # update selection presure...
        
            # Evaluate the individuals with an invalid fitness
            self.evaluate_population(new_pop, self.reporting_interval, self.toolbox, 
                                     self.ensemble)
    
            # Select the next generation population
            self.pop = self.toolbox.select(self.pop + new_pop, self.pop_size)
            self.stats_callback(self.pop)
            self.stats_callback.log_stats(self.called)
        else:
            super(epsNSGA2, self)._get_population()


class ParetoFront(HallOfFame):
    """The Pareto front hall of fame contains all the non-dominated individuals
    that ever lived in the population. That means that the Pareto front hall of
    fame can contain an infinity of different individuals.
    
    :param similar: A function that tels the Pareto front whether or not two
                    individuals are similar, optional.
    
    The size of the front may become very large if it is used for example on
    a continuous function with a continuous domain. In order to limit the number
    of individuals, it is possible to specify a similarity function that will
    return :data:`True` if the genotype of two individuals are similar. In that
    case only one of the two individuals will be added to the hall of fame. By
    default the similarity function is :func:`operator.__eq__`.
    
    Since, the Pareto front hall of fame inherits from the :class:`HallOfFame`, 
    it is sorted lexicographically at every moment.
    
    This is a  minutre modification to the original version in DEAP. Update now 
    returns the number of changes that have been made to the front.
    
    """
    def __init__(self, similar=compare):
        self.similar = similar
        HallOfFame.__init__(self, None)
    
    def update(self, population):
        """Update the Pareto front hall of fame with the *population* by adding 
        the individuals from the population that are not dominated by the hall
        of fame. If any individual in the hall of fame is dominated it is
        removed.
        
        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """
        added = 0
        removed = 0
        for ind in population:
            is_dominated = False
            has_twin = False
            to_remove = []
            for i, hofer in enumerate(self):    # hofer = hall of famer
                # replace with  np.any(nd.fitness.wvalues < hofer.fitness.wvalues)
                
                if isDominated(ind.fitness.wvalues, hofer.fitness.wvalues):
                    is_dominated = True
                    break
                elif isDominated(hofer.fitness.wvalues, ind.fitness.wvalues):
                    to_remove.append(i)
                elif ind.fitness == hofer.fitness and self.similar(ind, hofer):
                    has_twin = True
                    break
            
            for i in reversed(to_remove):       # Remove the dominated hofer
                self.remove(i)
                removed+=1
            if not is_dominated and not has_twin:
                self.insert(ind)
                added+=1
        return added, removed

class EpsilonParetoFront(HallOfFame):
    """
    
    an implementation of epsilon non-dominated sorting as discussed in 
    
    Deb et al. (2005)
    
    """
    def __init__(self, eps):
        self.eps = eps
        HallOfFame.__init__(self, None)
        self.update = self._init_update

    def dominates(self, option_a, option_b):
        option_a = np.floor(option_a/self.eps)
        option_b = np.floor(option_b/self.eps)
        return np.any(option_a<option_b)
    
    def sort_individual(self, solution):
        sol_values = -1 * np.asarray(solution.fitness.wvalues) # we assume minimization here for the time being
        sol_values = sol_values/self.normalize
        i = -1
        size = len(self.items) - 1
        
        removed = 0
        added = 0
        e_progress = 0
        
        same_box = False
        
        while i < size:
            i += 1
            archived_solution = self[i]
            arc_sol_values = -1*np.asarray(archived_solution.fitness.wvalues)  # we assume minimization here for the time being
            arc_sol_values = arc_sol_values/self.normalize
    
            a_dom_b = self.dominates(arc_sol_values, sol_values)
            b_dom_a = self.dominates(sol_values, arc_sol_values)
            if a_dom_b & b_dom_a:
                # non domination between a and b
                continue
            if a_dom_b:
                # a dominates b
                return removed, added, e_progress
            if b_dom_a:
                # b dominates a
                self.remove(i)
                removed +=1
                i -= 1
                size -= 1
                continue
            if (not a_dom_b) & (not b_dom_a):
                # same box, use solution closes to lower left corner
                box_left_corner = np.floor(sol_values/self.eps)*self.eps
                d_solution = np.sum((sol_values-box_left_corner)**2) 
                d_archive = np.sum((arc_sol_values-box_left_corner)**2)
                
                same_box = True
                
                if d_archive < d_solution:
                    return removed, added, e_progress
                else:
                    self.remove(i)
                    removed +=1
                    i -= 1
                    size -= 1
                    continue
        
        # non dominated solution
        self.insert(solution)
        added +=1
        if not same_box:
            e_progress += 1
        
        return removed, added, e_progress
    
    def _init_update(self, population):
        '''
        only called in the first iteration, used for
        determining normalization valuess
        '''
        values = []
        for entry in population:
            values.append(entry.fitness.wvalues)
        values = np.asarray(values)
        values = -1*values # we minimize
        self.normalize = np.max(np.abs(values), axis=0)

        self.update = self._update
        return self.update(population)
        
    
    def _update(self, population):
        """
        
        Update the epsilon Pareto front hall of fame with the *population* by adding 
        the individuals from the population that are not dominated by the hall
        of fame. If any individual in the hall of fame is dominated it is
        removed.
        
        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """
        added = 0
        removed = 0
        e_prog = 0
        for ind in population:
            ind_rem, ind_add, ind_e_prog = self.sort_individual(ind)
            added += ind_add
            removed += ind_rem    
            e_prog += ind_e_prog        
        return added, removed, e_prog

class NSGA2StatisticsCallback(object):
    '''
    Helper class for tracking statistics about the progression of the 
    optimization
    '''
    
    def __init__(self,
                 algorithm=None):
        '''
        
        :param weights: the weights on the outcomes
        :param nr_of_generations: the number of generations
        :param crossover_rate: the crossover rate
        :param mutation_rate: the mutation rate
        :param caching: parameter controling wether a list of tried solutions
                        should be kept
        
        
        '''
        self.stats = []
        self.algorithm = algorithm

        ema_logging.warning("currently testing epsilon based domination")

        self.change = []
        
        self.weights = self.algorithm.weights
        self.crossover_rate = self.algorithm.crossover_rate
        self.mutation_rate = self.algorithm.mutation_rate

        self.precision = "{0:.%if}" % 2
    
    
    def __get_hof_in_array(self):
        a = []
        for entry in self.algorithm.archive:
            a.append(entry.fitness.values)
        return np.asarray(a)
    
    def std(self, hof):
        return np.std(hof, axis=0)
    
    def mean(self, hof):
        return np.mean(hof, axis=0)
    
    def minima(self, hof):
        return np.min(hof, axis=0)
        
    def maxima(self, hof):
        return np.max(hof, axis=0)

    def log_stats(self, gen):
        functions = {"minima":self.minima,
                     "maxima":self.maxima,
                     "std":self.std,
                     "mean":self.mean,}
        kargs = {}
        hof = self.__get_hof_in_array()
        line = " ".join("{%s:<8}" % name for name in sorted(functions.keys()))
        
        for name  in sorted(functions.keys()):
            function = functions[name]
            kargs[name] = "[%s]" % ", ".join(map(self.precision.format, function(hof)))
        line = line.format(**kargs)
        line = "generation %s: " %gen + line
        ema_logging.info(line)

    def __call__(self, population):
        self.nr_of_generations = self.algorithm.called
        self.change.append(self.algorithm.archive.update(population))
        
        for entry in population:
            self.stats.append(entry.fitness.values)
        
        for entry in population:
            try:
                self.tried_solutions[entry] = entry
            except AttributeError:
                break