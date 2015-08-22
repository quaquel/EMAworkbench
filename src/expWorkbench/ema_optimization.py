'''

Helper module with functions used by the model ensemble when performing
optimization. 

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
import random 
import copy
import abc

import numpy as np
import pandas as pd

from deap import base, creator, tools
from deap.tools import HallOfFame

from . import ema_logging
from .ema_exceptions import EMAError
from .ema_optimization_util import (compare, mut_polynomial_bounded,
                        mut_uniform_int, select_tournament_dominance_crowding)

# Created on 21 okt. 2012
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ["NSGA2StatisticsCallback",
           "NSGA2",
           "epsNSGA2",
           "ParetoFront",
           "EpsilonParetoFront"]


class AbstractOptimizationAlgorithm(object):
    '''Abstract base class for optimization algorithms'''
    
    
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, evaluate_population, generate_individual, 
                 levers, reporting_interval, obj_function,
                 ensemble, crossover_rate, mutation_rate, weights,
                 pop_size):
        self.evaluate_population = evaluate_population
        self.levers = levers
        self.lever_keys = list(levers.keys())
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
    
    @abc.abstractmethod
    def _first_get_population(self):
        pass

    @abc.abstractmethod
    def _get_population(self):
        pass


class NSGA2(AbstractOptimizationAlgorithm):
    '''NSGA2 algorithm'''
    
    tournament_size = 2
    
    def __init__(self, weights, levers, generate_individual, obj_function,
                 pop_size, evaluate_population, nr_of_generations, 
                 crossover_rate,mutation_rate, reporting_interval,
                 ensemble, caching):
        super(NSGA2, self).__init__(evaluate_population, generate_individual, 
                 levers, reporting_interval, obj_function,
                 ensemble, crossover_rate, mutation_rate, weights,
                 pop_size)
        self.caching = caching
        self.cache = {}
        self.pop = []
        
        self.archive = ParetoFront(similar=compare)
        self.stats_callback = NSGA2StatisticsCallback(algorithm=self)
        
        self.toolbox.register("crossover", tools.cxOnePoint)
        self.toolbox.register("mutate", mut_polynomial_bounded)
        self.toolbox.register("select", tools.selNSGA2)

    def _first_get_population(self):
        ''' called only once to initialize some stuff, returns 
        a population. After the first call, _get_population is used instead.
        
        '''
        
        ema_logging.debug("Start of evolution")
        
        new_pop = self.toolbox.population(self.pop_size)
        
        # Evaluate the entire population
        self.evaluate_population(new_pop, self.reporting_interval, self.toolbox, 
                                 self.ensemble)
        self.pop = new_pop

        # This is just to assign the crowding distance to the individuals
        tools.emo.assignCrowdingDist(self.pop)        

        self.stats_callback(self.pop)
        self.stats_callback.log_stats(self.called)
        
        if self.caching:
            self._update_cache(self.pop)
        
        self.get_population = self._get_population
        
    
    def _get_population(self):
        self.called +=1
        
        offspring = select_tournament_dominance_crowding(self.pop[0:len(self.pop)], 
                                                         len(self.pop), 
                                                         self.tournament_size)
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
            self.toolbox.mutate(child1, self.mutation_rate, 
                                self.levers, self.lever_names, 0.05)
            self.toolbox.mutate(child2, self.mutation_rate, 
                                self.levers, self.lever_names, 0.05)
            
            del child1.fitness.values
            del child2.fitness.values
            
        # if caching is used, check if fitness value is already known 
        if self.caching:
            self._run_through_cache(offspring)
            
        # Evaluate the individuals with an invalid fitness
        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        
        if invalid_inds:
            # only evaluate if there is something to evaluate
            self.evaluate_population(invalid_inds, self.reporting_interval, 
                                     self.toolbox, self.ensemble)

            if self.caching:
                self._update_cache(invalid_inds)

            # Select the next generation population
            self.pop = self.toolbox.select(self.pop + offspring, self.pop_size)
        self.stats_callback(self.pop)
        self.stats_callback.log_stats(self.called)
        return self.pop

    def _update_cache(self, population):
        '''helper function to fill the cache with each new population
        
        '''
        
        for ind in population:
            # construct key
            key = [repr(ind.get(entry)) for entry in self.lever_keys]
            key = tuple(key)
            self.cache[key] = ind.fitness.values
    
    def _run_through_cache(self, individuals):
        '''Helper function, check whether individuals already have been 
        evaluated if so use the cached value '''
        invalid_inds = [ind for ind in individuals if not ind.fitness.valid]            
        ema_logging.info('nr. of invalid individuals before checking cache: {}'.format(len(invalid_inds)))
        
        for invalid_ind in invalid_inds:
            # construct key
            key = [repr(invalid_ind.get(entry)) for entry in self.lever_keys]
            key = tuple(key)
            
            try:
                # set value if in caching
                invalid_ind.fitness.values = self.cache[key]
            except KeyError:
                pass

        invalid_inds = [ind for ind in individuals if not ind.fitness.valid]            
        ema_logging.info('nr. of invalid individuals after checking cache: {}'.format(len(invalid_inds)))


class epsNSGA2(NSGA2):
    ''' epsilon-NSGA2 algorithm'''
    
    message = "reset population: pop size: {}; archive: {}; tournament size: {}"

    def __init__(self, weights, levers, generate_individual, obj_function,
                 pop_size, evaluate_population, nr_of_generations, 
                 crossover_rate,mutation_rate, reporting_interval,
                 ensemble, caching, eps, selection_pressure = 0.02):
        super(epsNSGA2, self).__init__(weights, levers, generate_individual, 
                 obj_function, pop_size, evaluate_population,
                 nr_of_generations, crossover_rate,mutation_rate, 
                 reporting_interval,ensemble, caching)
        self.archive = EpsilonParetoFront(eps)
        self.stats_callback = NSGA2StatisticsCallback(algorithm=self)
        self.selection_presure = selection_pressure
        self.desired_labda = 4
        
        # nr. of iterations without epsilon progress after which a reset of the
        # population is activated analogous to borg
        self.time_window = 10 
        self.last_eps_progress = 0
    
    def _rebuild_population(self):
        new_pop = [entry for entry in self.archive.items]
        
        while len(new_pop) < self.pop_size:
            rand_i = random.randint(0, len(self.archive.items)-1)
            individual = self.archive.items[rand_i]
            individual = copy.deepcopy(individual)
            mut_uniform_int(individual, self.levers, self.lever_names)
            
            # add to new_pop
            new_pop.append(individual)
        
        return new_pop
    
    def _restart_required(self):

        # restart checks
        # restart is due to either a 25% difference between the actual
        # archive size or the desired archive size
        archive_length = len(self.archive.items)
        labda = self.pop_size/archive_length
        condition1 = np.abs(1-(labda/self.desired_labda)) >= 0.25

        # or more than self.time_window generations since last epsilon
        # progress
        if (self.stats_callback.change[-1][-1]) > 0:
            self.last_eps_progress = 0
        else:
            self.last_eps_progress +=1
        condition2 = self.last_eps_progress >= self.time_window
        
        if condition1 or condition2:
            return True
        else:
            return False

    def _get_population(self):
        
        if self._restart_required():
            self.called +=1
            self.last_eps_progress = 0
            self.pop_size = self.desired_labda * len(self.archive.items)           
            
            new_pop = self._rebuild_population()

            ema_logging.info(self.message.format(self.pop_size,
                                                 len(self.archive.items),
                                                 self.tournament_size))
            
            # run new population through cache
            invalid_new_pop = new_pop
            if self.cache:
                invalid_new_pop = self._run_through_cache(new_pop)
        
            # update selection pressure...
            self.tournament_size = int(max(2,
                                        self.selection_presure*self.pop_size))

            
            if invalid_new_pop:
                # Evaluate the individuals with an invalid fitness
                self.evaluate_population(invalid_new_pop, self.reporting_interval, 
                                         self.toolbox, self.ensemble)
                
                # update cache with newly analysed population
                if self.caching:
                    self._update_cache(invalid_new_pop)

            # Select the next generation population
            self.pop = self.toolbox.select(self.pop + new_pop, self.pop_size)
            
            self.stats_callback(self.pop)
            self.stats_callback.log_stats(self.called)
            
            return self.pop
        else:
            return super(epsNSGA2, self)._get_population()


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
    
    This is a  minute modification to the original version in DEAP. Update now 
    returns the number of changes that have been made to the front.
    
    """
    def __init__(self, similar=compare):
        super(ParetoFront, self).__init__(None)
        self.similar = similar
    
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
                
                if ind.fitness.dominates(hofer.fitness):
                    to_remove.append(i)
                elif hofer.fitness.dominates(ind.fitness):
                    is_dominated = True
                    break
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
        super(EpsilonParetoFront, self).__init__(None)
        self.eps = eps


    def dominates(self, option_a, option_b):
        option_a = np.floor(option_a/self.eps)
        option_b = np.floor(option_b/self.eps)
        return np.any(option_a<option_b)
    
    def sort_individual(self, solution):
        sol_values = np.asarray(solution.fitness.wvalues) 

        # we assume minimization here for the time being
        sol_values = -1 * sol_values
        
        i = -1
        size = len(self.items) - 1
        removed = 0
        added = 0
        e_progress = 0
        same_box = False
        while i < size:
            i += 1
            archived_solution = self[i]

            # we assume minimization here for the time being
            arc_sol_values = -1*np.asarray(archived_solution.fitness.wvalues)  
    
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
                # same box, use solution closest to lower left corner
                norm_sol_values = sol_values/self.eps
                norm_arc_sol_values = arc_sol_values/self.eps
                box_left_corner = np.floor(norm_sol_values)
                d_solution = np.sum((norm_sol_values-box_left_corner)**2) 
                d_archive = np.sum((norm_arc_sol_values-box_left_corner)**2)
                
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
    
#     def _init_update(self, population):
#         '''
#         only called in the first iteration, used for
#         determining normalization values
#         '''
#         values = []
#         for entry in population:
#             values.append(entry.fitness.wvalues)
#         values = np.asarray(values)
#         values = -1*values # we minimize
#         
#         return self.update(population)
    
    def update(self, population):
        """
        
        Update the epsilon Pareto front hall of fame with the *population* by adding 
        the individuals from the population that are not dominated by the hall
        of fame. If any individual in the hall of fame is dominated it is
        removed.
        
        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """
        
#         if not self.init:
#             self.init = True
#             return self._init_update(population)
        
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
        
        :param algorithm: the optimization algorithm instance
        
        '''
        self.archive = algorithm.archive
        
        self.weights = algorithm.weights
        self.crossover_rate = algorithm.crossover_rate
        self.mutation_rate = algorithm.mutation_rate
        
        self.precision = "{0:.%if}" % 2
        self.nr_of_generations = 0
        self.stats = pd.DataFrame()
        self.change = []


    def __get_hof_in_array(self):
        '''return a numpy array representation of the fitness values of the
        archive'''
        
        return np.asarray([entry.fitness.values   for entry in self.archive])
    
    def std(self, hof):
        return np.std(hof, axis=0)
    
    def mean(self, hof):
        return np.mean(hof, axis=0)
    
    def minima(self, hof):
        return np.min(hof, axis=0)
        
    def maxima(self, hof):
        return np.max(hof, axis=0)

    def log_stats(self, gen):
        '''Log statistics on the progress of the evolution'''
        
        functions = {"min":self.minima,
                     "max":self.maxima,
                     "std":self.std,
                     "mean":self.mean,}

        hof = self.__get_hof_in_array()
        info_message = pd.DataFrame(index=['min', 'max', 'mean', 'std'],
                                    columns=['obj_{}'.format(i) for i in
                                             range(hof.shape[1])])
        for key, value in functions.iteritems():
            data = value(hof)
            info_message.loc[key] = data
            
        # let pandas do the formatting for us, but remove the trailing info
        # on the size of the DataFrame
        message = info_message.__str__()
        message = message.split('\n')[0:-2]
        message = "\n".join(message)
        line = "\ngeneration {}\n{}".format(gen,message)
        ema_logging.info(line)

    def __call__(self, population):
        '''the call function of the callback'''
        
        # update the archive
        scores = self.archive.update(population)
        self.change.append(copy.deepcopy(scores))
        self.nr_of_generations += 1
        
        values = [entry.fitness.values for entry in population]
        values = pd.DataFrame(values)
        self.stats = self.stats.append(values,  ignore_index=True)

