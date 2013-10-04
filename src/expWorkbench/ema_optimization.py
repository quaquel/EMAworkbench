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

from expWorkbench import ema_logging

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
    for key, attr in zip(sorted(keys), attr_list):
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


class TriedSolutions():
    '''
    Helper class to keep track of all solution that have been tried. It is 
    essentially a dict where the key is the individual. Given that individual
    is unhashable, we use make_name to obtain a string representation of the 
    individual. Currently, make_name only sorts the keys at the highest level
    of the dict. Lover level dicts are not sorted, so their is no guarantee
    that in such a situation the fact that two individuals are identical is
    being detected.
    '''
    
    tried_solutions = dict()
    
    def __getitem__(self, ind):
        return self.tried_solutions[make_name(ind)]
    
    def __setitem__(self, ind, value):
        self.tried_solutions[make_name(ind)] = value
        
    def values(self):
        return self.tried_solutions.values()
    
    def keys(self):
        return self.tried_solutions.keys()
        

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

    def dominates(self, option_a, option_b):
        option_a = np.floor(option_a/self.eps)
        option_b = np.floor(option_b/self.eps)
        return np.any(option_a<option_b)
    
    def sort_individual(self, solution):
        sol_values = -1 * np.asarray(solution.fitness.wvalues) # we assume minimization here for the time being
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
    
    def update(self, population):
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
            
#            is_dominated = False
#            has_twin = False
#            to_remove = []
#            for i, hofer in enumerate(self):    # hofer = hall of famer
#                if isDominated(ind.fitness.wvalues, hofer.fitness.wvalues):
#                    is_dominated = True
#                    break
#                elif isDominated(hofer.fitness.wvalues, ind.fitness.wvalues):
#                    to_remove.append(i)
#                elif ind.fitness == hofer.fitness and self.similar(ind, hofer):
#                    has_twin = True
#                    break
#            
#            for i in reversed(to_remove):       # Remove the dominated hofer
#                self.remove(i)
#                removed+=1
#            if not is_dominated and not has_twin:
#                self.insert(ind)
#                added+=1
        return added, removed, e_prog

class NSGA2StatisticsCallback(object):
    '''
    Helper class for tracking statistics about the progression of the 
    optimization
    '''
    
    def __init__(self,
                 weights=(),
                 nr_of_generations=None,
                 crossover_rate=None, 
                 mutation_rate=None,
                 pop_size=None,
                 caching=False):
        '''
        
        :param weights: the weights on the outcomes
        :param nr_of_generations: the number of generations
        :param crossover_rate: the crossover rate
        :param mutation_rate: the mutation rate
        :param caching: parameter controling wether a list of tried solutions
                        should be kept
        
        
        '''
        ema_logging.warning("currently testing epsilong based domination")
        
        self.stats = []
        if len(weights)>1:
#            self.hall_of_fame = ParetoFront(similar=compare)
            a = [1e-9]*len(weights)
            self.hall_of_fame = EpsilonParetoFront(np.asarray(a))
        else:
            self.hall_of_fame = HallOfFame(pop_size)

        if caching:            
            self.tried_solutions = TriedSolutions() 
        self.change = []
        
        self.weights = weights
        self.nr_of_generations = nr_of_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate=mutation_rate

        self.precision = "{0:.%if}" % 2
    
    
    def __get_hof_in_array(self):
        a = []
        for entry in self.hall_of_fame:
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
        self.change.append(self.hall_of_fame.update(population))
        
        for entry in population:
            self.stats.append(entry.fitness.values)
        
        for entry in population:
            try:
                self.tried_solutions[entry] = entry
            except AttributeError:
                break