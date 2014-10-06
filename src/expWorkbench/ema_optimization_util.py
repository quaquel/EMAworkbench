'''
Created on Oct 12, 2013

@author: jhkwakkel
'''
import random
from collections import defaultdict
import numpy.lib.recfunctions as recfunctions
# from deap.tools import isDominated
import copy
import ema_logging
from expWorkbench.ema_exceptions import EMAError

__all__ = ["mut_polynomial_bounded",
           "mut_uniform_int",
           "make_name",
           "select_tournament_dominance_crowding",
           "generate_individual_outcome",
           "generate_individual_robust",
           "evaluate_population_outcome",
           "evaluate_population_robust",
           "closest_multiple_of_four",
           "compare",
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


def select_tournament_dominance_crowding(individuals, k, nr_individuals):
    """Tournament selection based on dominance (D) between two individuals, if
    the two individuals do not interdominate the selection is made
    based on crowding distance (CD). 
    
    This selection requires the individuals to have a :attr:`crowding_dist`
    attribute, which can be set by the :func:`assignCrowdingDist` function.
    
    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.
    """
    def binary_tournament(ind1, ind2):
        if ind1.fitness.dominates(ind2.fitness):
            return ind1
        elif ind2.fitness.dominatres(ind1.fitness):
            return ind2

        if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
            return ind2
        elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
            return ind1

        if random.random() <= 0.5:
            return ind1
        return ind2

    def tournament(tour_individuals):
        best = tour_individuals[0]
        
        for entry in tour_individuals[1::]:
            best = binary_tournament(best, entry)
        return best
        
    chosen = []
    for i in xrange(0, k):
        tour_individuals = random.sample(individuals, nr_individuals)
        winner = tournament(tour_individuals)
        winner = copy.deepcopy(winner)
        chosen.append(winner)
    return chosen


class MakeName():
    def __init__(self):
        self.counter = 0
        
    def __call__(self):
        self.counter +=1
        
        return str(self.counter)

make_name = MakeName()

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
    return ind

def evaluate_population_robust(population, ri, toolbox, ensemble, cases=None, **kwargs):
    '''
    Helper function for evaluating a population in case of robust optimization
    
    :param population: the population to evaluate
    :param ri: reporinting interval
    :param toolbox: deap toolbox instance
    :param ensemble: the ensemble instance running the optimization
    :param cases: the cases to use in the robust optimization
    
    '''
    for policy in population:
        policy['name'] = make_name()
    
    policies = [dict(member) for member in population]
    
    ensemble._policies = policies
    experiments, outcomes = ensemble.perform_experiments(cases,
                                                reporting_interval=ri, 
                                                **kwargs)
    # debug validation of results
    # we should have an equal nr of scenarios for each policy
    counter = {}
    last = None
    for entry in set(experiments['policy']):
        logical = experiments['policy']==entry
        value = experiments[logical].shape[0]
        counter[entry] = value
        if not last:
            last = value
        else:
            if last != value:
                raise EMAError("something horribly wrong")
    
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
            member_outcomes[key] = value[associated_index]
            
        member.fitness.values = toolbox.evaluate(member_outcomes)

def mut_polynomial_bounded(individual, eta, policy_levers, keys, indpb):
    """Polynomial mutation as implemented in original NSGA-II algorithm in
    C by Deb. Modified to cope with categories, next to continuous variables. 
    
    TODO:: this should be done differently. It should be possible to specify
    the mutator type for each allele, preventing categorical data from using
    this mutator.
    
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