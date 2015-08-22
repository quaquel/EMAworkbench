'''

This module provides utility classes and functions for performing optimization


'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
import random
import tempfile
import numpy as np
import numpy.lib.recfunctions as recfunctions

import copy
from .import ema_logging
from . import ema_exceptions
from .callbacks import DefaultCallback

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

# Created on Oct 12, 2013
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

#create a correct way of initializing the individual
def generate_individual_outcome(icls, attr_list, keys):
    '''
    Helper function for generating an individual in case of outcome 
    optimization

    Parameters
    ----------
    icls : class of the individual
    attr_list : list
                list of initializers for each attribute
    keys : str
           the name of each attribute
    
    Returns
    -------
    an instantiated individual
    
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

    Parameters
    ----------
    individuals : list
                  A list of individuals to select from.
    k : int
        The number of individuals to select.
    
    Returns
    -------
    A list of selected individuals.
    """
    def binary_tournament(ind1, ind2):
        if ind1.fitness.dominates(ind2.fitness):
            return ind1
        elif ind2.fitness.dominates(ind1.fitness):
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
    for _ in xrange(0, k):
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

    Parameters
    ----------
    icls: class of the individual
    attr_list : list
                list of initializers for each attribute
    keys : str
           the name of each attribute
    
    Returns
    -------  
    an instantiated individual
    
    '''
    ind = generate_individual_outcome(icls, attr_list, keys)
    return ind


def evaluate_population_robust(population, ri, toolbox, ensemble, cases=None, 
                               **kwargs):
    '''
    Helper function for evaluating a population in case of robust optimization

    Parameters
    ----------
    population : list
                 the population to evaluate
    ri : int
        reporinting interval
    toolbox: deap toolbox instance
    ensemble : ModelEnsemble instance
               the ensemble instance running the optimization
    cases : list
            the cases to use in the robust optimization
    
    '''
    for policy in population:
        policy['name'] = make_name()
    
    policies = [dict(member) for member in population]
    
    ensemble._policies = policies
    experiments, outcomes = ensemble.perform_experiments(cases,
                                                reporting_interval=ri, 
                                                callback=MemmapCallback,
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
                raise ema_exceptions.EMAError("something horribly wrong")
    
    for member in population:
        member_outcomes = {}
        for key, value in outcomes.items():
            logical = experiments["policy"] == member["name"]
            member_outcomes[key] = value[logical]
            
        member.fitness.values = toolbox.evaluate(member_outcomes)


def evaluate_population_outcome(population, ri, toolbox, ensemble):
    '''
    Helper function for evaluating a population in case of outcome optimization

    Parameters
    ----------
    population : list
                  the population to evaluate
    ri : int    
         reporting interval
    toolbox : deap toolbox instance
    ensemble : ModelEnsemble instance
               the ensemble instance running the optimization
    
    '''
    
    cases = [dict(member) for member in population]
    experiments, outcomes = ensemble.perform_experiments(cases,
                                                reporting_interval=ri)

    # TODO:: model en policy moeten er wel in blijven, 
    # dit stelt je in staat om ook over policy en models heen te kijken
    # naar wat het optimimum is. Dus je moet aan x
    # standaard alle models en alle policies toevoegen en dan pas 
    # je index opvragen
    # Dit levert wel 2 extra geneste loops op... 
    
    experiments = recfunctions.drop_fields(experiments,\
                                           drop_names=['model', 'policy'], 
                                           asrecarray = True)    
    ordering = [entry[0] for entry in experiments.dtype.descr]
    
    experiments = experiments.tolist()
    indices = {tuple(experiments[i]):i for i in range(len(experiments))}
    
    # we need to map the outcomes of the x back to the 
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

    Parameters
    ----------
    individual : object
                 Individual to be mutated.
    eta : float
          Crowding degree of the mutation. A high eta will produce a mutant 
          resembling its parent, while a small eta will produce a solution 
          much more different.
    policy_levers :
    keys :
    indpb : 
    
    Returns
    ------- 
    A tuple of one individual.
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

    Parameters
    ----------
    ind1 : object
           individual 1
    ind2 : object
           individual 2
    
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

    Parameters
    ----------
    low: The lower bound of the range from wich to draw the new
                integer.
    up: The upper bound of the range from wich to draw the new
                integer.
    indpb: Probability for each attribute to be mutated.
    
    Returns
    ------- 
    A tuple of one individual.
    """
    for _, entry in enumerate(policy_levers.iteritems()):
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


class MemmapCallback(DefaultCallback):
    '''simple extension of default callback that uses memmap for storing 
    
    This resolves getting memory errors due to adaptive population sizing
    
    '''
    
    def _store_result(self, case_id, result):
        for outcome in self.outcomes:
            ema_logging.debug("storing {}".format(outcome))
            
            try:
                outcome_res = result[outcome]
            except KeyError:
                ema_logging.debug("%s not in msi" % outcome)
            else:
                try:
                    self.results[outcome][case_id, ] = outcome_res
                    self.results[outcome].flush()
                except KeyError: 
                    data =  np.asarray(outcome_res)
                    
                    shape = data.shape
                    
                    if len(shape)>2:
                        raise ema_exceptions.EMAError(self.shape_error_msg.format(len(shape)))
                    
                    shape = list(shape)
                    shape.insert(0, self.nr_experiments)
                    shape = tuple(shape)
                    
                    fh = tempfile.TemporaryFile()
                    self.results[outcome] =  np.memmap(fh, 
                                                       dtype=data.dtype, 
                                                       shape=shape)
                    self.results[outcome][:] = np.NAN
                    self.results[outcome][case_id, ] = data
                    self.results[outcome].flush()