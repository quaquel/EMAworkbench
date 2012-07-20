from __future__ import division

'''
Created on 27 apr. 2011

@author: j.h. kwakkel

this module offers support for the generation of 
regret figures similar to those reported by RAND
in shaping the next one hundred years.
'''
import numpy as np
from expWorkbench.uncertaintySpace import FullFactorialSampler
from expWorkbench import EMAlogging

#==============================================================================
#
#    helper functions for preparing the results from runExperiments for easier
#    regret calculations
#
#==============================================================================

def build_dictonary_structure(results):
    '''
    this helper function transform the results list into
    a dictionary with the case and policy as key, and the
    associated outcome vector as value
    
    input:
    results         default returnValue from modelEnsemble.runExperiments()
    
    returns:
    outcomes        dict of dicts, indexed by policy and case
    uncertainties   list uncertainties, order is order in case tuple
    '''
    
    outcomes = {}
    uncertainties = []
    for result in results:
        case = result[0][0]
        policy = result[0][1]['name']
        outcome = result[1]
        
        if not outcomes.has_key(policy):
            outcomes[policy] = {}
        if not uncertainties:
            uncertainties = case.keys()
        case = [case.get(key) for key in uncertainties]
        case = tuple(case)
        
        outcomes[policy][case] = outcome
    
    return outcomes, uncertainties

def build_performance_arrays(results):
    '''
    this helper function transforms the results list into
    a performance array for each policy. The performance 
    array is indexed case by outcome.
    
    this function returns a list containing the cases, and
    an array for each policy
    
    input:
    results         default returnValue from modelEnsemble.runExperiments()
    
    returns: 
    outcomes        dict with the performance array for each policy
    cases           list with the cases, order is identical to the 
                    rows in the array
    OoIs            list outcomes of interest, order is identical to 
                    columns in array
    uncertainties   list uncertainties, order is identical to order 
                    of entry in cases
    '''
    results, uncertainties = build_dictonary_structure(results)
    cases = []
    outcomes = {}
    OoIs = []
    for key, value in results.items():
        if not cases:
            cases =  value.keys()
        if not OoIs:
            OoIs = value.values()[0].keys()
        
        outcome = []
        for case in cases:
            case = value.get(case)
            casePerformance = []
            for OoI in OoIs:
                if len(case.get(OoI)) != 1:
                    casePerformance.append(case[OoI][-1])
                else:
                    casePerformance.append(case[OoI])
            outcome.append(casePerformance)
        
        outcome = np.asarray(outcome)
        outcomes[key] = (outcome)
    return outcomes, cases, OoIs, uncertainties

#==============================================================================
#
#   normalizing arrays using specified minima and maxima
#
#==============================================================================
def normalize_array(array, minima, maxima):
    '''
    normalize an array on a column by column basis
    
    input
    array    2-d array to be normalized
    minima   1-d array with minima for each column
    maxima   1-d array with maxima for each column
    
    
    return
    normalized array
    '''
    c = minima-maxima
    
    
    a  = -1/(c)
    
    b = 1+ maxima/(c)
    d = [i for i, entry in enumerate(c == 0) if entry]
    array = a * array + b
    
    #everything that is NaN given by d should be set to 0
    #NaN occurs in case minima-maxima = 0
    for entry in d:
        array[:, entry] = 0 

    
    return array

#==============================================================================
#
#    regret calculation functions
#
#==============================================================================
def perform_regret_analysis(results,
                          policyOfInterest,
                          uncertainty1,
                          uncertainty2,
                          resolution,
                          outcomeNames = []):
    '''
    perform a RAND-style regret analysis. That is, calculate regret across 
    all runs. Regret is here understood as the regret of the policy of 
    interest as compared to the best performing other policy. 
    
    Identify the case in which the regret is maximized. Show a 2-d slice 
    across two specified uncertainties, which contains the case where the 
    regret is maximized. So, in this slice all the uncertainties apart from 
    the 2 specified, are equal to their value in the case were the regret 
    is maximized. 
    
    Function requires a full factorial sampling as the experimental design
    to work.
    
    input:
    results             default returnValue from modelEnsemble.runExperiments()
    policyOfInterest    name of policy for which you want to calculate the 
                        regret
    uncertainty1        the uncertainty across which you want to slice
    uncertainty2        the uncertainty across which you want to slice
    resolution          resolution used in generating the full factorial
    outcomeNames        if provided, this should be a list of names of outcomes 
                        where high is bad the normalized results for these 
                        outcomes will be reverted
    
    NOTE: please provide the actual uncertainty, not their name
    
    returns:
    regret          1-d array that specifies the regret of policy to 
                    all other policies
    case    
    '''
    def getIndex(range, resolution, value):
        '''
        helper function to transform a case to an index in the regretPlotArray
        '''
        
        return ((resolution-1) * (value- range[0]))/ (range[1]-range[0]) 
        
    
    regret, cases, uncertainties = calculate_regret(results, 
                                                    policyOfInterest,
                                                    outcomeNames)

    # transform regret into a dictionary for quick lookup    
    regretDict = {}
    for entry in zip(cases, regret):
        regretDict[entry[0]] = entry[1]

    #identify maximum regret case
    maximumRegret, case = max_regret(regret, cases)
    
    # generate the cases that should be in the slice
    #
    # by generating the cases we need for the slice here
    # and combining it with the dict structure, we can fill the 
    # slice up quickly 
    #
    # another alternative approach would be to filter the available cases
    # based on the case that maximizes the regret. Only the specified 
    # uncertainties should be allowed to vary. This, however, would require 
    # us to go over the entire list of cases which can potentially become 
    # very slow
    #
    sampler = FullFactorialSampler()
    designs = sampler.generate_design([uncertainty1, 
                                      uncertainty2], 
                                      resolution)[0]
    designs = [design for design in designs]
    
    # get the indexes of the uncertainties
    # we use the max regret case and only modify the entries for
    # the uncertainties across which we want to slice
    index1 = uncertainties.index(uncertainty1.name)
    index2 = uncertainties.index(uncertainty2.name)
    
    # deduce the shape of the slice
    if len(designs) < resolution**2:
        resolution1 = len(set(np.asarray(designs)[:, 0]))
        resolution2 = len(set(np.asarray(designs)[:, 1]))
        shape = (resolution1, resolution2)
    else:
        shape = (resolution, resolution)
   
    regretPlot = np.zeros(shape)  
    case = list(case)
    for design in designs:
        case[index1] = design[0]
        case[index2] = design[1]
    
        # map case values back to index in regretPlot
        i = int(round( getIndex(uncertainty1.get_values(), 
                                regretPlot.shape[0], 
                                design[0]), 0)) 
        j = int(round( getIndex(uncertainty2.get_values(), 
                                regretPlot.shape[1], 
                                design[1]), 0))
        
        # retrieve regret for particular case
        try: 
            a = regretDict.get(tuple(case))
#            print a
            regretPlot[i, j] = np.max(a)
        except KeyError as e:
            EMAlogging.exception('case not found')
            raise e
    return regretPlot    

def calculate_regret(results, 
                    policyOfInterest,
                    outcomeNames = []):
    '''
    calculates the regret on a case by case level
    
    input:
    results             default returnValue from modelEnsemble.runExperiments()
    policyOfInterest    name of policy for which you want to calculate the 
                        regret
    outcomeNames        if provided, this should be a list of names of 
                        outcomes where high is bad the normalized results 
                        for these outcomes will be reverted
    
    returns:
    regret        1-d array that specifies the regret of policy to 
                  all other policies
    cases         list of cases, order is same as the 1-d regret array
    '''
    
    outcomes, cases, OoIs, uncertainties = build_performance_arrays(results)
    
    #normalize results
    overallOutcomes = []
    for outcome in outcomes.values():
        overallOutcomes.append(outcome)
    overallOutcomes = np.concatenate(overallOutcomes)
    minima = np.min(overallOutcomes, axis=0)
    maxima = np.max(overallOutcomes, axis=0)
    
    for policy, outcome  in outcomes.items():
        outcomes[policy] = normalize_array(outcome, minima, maxima)
    
    #invert the normalized results for the specified outcomes
    for name in outcomeNames:
        index =  OoIs.index(name)
        for policy in outcomes.keys():
            a = outcomes[policy]
            a[:, index] = a[:, index] - 1
            a[:, index] = np.abs(a[:, index])
            outcomes[policy] = a 
    
    # calculate performance of policy of interest and all other policies
#    print outcomes.keys()
    policyOfInterest = outcomes.pop(policyOfInterest)
    policyOfInterest = L2(policyOfInterest)
#    print outcomes.keys()
    policyPerformance = np.zeros((len(cases), len(outcomes.keys())))
    for i, value in enumerate(outcomes.values()):
        policyPerformance[:, i] =  L2(value)

#    for x in range(policyOfInterest.shape[0]):
#        print str(policyOfInterest[x]) + "\t" + str(policyPerformance[x])

    # calculate regret for policy of interest compared to all other policies
    regret = policyPerformance - policyOfInterest[:, np.newaxis]
    regret[regret < 0 ] = 0

    return regret, cases, uncertainties


def max_regret(regret, cases):
    '''
    find the maximum regret of a policy
    
    input:
    regret    the first return value of calculate regret
    cases     the second return value of calculate regret
    
    returns:
    maxRegret    the value of the maximum regret
    case         the case in which the maximum regret is found
    
    '''
    #get maximum regret, associated case and index of policy with which the 
    #regret is maximal
    maxRegret = np.max(regret)
    caseIndex, policyIndex = divmod(np.argmax(regret), regret.shape[1])
    case = cases[caseIndex]
    
    return maxRegret, case


#==============================================================================
#
#    collection of distance metrics that can be used in calculating regret
#
#==============================================================================

def L1(array):
    '''
    returns the L1, cityblock, length
    '''
    
    return np.sum(array, axis=1)

def L2(array):
    '''
    returns the euclidian length
    '''
    
    return np.sum(array*array, axis=1)

#==============================================================================
#
#    test code
#
#==============================================================================
 
if __name__ == '__main__':
    
    a = np.random.rand(20, 2)+1
    min = np.min(a, axis=0)
    max = np.max(a, axis=0)
    
    normalize_array(a, min, max)