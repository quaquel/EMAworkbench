'''
Created on Feb 28, 2012

@author: jhkwakkel
'''
from expWorkbench import util
import numpy as np
SVN_ID = '$Id: optimization_callbacks.py 1027 2012-11-21 19:56:50Z jhkwakkel $'

class OptimizationCallback(util.AbstractCallback):    
    '''
    Base class from which different call back classes can be derived.
    Callback is responisble for storing the results of the runs.
    
    '''
    
    def __init__(self, 
                 uncertainties, 
                 outcomes, 
                 nrOfExperiments,
                 reporting_interval=100,
                 internalPopulation=None,
                 alleleOrder=None):
        '''
        
        :param uncertainties: list of :class:`~uncertianties.AbstractUncertainty` 
                              children
        :param outcomes: list of :class:`~outcomes.Outcome` instances
        :param nrOfExperiments: the total number of runs
        
        '''
        self.i = 0
        self.internalPopulation = internalPopulation
        self.alleleOrder = alleleOrder
        self.reporting_interval=reporting_interval
    
    def __call__(self, case, policy, name, result ):
        '''
        Method responsible for storing results. The implementation in this
        class only keeps track of how many runs have been completed and 
        logging this. 
        
        :param case: the case to be stored
        :param policy: the name of the policy being used
        :param name: the name of the model being used
        :param result: the result dict
        
        '''
        genome = self.internalPopulation[self.i]
        genome.genomeList = [case.get(key) for key in self.alleleOrder]
        score = genome.evaluator.funcList[0](result)
        genome.score = score
        
        super(OptimizationCallback, self).__call__(case, policy, name, result)
    
    def get_results(self):    
        return self.internalPopulation

class StatisticsCallback(object):
    
    i = -1
    
    def __init__(self, nrOfGenerations, nrOfPopMembers):
        self.stats = {}
        self.rawScore = np.empty((nrOfGenerations, nrOfPopMembers))
        self.fitnessScore = np.empty((nrOfGenerations, nrOfPopMembers))
        self.outcomes = None
        self.nrOfGenerations = nrOfGenerations
    
    def __store_stats(self, result):
        
        if not self.outcomes:
            self.outcomes = result.keys()
        
        for outcome in self.outcomes:
            try:
                self.stats[outcome][self.i, :] = result[outcome]
            except KeyError:
                ncol= 1
                self.stats[outcome] = np.empty((self.nrOfGenerations, ncol))
                self.stats[outcome][self.i, :] = result[outcome]

    def __store_individuals(self, internalPop):
        for j, entry in enumerate(internalPop):
            self.rawScore[self.i,j] = entry.score
            self.fitnessScore[self.i,j] = entry.fitness 

    def __call__(self, gaInstance):
        self.i+=1
        self.__store_stats(gaInstance.internalPop.getStatistics().internalDict)
        self.__store_individuals(gaInstance.internalPop)
        
        return False