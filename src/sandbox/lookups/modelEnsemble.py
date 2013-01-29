'''
Created on 23 dec. 2010

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import division
import types
import copy

#from pyevolve import G1DList, GAllele, Mutators, Initializators, Consts #@UnresolvedImport
#from pyevolve import Crossovers #@UnresolvedImport

import expWorkbench.samplers as samplers
import expWorkbench.util as util

from expWorkbench.EMAparallel import CalculatorPool
from expWorkbench.EMAlogging import info, warning, exception
from expWorkbench.EMAexceptions import CaseError, EMAError, EMAParallelError
from expWorkbench.uncertainties import INTEGER , CategoricalUncertainty

from expWorkbench.optimizationCallbacks import StatisticsCallback
from expWorkbench.EMAoptimization import RobustOptimizationPopulation, BaseEMAPopulation,\
    EMAGA, OutcomeOptimizationPopulation, MaximinOptimizationPopulation
from expWorkbench import EMAoptimization, EMAlogging


SVN_ID = '$Id: modelEnsemble.py 889 2012-09-11 19:32:32Z jhkwakkel $'
__all__ = ['ModelEnsemble']

class ModelEnsemble(object):
    '''
    One of the two main classes for performing EMA. The ensemble class is 
    responsible for running experiments on one or more model structures across
    one or more policies, and returning the results. 
    
    The sampling is delegated to a sampler instance.
    The storing or results is delegated to a callback instance
    
    the class has an attribute 'parallel' that specifies whether the 
    experiments are to be run in parallel or not. By default, 'parallel' is 
    False.
    
    .. rubric:: an illustration of use
    
    >>> model = UserSpecifiedModelInterface(r'working directory', 'name')
    >>> ensemble = SimpleModelEnsemble()
    >>> ensemble.set_model_structure(model)
    >>> ensemble.parallel = True #parallel processing is turned on
    >>> results = ensemble.perform_experiments(1000) #perform 1000 experiments
    
    In this example, a 1000 experiments will be carried out in parallel on 
    the user specified model interface. The uncertainties are retrieved from 
    model.uncertainties and the outcomes are assumed to be specified in
    model.outcomes.
    
    '''
    
    #: In case of parallel computing, the number of 
    #: processes to be spawned. Default is None, meaning
    #: that the number of processes will be equal to the
    #: number of available cores.
    processes=None
    
    #: boolean for turning parallel on (default is False)
    parallel = False
    
    _pool = None
    
    _policies = []
    
    def __init__(self, sampler=samplers.LHSSampler()):
        """
        Class responsible for running experiments on diverse model 
        structures and storing the results.

        :param sampler: the sampler to be used for generating experiments. 
                        By default, the sampling technique is 
                        :class:`~samplers.LHSSampler`.  
        """
        super(ModelEnsemble, self).__init__()
        self.output = {}
        self._policies = []
        self._modelStructures = []
        self.sampler = sampler

    def add_policy(self, policy):
        """
        Add a policy. 
        
        :param policy: policy to be added, policy should be a dict with at 
                       least a name.
        
        """
        self._policies.append(policy)
        
    def add_policies(self, policies):
        """
        Add policies, policies should be a collection of policies.
        
        :param policies: policies to be added, every policy should be a 
                         dict with at  least a name.
        
        """
        [self._policies.append(policy) for policy in policies]
 
    def set_model_structure(self, modelStructure):
        '''
        Set the model structure. This function wraps the model structure
        in a tuple, limiting the number of model structures to 1.
        
        :param modelStructure: a :class:`~model.ModelStructureInterface` 
                               instance.
        
        '''
        
        self._modelStructures = tuple([modelStructure])
                     
    def add_model_structure(self, ms):
        '''
        Add a model structure to the list of model structures.
        
        :param ms: a :class:`~model.ModelStructureInterface` instance.
        
        '''
        
        self._modelStructures.append(ms)   
    
    def add_model_structures(self, mss):
        '''
        add a collection of model structures to the list of model structures.
        
        :param mss: a collection of :class:`~model.ModelStructureInterface` 
                    instances
        
        '''
        
        [self._modelStructures.append(ms) for ms in mss]  
    
    def determine_intersecting_uncertainties(self):
        #get the intersection of the uncertainties of the different models
        if len(self._modelStructures)  >1:
            # this seems opaque... but the reason for doing it this way is
            # that the name alone is not enough for identity. The 
            # ranges of the uncertainties should also be the same, hence
            # the identity function on the uncertainty. 
            
            uncertainties = []
            for msi in self._modelStructures:
                u = [uncertainty.identity() for uncertainty in msi.uncertainties]
                uncertainties.append(u)
            shared_uncertainties = set(uncertainties[0]).intersection(*uncertainties[1:])
            
            # determine unshared
            unshared = {}
            for i, msi in enumerate(self._modelStructures):
                un = set(uncertainties[i]) - set(shared_uncertainties)
                a = {}
                for u in msi.uncertainties:
                    a[u.name] = u
                u = [a.get(u[0]) for u in un]
                unshared[msi.name] = u 
            
            a = {}
            for u in self._modelStructures[0].uncertainties:
                a[u.name] = u
            shared_uncertainties = [a.get(u[0]) for u in shared_uncertainties]
            info("intersection contains %s uncertainties" %len(shared_uncertainties))
        else:
            shared_uncertainties = set(self._modelStructures[0].uncertainties)
            unshared = None
        
        return shared_uncertainties, unshared   
    
    def _generate_cases(self, nrOfCases):
        '''
        number of cases specifies the number of cases to generate in case
        of Monte Carlo and Latin Hypercube sampling.
        
        In case of full factorial sampling it specifies the resolution on
        non categorical uncertainties.
        
        In case of multiple model structures, the uncertainties over
        which to explore is the intersection of the sets of uncertainties of
        the model interface instances.
        
        :param nrOfCases: In case of Latin Hypercube sampling and Monte Carlo 
                          sampling, nrOfCases specifies the number of cases to
                          generate. In case of Full Factorial sampling,
                          nrOfCases specifies the resolution to use for sampling
                          continuous uncertainties.
        
        '''
        shared_uncertainties, unshared = self.determine_intersecting_uncertainties()
         
        info("generating cases")
        shared_designs = self.sampler.generate_design(shared_uncertainties, nrOfCases)
        information = shared_designs[1]
        shared_designs = shared_designs[0]
        cases = []
        for design in shared_designs:
            case = {}
            for i, name in enumerate(information):
                case[name] = design[i]
            cases.append(case)
        
        info(str(len(cases)) + " cases generated")
        
        return cases, shared_uncertainties
    
    def __make_pool(self, modelKwargs):
        self._pool = CalculatorPool(self._modelStructures, 
                                    processes=self.processes,
                                    kwargs=modelKwargs)

    def perform_experiments(self, 
                           cases,
                           callback = util.DefaultCallback,
                           reporting_interval=100,
                           modelKwargs = {},
                           **kwargs):
        """
        Method responsible for running the experiments on a structure. In case 
        of multiple model structures, the outcomes are set to the intersection 
        of the sets of outcomes of the various models.         
        
        :param cases: In case of Latin Hypercube sampling and Monte Carlo 
                      sampling, cases specifies the number of cases to
                      generate. In case of Full Factorial sampling,
                      cases specifies the resolution to use for sampling
                      continuous uncertainties. Alternatively, one can supply
                      a list of dicts, where each dicts contains a case.
                      That is, an uncertainty name as key, and its value. 
        :param callback: Class that will be called after finishing a 
                         single experiment,
        :param reporting_interval: parameter for specifying the frequency with
                                   which the callback reports the progress.
                                   (Default is 100) 
        :param modelKwargs: dictonary of keyword arguments to be passed to 
                            model_init
        :param kwargs: generic keyword arguments to pass on to callback
         
                       
        :returns: a `structured numpy array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_ 
                  containing the experiments, and a dict with the names of the 
                  outcomes as keys and an numpy array as value.
                
        .. rubric:: suggested use
        
        In general, analysis scripts require both the structured array of the 
        experiments and the dictionary of arrays containing the results. The 
        recommended use is the following::
        
        >>> results = ensemble.perform_experiments(10000) #recommended use
        >>> experiments, output = ensemble.perform_experiments(10000) #will work fine
        
        The latter option will work fine, but most analysis scripts require 
        to wrap it up into a tuple again::
        
        >>> data = (experiments, output)
        
        Another reason for the recommended use is that you can save this tuple
        directly::
        
        >>> import expWorkbench.util as util
        >>> util.save_results(results, file)
          
        
        
        """
        
        if type(cases) ==  types.IntType:
            cases, uncertainties = self._generate_cases(cases)
        elif type(cases) == types.ListType:
            uncertainties = self.determine_intersecting_uncertainties()[0]
            uncertaintyNames = cases[0].keys()
            uncertainties = [uncertainty for uncertainty in uncertainties if 
                             uncertainty.name in uncertaintyNames]
        else:
            raise EMAError("unknown type for cases")
        
        if not self._policies:
            self._policies.append({"name": "None"})

        nrOfExperiments =len(cases)*len(self._policies)*len(self._modelStructures) 
        info(str(nrOfExperiments) + 
             " experiment will be executed")
        
        #set outcomes to the intersect of outcomes across models
        outcomes = [msi.outcomes for msi in self._modelStructures]
        outcomes = set(outcomes[0]).intersection(*outcomes[:1])
        for msi in self._modelStructures:
            msi.outcomes = list(outcomes)
        if not outcomes:
            raise EMAError("no outcomes of interest defined")
                
        #initialize the callback object
        callback = callback(uncertainties, 
                            outcomes, 
                            nrOfExperiments,
                            reporting_interval=reporting_interval,
                            **kwargs)
                
        if self.parallel:
            info("preparing to perform experiment in parallel")
            
            if not self._pool:
                self.__make_pool(modelKwargs)
            info("starting to perform experiments in parallel")

            results = self._pool.runExperiments(cases, self._policies)
            
            for entry in results:
                try:
                    callback(*entry.get())
                except EMAParallelError as e:
                    exception(e)
                except Exception as e:
                    raise
        else:
            info("starting to perform experiments sequentially")

            def cleanup(modelInterfaces):
                for msi in modelInterfaces:
                    msi.cleanup()
                    del msi

            for policy in self._policies:
                for msi in self._modelStructures:
                    policyToRun = copy.deepcopy(policy)
                    try:
                        msi.model_init(policyToRun, modelKwargs)
                    except (EMAError, NotImplementedError) as inst:
                        exception(inst)
                        cleanup(self._modelStructures)
                        raise
    
                    for case in cases:
                        caseToRun = copy.deepcopy(case)
                        try:
                            msi.run_model(caseToRun)
                        except CaseError as e:
                            warning(str(e))
                        result = msi.retrieve_output()
                        msi.reset_model()
                        callback(case, policy, msi.name, 
                                 result
                                 )
            cleanup(self._modelStructures)
        
        results = callback.get_results()
        info("experiments finished")
        
        return results
    
#    def __optimize(self, 
#                  allele_order,
#                  setOfAlleles, 
#                  obj_function,
#                  nrOfGenerations,
#                  nrOfPopMembers,
#                  minimax,
#                  crossoverRate,
#                  mutationRate,
#                  elitism,
#                  reporting_interval,
#                  population=BaseEMAPopulation):
#        # make a genome with a length equal to the list of alleles
#        genome = G1DList.G1DList(len(setOfAlleles))
#        genome.setParams(allele=setOfAlleles)
#        
#        # The evaluator function (objective function)
#        # to be decided what to use as test function. In principle
#        # the test function is a function that transforms the genome
#        # to a case, runs the model, and returns the results
#        # ideally, we might remove that step entirely by not
#        # using ind.evaluate(**args) in the population...
#        genome.evaluator.set(obj_function)
#        genome.crossover.set(Crossovers.G1DListCrossoverSinglePoint)
#        genome.mutator.set(Mutators.G1DListMutatorAllele)
#        genome.initializator.set(Initializators.G1DListInitializatorAllele)
#        
#        stats = StatisticsCallback(nrOfGenerations, nrOfPopMembers)
#        ga = EMAGA(genome, population)
#        ga.internalPop = population(genome, allele_order, self, reporting_interval)
#        ga.setMinimax(Consts.minimaxType[minimax])
#        ga.stepCallback.set(stats)
#        ga.selector.set(EMAoptimization.EMARankSelector)
#        
#        if elitism:
#            ga.setElitism(True)
#            ga.setElitismReplacement(elitism)
#        
#        # a generation contains nrOfPopMembers individuals
#        ga.setPopulationSize(nrOfPopMembers)
#        
#        # there are nrOfGeneration generations
#        ga.setGenerations(nrOfGenerations)
#        
#        # crossover and mutation    
#        ga.setCrossoverRate(crossoverRate)
#        ga.setMutationRate(mutationRate)
#
#        # perform optimization, print every 10 generations
#        # ideally, we intercept these messages and redirect them to
#        # EMAlogging.
#        EMAlogging.info("starting optimization")
#        ga.evolve()
#        
#        # return results for best fit
#        best_individual = ga.bestIndividual()
#        
#        best_case = {}
#        for i, key in enumerate(allele_order):
#            best_case[key] = best_individual.genomeList[i]
#        
#        c = ""
#        for key, value in best_case.items():
#            c += key
#            c += " : "
#            c += str(value)
#            c += '\n'
#        
#        info('best case:\n' + c )
#        info('raw score: ' + str(best_individual.score))
#        
#        results = {"best individual score": best_individual.score,
#                   "best individual ": best_individual,
#                   "stats": stats.stats,
#                   "raw": stats.rawScore,
#                   "fitness": stats.fitnessScore,
#                   "mutation ration": mutationRate,
#                   "crossover rate": crossoverRate,
#                   "minimax": minimax,
#                   "time elapsed": ga.get_time_elapsed()}
#        
#        return results    
#    
##    def perform_outcome_optimization(self, 
##                                     reporting_interval=100,
##                                     obj_function=None,
##                                     minimax = "maximize",
##                                     nrOfGenerations = 100,
##                                     nrOfPopMembers=100,
##                                     crossoverRate = 0.5, 
##                                     mutationRate = 0.02,
##                                     elitism = 0
##                                     ):
##        """
##        Method responsible for performing the optimization.
##        
##        :param reporting_interval: Parameter for specifying the frequency with
##                           which the callback reports the progress.
##                           (Default = 100) 
##        :param obj_function: The objective function to use. This objective 
##                             function receives the results for a single model
##                             run for all the specified outcomes of interest and
##                             should return a single score which should be 
##                             positive. 
##        :param minimax: String indicating whether to minimize or maximize the
##                        obj_function.
##        :param nrOfGenerations: The number of generations to evolve over.
##        :param nrOfPopulationMembers: The number of population members in a 
##                                      single generation.
##        :param crossoverRate: The crossover rate, between 0.0 and 1.0. 
##                              see `wikipedia <http://en.wikipedia.org/wiki/Crossover_%28genetic_algorithm%29>`__
##                              for details. (Default = 0.5)
##        :param mutationRate: The mutation rate, between 0.0 and 1.0.
##                             see `wikipedia <http://en.wikipedia.org/wiki/Mutation_%28genetic_algorithm%29>`__
##                             for details. (Default = 0.02)
##        :param elitism: The number of best individuals to copy to the next 
##                        generation. (Default = 0)
##        
##        :returns: A dict with info on the optimization including stats, best
##                  individual, and information on the optimization setup
##        
##        """
##
##        # Genome instance
##        setOfAlleles = GAllele.GAlleles()
##
##        allele_order = []
##        # deduce the alleles from the overlapping set of model structure 
##        # uncertainties
##        # the alleles should use the limits of uncertainty, and their dType
##        # in case of categorical uncertainties, the transform to the 
##        # category is delegated to a later stage (to be decided)
##        shared_uncertainties = self.determine_intersecting_uncertainties()[0]
##        for uncertainty in shared_uncertainties:
##            values = uncertainty.get_values()
##            dist = uncertainty.dist
##
##            if isinstance(uncertainty, CategoricalUncertainty):
##                allele = GAllele.GAlleleList(uncertainty.categories)
##            elif dist== INTEGER:
##                allele = GAllele.GAlleleRange(values[0], values[1])
##            else:
##                allele = GAllele.GAlleleRange(values[0], values[1], real=True)
##            
##            setOfAlleles.add(allele)
##            allele_order.append(uncertainty.name)
##        return self.__optimize(allele_order, 
##                               setOfAlleles, obj_function, 
##                              nrOfGenerations, nrOfPopMembers, minimax, 
##                              crossoverRate, mutationRate, elitism,
##                              reporting_interval,
##                              population=OutcomeOptimizationPopulation)
##
##
##    def perform_robust_optimization(self, 
##                                    cases,
##                                    reporting_interval=100,
##                                    obj_function=None,
##                                    policy_levers={},
##                                    minimax="maximize",
##                                    nrOfGenerations=100,
##                                    nrOfPopMembers=100,
##                                    crossoverRate=0.5, 
##                                    mutationRate=0.02,
##                                    elitism=0
##                                    ):
##        """
##        Method responsible for performing robust optimization.
##        
##        :param cases: In case of Latin Hypercube sampling and Monte Carlo 
##                      sampling, cases specifies the number of cases to
##                      generate. In case of Full Factorial sampling,
##                      cases specifies the resolution to use for sampling
##                      continuous uncertainties. Alternatively, one can supply
##                      a list of dicts, where each dicts contains a case.
##                      That is, an uncertainty name as key, and its value.
##        :param reporting_interval: Parameter for specifying the frequency with
##                                   which the callback reports the progress.
##                                   (Default = 100)         
##        :param obj_function: The objective function to use. This objective 
##                             function receives the results for a policy and
##                             the provided cases for all the specified outcomes 
##                             of interest and should return a single score which 
##                             should be positive. 
##        :param policy_levers: A dictionary with model parameter names as key
##                              and a dict as value. The dict should have two 
##                              fields: 'type' and 'values. Type is either
##                              list or range, and determines the appropriate
##                              allele type. Values are the parameters to 
##                              be used for the specific allele. 
##        :param minimax: String indicating whether to minimize or maximize the
##                        obj_function.
##        :param nrOfGenerations: The number of generations to evolve over.
##        :param nrOfPopulationMembers: The number of population members in a 
##                                      single generation.
##        :param crossoverRate: The crossover rate, between 0.0 and 1.0. 
##                              see `wikipedia <http://en.wikipedia.org/wiki/Crossover_%28genetic_algorithm%29>`__
##                              for details. (Default = 0.5)
##        :param mutationRate: The mutation rate, between 0.0 and 1.0.
##                             see `wikipedia <http://en.wikipedia.org/wiki/Mutation_%28genetic_algorithm%29>`__
##                             for details. (Default = 0.02)
##        :param elitism: The number of best individuals to copy to the next 
##                        generation. (Default = 0) 
##        
##        :returns: A dict with info on the optimization including stats, best
##                  individual, and information on the optimization setup
##        
##        """
##
##        # Genome instance
##        setOfAlleles = GAllele.GAlleles()
##        allele_order = []
##        for key, value in policy_levers.items():
##            type_allele = value['type'] 
##            value = value['values']
##            if type_allele=='range':
##                allele = GAllele.GAlleleRange(value[0], value[1], real=True)
##            elif type_allele=='list':
##                allele = GAllele.GAlleleList(value)
##            else:
##                raise EMAError("unknown allele type: possible types are range and list")
##            
##            setOfAlleles.add(allele)
##            allele_order.append(key)
##        
##        RobustOptimizationPopulation.cases = cases
##        return self.__optimize(allele_order, 
##                               setOfAlleles, 
##                               obj_function, 
##                               nrOfGenerations, 
##                               nrOfPopMembers, 
##                               minimax, 
##                               crossoverRate, 
##                               mutationRate,
##                               elitism,
##                               reporting_interval, 
##                               population=RobustOptimizationPopulation)
##    
##    def perform_maximin_optimization(self, 
##                                    reporting_interval=100,
##                                    obj_function1=None,
##                                    policy_levers={},
##                                    minimax1 = "minimize",
##                                    minimax2 = "maximize",                                   
##                                    nrOfGenerations1 = 100,
##                                    nrOfPopMembers1 = 100,
##                                    crossoverRate1 = 0.5, 
##                                    mutationRate1 = 0.02,
##                                    elitism1 = 0,
##                                    nrOfGenerations2 = 100,
##                                    nrOfPopMembers2 = 100,
##                                    crossoverRate2 = 0.5, 
##                                    mutationRate2 = 0.02,
##                                    elitism2 = 0
##                                    ):
##        
##        # Genome instance
##        setOfAlleles = GAllele.GAlleles()
##        allele_order = []
##        for key, value in policy_levers.items():
##            allele = GAllele.GAlleleRange(value[0], value[1], real=True)
##            
##            setOfAlleles.add(allele)
##            allele_order.append(key)
##        
##        MaximinOptimizationPopulation.optimizationType = minimax2
##        MaximinOptimizationPopulation.nrOfGenerations = nrOfGenerations2
##        MaximinOptimizationPopulation.nrOfPopMembers = nrOfPopMembers2
##        MaximinOptimizationPopulation.crossoverRate = crossoverRate2
##        MaximinOptimizationPopulation.mutationRate = mutationRate2
##        MaximinOptimizationPopulation.elitism = elitism2
##        
##        return self.__optimize(allele_order, 
##                               setOfAlleles, 
##                               obj_function1, 
##                               nrOfGenerations1, 
##                               nrOfPopMembers1, 
##                               minimax1, 
##                               crossoverRate1, 
##                               mutationRate1,
##                               elitism1,
##                               reporting_interval, 
##                               population=MaximinOptimizationPopulation)