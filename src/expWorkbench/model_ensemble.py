'''

Created on 23 dec. 2010

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

'''
from __future__ import division

import types
import copy
import functools
import os
import itertools
from collections import defaultdict

from ema_parallel_multiprocessing import CalculatorPool

from expWorkbench.ema_logging import info, warning, exception, debug
from expWorkbench.ema_exceptions import CaseError, EMAError

from expWorkbench.ema_optimization import NSGA2
from expWorkbench.ema_optimization_util import evaluate_population_outcome,\
                                               generate_individual_outcome,\
                                               generate_individual_robust,\
                                               evaluate_population_robust                                               

from samplers import FullFactorialSampler, LHSSampler
from uncertainties import ParameterUncertainty, CategoricalUncertainty
from callbacks import DefaultCallback
from expWorkbench.ema_parallel import MultiprocessingPool

__all__ = ['ModelEnsemble', 'ExperimentRunner','MINIMIZE', 'MAXIMIZE', 'UNION', 
           'INTERSECTION']

MINIMIZE = -1.0
MAXIMIZE = 1.0

INTERSECTION = 'intersection'
UNION = 'union'

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
    
    #: boolean for turning parallel on (default is False)
    parallel = False
    
    pool = None
    
    def __init__(self, sampler=LHSSampler()):
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
        self._msis = {}
        self._policies = {'None': {'name':'None'}}
        self.sampler = sampler

    @property
    def policies(self):
        return self._policies.values()
   
    @policies.setter
    def policies(self, policies):
        self._policies = {policy['name'] for policy in policies}
   
    @property
    def model_structures(self):
        return self._msis.values()
    
    @model_structures.setter
    def model_structures(self, msis):
        self._msis = {msi.name:msi for msi in msis}
    
    @property
    def model_structure(self):
        return self.model_structures[0]
    
    @model_structure.setter
    def model_structure(self, msi):
        self.model_structures = [msi]
    
    def determine_uncertainties(self):
        '''
        Helper method for determining the unique uncertainties and how
        the uncertainties are shared across multiple model structure 
        interfaces.
        
        :returns: An overview dictionary which shows which uncertainties are
                  used by which model structure interface, or interfaces, and
                  a dictionary with the unique uncertainties across all the 
                  model structure interfaces, with the name as key. 
        
        '''
        return self._determine_unique_attributes('uncertainties')

    def perform_experiments(self, 
                           cases,
                           callback=DefaultCallback,
                           reporting_interval=100,
                           model_kwargs = {},
                           which_uncertainties=INTERSECTION,
                           which_outcomes=INTERSECTION,
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
        :param model_kwargs: dictionary of keyword arguments to be passed to 
                            model_init
        :param which_uncertainties: keyword argument for controlling whether,
                                    in case of multiple model structure 
                                    interfaces, the intersection or the union
                                    of uncertainties should be used. 
                                    (Default is intersection).  
        :param which_uncertainties: keyword argument for controlling whether,
                                    in case of multiple model structure 
                                    interfaces, the intersection or the union
                                    of outcomes should be used. 
                                    (Default is intersection).  
        :param kwargs: generic keyword arguments to pass on to callback
         
                       
        :returns: a `structured numpy array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_ 
                  containing the experiments, and a dict with the names of the 
                  outcomes as keys and an numpy array as value.
                
        .. rubric:: suggested use
        
        In general, analysis scripts require both the structured array of the 
        experiments and the dictionary of arrays containing the results. The 
        recommended use is the following::
        
        >>> results = ensemble.perform_experiments(10000) #recommended use
        >>> experiments, output = ensemble.perform_experiments(10000) 
        
        The latter option will work fine, but most analysis scripts require 
        to wrap it up into a tuple again::
        
        >>> data = (experiments, output)
        
        Another reason for the recommended use is that you can save this tuple
        directly::
        
        >>> import expWorkbench.util as util
        >>> util.save_results(results, filename)
          
        .. note:: The current implementation has a hard coded limit to the 
          number of designs possible. This is set to 50.000 designs. 
          If one want to go beyond this, set `self.max_designs` to
          a higher value.
        
        """
        
        # identify the uncertainties and sample over them
        if type(cases) ==  types.IntType:
            sampled_unc, unc_dict = self._generate_samples(cases, 
                                                           which_uncertainties)
            nr_of_exp =self.sampler.deterimine_nr_of_designs(sampled_unc)\
                      *len(self.policies)*len(self.model_structures)
            experiments = self._generate_experiments(sampled_unc)
        elif type(cases) == types.ListType:
            unc_dict = self.determine_uncertainties()[1]
            unc_names = cases[0].keys()
            sampled_unc = {name:[] for name in unc_names}
            nr_of_exp = len(cases)*len(self.policies)*len(self.model_structures)
            experiments = self._generate_experiments(cases)
        else:
            raise EMAError("unknown type for cases")
        uncertainties = [unc_dict[unc] for unc in sorted(sampled_unc)]

        # identify the outcomes that are to be included
        overview_dict, element_dict = self._determine_unique_attributes("outcomes")
        if which_outcomes==UNION:
            outcomes = element_dict.keys()
        elif which_outcomes==INTERSECTION:
            outcomes = overview_dict[tuple([msi.name for msi in self.model_structures])]
            outcomes = [outcome.name for outcome in outcomes]
        else:
            raise ValueError("incomplete value for which_outcomes")
         
        info(str(nr_of_exp) + " experiment will be executed")
                
        #initialize the callback object
        callback = callback(uncertainties, 
                            outcomes, 
                            nr_of_exp,
                            reporting_interval=reporting_interval,
                            **kwargs)

        if self.parallel:
            info("preparing to perform experiment in parallel")
            
            if not self.pool:
                self.pool = MultiprocessingPool(self.model_structures, kwargs)
            info("starting to perform experiments in parallel")

            self.pool.perform_experiments(callback, experiments)
        else:
            info("starting to perform experiments sequentially")
            
            cwd = os.getcwd() 
            runner = ExperimentRunner(self._msis, kwargs)
            for experiment in experiments:
                runner.run_experiment(experiment)
            os.chdir(cwd)
       
        results = callback.get_results()
        info("experiments finished")
        
        return results

    def perform_robust_optimization(self, 
                                    cases,
                                    reporting_interval=100,
                                    algorithm=NSGA2,
                                    eval_pop=evaluate_population_robust,
                                    obj_function=None,
                                    policy_levers={},
                                    weights = (),
                                    nr_of_generations=100,
                                    pop_size=100,
                                    crossover_rate=0.5, 
                                    mutation_rate=0.02,
                                    caching=False,
                                    **kwargs):
        """
        Method responsible for performing robust optimization.
        
        :param cases: In case of Latin Hypercube sampling and Monte Carlo 
                      sampling, cases specifies the number of cases to
                      generate. In case of Full Factorial sampling,
                      cases specifies the resolution to use for sampling
                      continuous uncertainties. Alternatively, one can supply
                      a list of dicts, where each dicts contains a case.
                      That is, an uncertainty name as key, and its value. 
        :param reporting_interval: parameter for specifying the frequency with
                                   which the callback reports the progress.
                                   (Default is 100) 
        
        :param algorithm:
        :param eval_pop:
        :param obj_function: the objective function used by the optimization
        :param policy_levers: A dictionary with model parameter names as key
                              and a dict as value. The dict should have two 
                              fields: 'type' and 'values. Type is either
                              list or range, and determines the appropriate
                              allele type. Values are the parameters to 
                              be used for the specific allele. 
        :param weights: tuple of weights on the various outcomes of the 
                        objective function. Use the constants MINIMIZE and 
                        MAXIMIZE.
        :param nr_of_generations: the number of generations for which the 
                                  GA will be run
        :param pop_size: the population size for the GA
        :param crossover_rate: crossover rate for the GA
        :param mutation_rate: mutation_rate for the GA
        :param caching: keep track of tried solutions. This is memory 
                intensive, so should be used sparingly. Defaults to
                False. 

        
        """
        evaluate_population = functools.partial(eval_pop, 
                                                cases=cases)

        return self._run_optimization(generate_individual_robust, 
                                      evaluate_population, 
                                      weights=weights,
                                      levers=policy_levers,
                                      algorithm=algorithm, 
                                      obj_function=obj_function, 
                                      pop_size=pop_size, 
                                      reporting_interval=reporting_interval, 
                                      nr_of_generations=nr_of_generations, 
                                      crossover_rate=crossover_rate, 
                                      mutation_rate=mutation_rate, 
                                      caching=caching,
                                      **kwargs)

    def perform_outcome_optimization(self, 
                                     algorithm=NSGA2,
                                     reporting_interval=100,
                                     obj_function=None,
                                     weights = (),
                                     nr_of_generations=100,
                                     pop_size=100,
                                     crossover_rate=0.5, 
                                     mutation_rate=0.02,
                                     caching=False,
                                     **kwargs
                                     ):
        """
        Method responsible for performing outcome optimization. The 
        optimization will be performed over the intersection of the 
        uncertainties in case of multiple model structures. 
        
        :param reporting_interval: parameter for specifying the frequency with
                                   which the callback reports the progress.
                                   (Default is 100) 
        :param obj_function: the objective function used by the optimization
        :param weights: tuple of weights on the various outcomes of the 
                        objective function. Use the constants MINIMIZE and 
                        MAXIMIZE.
        :param nr_of_generations: the number of generations for which the 
                                  GA will be run
        :param pop_size: the population size for the GA
        :param crossover_rate: crossover rate for the GA
        :param mutation_rate: mutation_rate for the GA
        :param caching: keep track of tried solutions. This is memory 
                        intensive, so should be used sparingly. Defaults to
                        False. 
       
        """

        # Attribute generator
        od = self._determine_unique_attributes('uncertainties')[0]
        shared_uncertainties = od[tuple([msi.name for msi in 
                                         self.model_structures])]

        #make a dictionary with the shared uncertainties and their range
        uncertainty_dict = {}
        for uncertainty in shared_uncertainties:
                uncertainty_dict[uncertainty.name] = uncertainty
        keys = sorted(uncertainty_dict.keys())
        
        levers = {}
        for key in keys:
            specification = {}
            uncertainty = uncertainty_dict[key]
            value = uncertainty.values
             
            if isinstance(uncertainty, CategoricalUncertainty):
                value = uncertainty.categories
                specification["type"]='list'
                specification['values']=value
            elif isinstance(uncertainty, ParameterUncertainty):
                if uncertainty.dist=='integer':
                    specification["type"]='range int'
                else:
                    specification["type"]='range float'
                specification['values']=value     
            else:
                raise EMAError("unknown allele type: possible types are range and list")
            levers[key] = specification

        return self._run_optimization(generate_individual_outcome,
                                      evaluate_population_outcome,
                                      weights=weights,
                                      levers=levers,
                                      algorithm=algorithm, 
                                      obj_function=obj_function, 
                                      pop_size=pop_size, 
                                      reporting_interval=reporting_interval, 
                                      nr_of_generations=nr_of_generations, 
                                      crossover_rate=crossover_rate, 
                                      mutation_rate=mutation_rate, **kwargs)


    
    def _determine_unique_attributes(self, attribute):
        '''
        Helper method for determining the unique values on attributes of model 
        interfaces, and how these values are shared across multiple model 
        structure interfaces. The working assumption is that this function 
        
        :param attribute: the attribute to check on the msi
        :returns: An overview dictionary which shows which uncertainties are
                  used by which model structure interface, or interfaces, and
                  a dictionary with the unique uncertainties across all the 
                  model structure interfaces, with the name as key. 
        
        '''    
        # check whether uncertainties exist with the same name 
        # but different other attributes
        element_dict = {}
        overview_dict = {}
        for msi in self.model_structures:
            elements = getattr(msi, attribute)
            for element in elements:
                if element_dict.has_key(element.name):
                    if element==element_dict[element.name]:
                        overview_dict[element.name].append(msi)
                    else:
                        raise EMAError("%s `%s` is shared but has different state" 
                                       % (element.__class__.__name__, 
                                          element.name))
                else:
                    element_dict[element.name]= element
                    overview_dict[element.name] = [msi]
        
        temp_overview = defaultdict(list)
        for key, value in overview_dict.iteritems():
            temp_overview[tuple([msi.name for msi in value])].append(element_dict[key])  
        overview_dict = temp_overview
        
        return overview_dict, element_dict 
     
    def _determine_outcomes(self):
        '''
        Helper method for determining the unique outcomes and how
        the outcomes are shared across multiple model structure 
        interfaces.
        
        :returns: An overview dictionary which shows which uncertainties are
                  used by which model structure interface, or interfaces, and
                  a dictionary with the unique uncertainties across all the 
                  model structure interfaces, with the name as key. 
        
        '''    
        return self._determine_unique_attributes('outcomes')
        
    
    def _generate_samples(self, nr_of_samples, which_uncertainties):
        '''
        number of cases specifies the number of cases to generate in case
        of Monte Carlo and Latin Hypercube sampling.
        
        In case of full factorial sampling nr_of_cases specifies the resolution 
        on non categorical uncertainties.
        
        In case of multiple model structures, the uncertainties over
        which to explore is the intersection of the sets of uncertainties of
        the model interface instances.
        
        :param nr_of_samples: The number of samples to generate for the 
                              uncertainties. In case of mc and lhs sampling,
                              the number of samples is generated. In case of 
                              ff sampling, the number of samples is interpreted
                              as the upper limit for the resolution to use.
        :returns: a dict with the samples uncertainties and a dict
                  with the unique uncertainties. The first dict containsall the 
                  unique uncertainties. The dict has the uncertainty name as 
                  key and the values are the generated samples. The second dict
                  has the name as key and an instance of the associated 
                  uncertainty as value.
        
        '''
        overview_dict, unc_dict = self.determine_uncertainties()
        
        if which_uncertainties==UNION:
            if isinstance(self.sampler, FullFactorialSampler):
                raise EMAError("full factorial sampling cannot be combined with exploring the union of uncertainties")
            
            sampled_unc = self.sampler.generate_samples(unc_dict.values(), 
                                                   nr_of_samples)
        elif which_uncertainties==INTERSECTION:
            uncertainties = overview_dict[tuple([msi.name for msi in self.model_structures])]
            unc_dict = {key.name:unc_dict[key.name] for key in uncertainties}
            uncertainties = [unc_dict[unc.name] for unc in uncertainties]
            sampled_unc = self.sampler.generate_samples(unc_dict.values(), 
                                                   nr_of_samples)
        else:
            raise ValueError("incompatible value for which_uncertainties")
        
        return sampled_unc, unc_dict
    
    def _generate_experiments(self, sampled_unc):
        '''
        Helper method for turning the sampled uncertainties into actual
        complete experiments, including the model structure interface and the 
        policy. The actual generation is delegated to the experiments_generator 
        function, which returns a generator. In this way, not all the 
        experiments are kept in memory, but they are generated only when 
        needed.
        
        :param sampled_unc: the sampled uncertainty dictionary
        :returns: a generator object that yields the experiments
        
        '''
        if isinstance(sampled_unc, list):
            return experiment_generator_predef_cases(copy.deepcopy(sampled_unc),\
                                                     self.model_structures,\
                                                     self.policies,)
        
        return experiment_generator(sampled_unc, self.model_structures,\
                                   self.policies, self.sampler)




    def _run_optimization(self, generate_individual, 
                           evaluate_population,algorithm=None, 
                           obj_function=None,
                           weights=None, levers=None, 
                           pop_size=None, reporting_interval=None, 
                           nr_of_generations=None, crossover_rate=None, 
                           mutation_rate=None,
                           caching=False,
                           **kwargs):
        '''
        Helper function that runs the actual optimization
                
        :param toolbox: 
        :param generate_individual: helper function for generating an 
                                    individual
        :param evaluate_population: helper function for evaluating the 
                                    population
        :param attr_list: list of attributes (alleles)
        :param keys: the names of the attributes in the same order as attr_list
        :param obj_function: the objective function
        :param pop_size: the size of the population
        :param reporting_interval: the interval for reporting progress, passed
                                   on to perform_experiments
        :param weights: the weights on the outcomes
        :param nr_of_generations: number of generations for which the GA will 
                                  be run
        :param crossover_rate: the crossover rate of the GA
        :param mutation_rate: the muation rate of the GA
        :param levers: a dictionary with param keys as keys, and as values
                       info used in mutation.
        
        '''
        self.algorithm = algorithm(weights, levers, generate_individual, obj_function, 
                          pop_size, evaluate_population, nr_of_generations, 
                          crossover_rate, mutation_rate, reporting_interval,
                          self, caching, **kwargs)

        # Begin the generational process
        for _ in range(nr_of_generations):
            pop = self.algorithm.get_population()
        info("-- End of (successful) evolution --")

        return self.algorithm.stats_callback, pop        


class ExperimentRunner(object):
    '''Helper class for running the experiments'''
    
    def __init__ (self, msis, model_kwargs):
        self.msi_initialization = {}
        self.msis = msis
        self.model_kwargs = model_kwargs
    
    def cleanup(self):
        for msi in self.msis.values():
            msi.cleanup()
        self.msis = None
    
    def run_experiment(self, experiment):
        '''The logic for running a single model. This code makes
        sure that the model is initialized properly if this has not
        already been done.
        
        Parameters
        ----------
        experiment : dict
        
        Returns
        -------
        experiment_id: int
        case : dict
        policy : str
        model_name : str
        result :  dict
        
        Raises
        ------
        EMAError
            if the model instance raises an EMA error, these are reraised.
        Exception
            Catch all for all other exceptions being raised by the model. 
            These are reraised.
        
        '''
        
        policy = experiment.pop('policy')
        model_name = experiment.pop('model')
        experiment_id = experiment.pop('experiment id')
        policy_name = policy['name']
        
        debug("running policy {} for experiment {}".format(policy_name, 
                                                           experiment_id))
        
        # check whether we already initialized the model for this 
        # policy
        if not self.msi_initialization.has_key((policy_name, model_name)):
            try:
                debug("invoking model init")
                self.msis[model_name].model_init(copy.deepcopy(policy), 
                                     copy.deepcopy(self.model_kwargs))
            except EMAError as inst:
                exception(inst)
                self.cleanup()
                raise inst
            except Exception as inst:
                exception("some exception occurred when invoking the init")
                self.cleanup()
                raise inst
                
            debug("initialized model %s with policy %s" % (model_name, policy_name))

            self.msi_initialization = {(policy_name, model_name):self.msis[model_name]}
        msi = self.msis[model_name]

        case = copy.deepcopy(experiment)
        try:
            debug("trying to run model")
            msi.run_model(case)
        except CaseError as e:
            warning(str(e))
            
        debug("trying to retrieve output")
        result = msi.retrieve_output()
        
        debug("trying to reset model")
        msi.reset_model()
        return experiment_id, case, policy, model_name, result        
        

def experiment_generator_predef_cases(designs, model_structures, policies):
    '''
    
    generator function which yields experiments
    
    '''
    
    # experiment is made up of case, policy, and msi
    # to get case, we need msi
    job_counter = itertools.count()
    
    for msi in model_structures:
        debug("generating designs for model %s" % (msi.name))

        for policy in policies:
            debug("generating designs for policy %s" % (policy['name']))
            for experiment in designs:
                experiment['policy'] = copy.deepcopy(policy)
                experiment['model'] = msi.name
                experiment['experiment id'] = job_counter.next()
                yield experiment
    
def experiment_generator(sampled_unc, model_structures, policies, sampler):
    '''
    
    generator function which yields experiments
    
    '''
    
    # experiment is made up of case, policy, and msi
    # to get case, we need msi
    job_counter = itertools.count()
    
    for msi in model_structures:
        debug("generating designs for model %s" % (msi.name))
        
        samples = [sampled_unc[unc.name] for unc in msi.uncertainties if\
                   sampled_unc.has_key(unc.name)]
        uncertainties = [unc.name for unc in msi.uncertainties if\
                         sampled_unc.has_key(unc.name)]
        for policy in policies:
            debug("generating designs for policy %s" % (policy['name']))
            designs = sampler.generate_designs(samples)
            for design in designs:
                experiment = {uncertainties[i]: design[i] for i in\
                                range(len(uncertainties))}
                experiment['policy'] = policy
                experiment['model'] = msi.name
                experiment['experiment id'] = job_counter.next()
                yield experiment