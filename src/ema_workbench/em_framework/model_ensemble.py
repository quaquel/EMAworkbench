'''
This module specifies the ensemble class which keeps track of the models,
policies, and the running and storing of experiments. 

.. rubric:: an illustration of use

>>> model = UserSpecifiedModelInterface('./model/', 'name')
>>> ensemble = ModelEnsemble()
>>> ensemble.model_structure = model
>>> ensemble.parallel = True #parallel processing is turned on
>>> results = ensemble.perform_experiments(1000) #perform 1000 experiments

In this example, 1000 experiments will be carried out in parallel on 
the user specified model interface. The uncertainties are retrieved from 
model.uncertainties and the outcomes are assumed to be specified in
model.outcomes.

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
import six
from functools import reduce, partial
import os
import itertools
from collections import defaultdict

from ..util import info, debug, EMAError

from .ema_optimization import NSGA2
from .ema_optimization_util import (evaluate_population_outcome, 
                    generate_individual_outcome, generate_individual_robust, 
                    evaluate_population_robust)                                               

from .samplers import FullFactorialSampler, LHSSampler
from .uncertainties import ParameterUncertainty, CategoricalUncertainty
from .callbacks import DefaultCallback

from .experiment_runner import ExperimentRunner
from .ema_parallel import MultiprocessingPool

# Created on 23 dec. 2010
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['ModelEnsemble', 'MINIMIZE', 'MAXIMIZE', 'UNION', 
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
    
    Parameters
    ----------
    sampler: Sampler instance
             the sampler to be used for generating experiments. 
             (the default is  :class:`~samplers.LHSSampler`)
    

    The generation of designs is delegated to a sampler. See :mod:`samplers`
    for details. The storing of results is delegated to a callback. See 
    :mod:`callbacks` for details. 
    
    
    Attributes
    ----------
    parallel : bool
               whether to run in parallel or not. Default is false
    pool : MultiprocessingPool instance or IpyparallelPool instance
           the pool to delegate the running of experiments to in case
           of running in parallel. If parallel is true and pool is none
           a :class:`MultiprocessingPool` will be set up and used. 
    processes : int
                the number of processes to use when running in parallel. 
                Default is None, meaning that the number of processes is
                determined by the pool itself.     
    policies : list
               a list of the policies to be explored. By default this contains
               a single policy called None. The moment you assign new
               policies to this attribute, this none policy is automoatically
               removed. 
    model_structures  : list
                        a list with model structures over which to conduct
                        the experiments.
    
    '''
    
    parallel = False
    pool = None
    processes = None
    
    def __init__(self, sampler=LHSSampler()):
        super(ModelEnsemble, self).__init__()
        self._policies = []
        self._msis = {}
        self._policies = {'None': {'name':'None'}}
        self.sampler = sampler

    @property
    def policies(self):
        return self._policies.values()
   
    @policies.setter
    def policies(self, policies):
        try:
            self._policies = {policy['name']:policy for policy in policies}
        except TypeError:
            # it probably is a single policy
            self._policies = {policies['name']:policies}
   
    @property
    def model_structures(self):
        return self._msis.values()
    
    @model_structures.setter
    def model_structures(self, msis):
        self._msis = {msi.name:msi for msi in msis}
    
    @property
    def model_structure(self):
        return list(self.model_structures)[0]
    
    @model_structure.setter
    def model_structure(self, msi):
        self.model_structures = [msi]
    
    def determine_uncertainties(self):
        '''
        Helper method for determining the unique uncertainties and how
        the uncertainties are shared across multiple model structure 
        interfaces.
        
        Returns
        -------
        dict
            An overview dictionary which shows which uncertainties are
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
        
        Parameters
        ----------    
        cases : int or iterable
                In case of Latin Hypercube sampling and Monte Carlo 
                sampling, cases specifies the number of cases to
                generate. In case of Full Factorial sampling,
                cases specifies the resolution to use for sampling
                continuous uncertainties. Alternatively, one can supply
                a list of dicts, where each dicts contains a case.
                That is, an uncertainty name as key, and its value. 
        callback : callback, optional
                   callable that will be called after finishing a 
                   single experiment (default is :class:`~callbacks.DefaultCallback`)
        reporting_interval : int, optional
                             parameter for specifying the frequency with
                             which the callback reports the progress.
                             (Default is 100) 
        model_kwargs : dict, optional
                       dictionary of keyword arguments to be passed to 
                       model_init
        which_uncertainties : {INTERSECTION, UNION}, optional
                              keyword argument for controlling whether,
                              in case of multiple model structure 
                              interfaces, the intersection or the union
                              of uncertainties should be used. 
        which_outcomes : {INTERSECTION, UNION}, optional
                          keyword argument for controlling whether,
                          in case of multiple model structure 
                          interfaces, the intersection or the union
                          of outcomes should be used. 
        kwargs : dict, optional
                 generic keyword arguments to pass on to the callback

        Returns
        -------
        tuple 
            a `structured numpy array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_ 
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

        >>> import util as util
        >>> util.save_results(results, filename)

        """
        return_val = self._generate_experiments(cases, which_uncertainties)
        
        experiments, nr_of_exp, uncertainties = return_val
        # identify the outcomes that are to be included
        overview_dict, element_dict = self._determine_unique_attributes("outcomes")
        if which_outcomes==UNION:
            outcomes = element_dict.keys()
        elif which_outcomes==INTERSECTION:
            outcomes = overview_dict[tuple([msi.name for msi in 
                                            self.model_structures])]
            outcomes = [outcome.name for outcome in outcomes]
        else:
            raise ValueError("unknown value for which_outcomes")
         
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
                self.pool = MultiprocessingPool(self.model_structures, 
                        model_kwargs=model_kwargs, nr_processes=self.processes)
            info("starting to perform experiments in parallel")

            self.pool.perform_experiments(callback, experiments)
        else:
            info("starting to perform experiments sequentially")
            
            cwd = os.getcwd() 
            runner = ExperimentRunner(self._msis, model_kwargs)
            for experiment in experiments:
                experiment_id, case, policy, model_name, result = runner.run_experiment(experiment)
                callback(experiment_id, case, policy, model_name, result)
            runner.cleanup()
            os.chdir(cwd)
       
        results = callback.get_results()
        info("experiments finished")
        
        return results

    def perform_robust_optimization(self, 
                                    cases,
                                    obj_function=None,
                                    policy_levers={},
                                    weights = (),
                                    reporting_interval=100,
                                    algorithm=NSGA2,
                                    eval_pop=evaluate_population_robust,
                                    nr_of_generations=100,
                                    pop_size=100,
                                    crossover_rate=0.5, 
                                    mutation_rate=0.02,
                                    caching=False,
                                    **kwargs):
        """
        Method responsible for performing robust optimization.
        
        Parameters
        ----------
        cases : int or list
                In case of Latin Hypercube sampling and Monte Carlo sampling, 
                cases specifies the number of cases to generate. In case of 
                Full Factorial sampling, cases specifies the resolution to use 
                for sampling continuous uncertainties. Alternatively, one can 
                supply a list of dicts, where each dicts contains a case. That 
                is, an uncertainty name as key, and its value. 
        obj_function : callable
                       the objective function used by the optimization
        policy_levers : dict
                        A dictionary with model parameter names as key
                        and a dict as value. The dict should have two 
                        fields: 'type' and 'values. Type is either
                        list or range, and determines the appropriate
                        allele type. Values are the parameters to 
                        be used for the specific allele. 
        weights : tuple 
                  weights on the various outcomes of the objective function. 
                  Use the constants MINIMIZE and MAXIMIZE.
        reporting_interval : int , optional
                             parameter for specifying the frequency with which 
                             the callback reports the progress. 
                             (Default is 100) 
        algorithm : NSGA2 instance, optional
        eval_pop : callable, optional
                   function for evaluating a population, defaults
                   to :func:`evaluate_population_robust`
        nr_of_generations : int, optional
                            the number of generations for which the GA will be 
                            run
        pop_size : int, optional
                   the population size for the GA
        crossover_rate : float, optional
                         crossover rate for the GA
        mutation_rate : float, optional
                        mutation_rate for the GA
        caching: bool, optional
                 keep track of tried solutions. This is memory 
                 intensive, so should be used sparingly. Defaults to
                 False. 
        
        """
        evaluate_population = partial(eval_pop, cases=cases)

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
                                     obj_function=None,
                                     weights = (),
                                     algorithm=NSGA2,
                                     reporting_interval=100,
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
        
        Parameters
        ----------    
        obj_function : callable
                       the objective function used by the optimization
        weights : tuple
                  tuple of weights on the various outcomes of the objective 
                  function. Use the constants MINIMIZE and MAXIMIZE.
        reporting_interval : int, optional
                             parameter for specifying the frequency with
                             which the callback reports the progress.
                             (Default is 100) 
        nr_of_generations : int, optional
                            the number of generations for which the GA will be 
                            run
        pop_size : int, optional
                   the population size for the GA
        crossover_rate : float, optional
                         crossover rate for the GA
        mutation_rate : float, optional
                        mutation_rate for the GA
        caching : bool, optional
                  keep track of tried solutions. This is memory intensive, 
                  so should be used sparingly. Defaults to False. 
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
        
        Parameters
        ----------
        attribute : {'uncertainties', 'outcomes'}
                    the attribute to check on the msi
        
        Returns
        -------
        tuple of dicts
            An overview dictionary which shows which uncertainties or outcomes 
            are used by which model structure interface, or interfaces, and a 
            dictionary with the unique uncertainties or outcomes across all the 
            model structure interfaces, with the name as key. 
        
        '''    
        # check whether uncertainties exist with the same name 
        # but different other attributes
        element_dict = {}
        overview_dict = {}
        for msi in self.model_structures:
            elements = getattr(msi, attribute)
            for element in elements:
                if element.name in element_dict.keys():
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
        for key, value in overview_dict.items():
            temp_overview[tuple([msi.name for msi in value])].append(element_dict[key])  
        overview_dict = temp_overview
        
        return overview_dict, element_dict 
        
    def _generate_experiments(self, cases, which_uncertainties):
        '''
        Helper method for generating experiments
        
        Parameters
        ----------
        cases : int or list
        which_uncertianties : {INTERSECTION, UNION}

        Returns
        -------
        generator
            a generator that yields experiment dicts
        int
            the total number of experiments 
            so: nr_cases * nr of models * nr of policies)
        list
            list of the uncertainties over which the experiments are designed
        
        '''
        overview_dict, unc_dict = self.determine_uncertainties()
        # identify the uncertainties and sample over them
        if isinstance(cases, int):
            if which_uncertainties==UNION:
                if isinstance(self.sampler, FullFactorialSampler):
                    raise EMAError("full factorial sampling cannot be combined with exploring the union of uncertainties")
                uncertainties = unc_dict.values()
            elif which_uncertainties==INTERSECTION:
                uncertainties = overview_dict[tuple([msi.name for msi in 
                                                     self.model_structures])]
                unc_dict = {key.name:unc_dict[key.name] for key in uncertainties}
                uncertainties = [unc_dict[unc.name] for unc in uncertainties]
            else:
                raise ValueError("incompatible value for which_uncertainties")            

            designs, nr_of_designs = self.sampler.generate_designs(uncertainties, 
                                                   cases)
        elif isinstance(cases, list):
            unc_names = reduce(set.union, map(set, map(dict.keys, cases)))
            uncertainties = [unc_dict[unc] for unc in unc_names]
            designs = cases
            nr_of_designs = len(designs)
        else:
            raise EMAError("unknown type for cases")

        nr_of_exp = nr_of_designs*len(self.policies)*len(self.model_structures)
        experiments = experiment_generator(designs, self.model_structures,\
                                           self.policies)
        
        return experiments, nr_of_exp, uncertainties
        
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
           
        Parameters
        ----------     
        toolbox : 
        generate_individual : callable
                              helper function for generating an individual
        evaluate_population : callable
                              helper function for evaluating the population
        attr_list : list
                    list of attributes (alleles)
        keys : list
               the names of the attributes in the same order as attr_list
        obj_function : callable
                       the objective function
        pop_size : int
                   the size of the population
        reporting_interval : int
                             the interval for reporting progress, passed on to 
                             perform_experiments
        weights : tuple
                  the weights on the outcomes
        nr_of_generations : int
                            number of generations for which the GA will 
                            be run
        crossover_rate : float
                         the crossover rate of the GA
        mutation_rate : float
                        the mutation rate of the GA
        levers : dict
                 a dictionary with param keys as keys, and as values info used 
                 in mutation.
        
        '''
        self.algorithm = algorithm(weights, levers, generate_individual, 
                          obj_function, pop_size, evaluate_population, 
                          nr_of_generations, crossover_rate, mutation_rate, 
                          reporting_interval, self, caching, **kwargs)

        # Begin the generational process
        for _ in range(nr_of_generations):
            pop = self.algorithm.get_population()
        info("-- End of (successful) evolution --")

        return self.algorithm.stats_callback, pop        


def experiment_generator(designs, model_structures, policies):
    '''
    
    generator function which yields experiments
    
    Parameters
    ----------
    designs : iterable of dicts
    model_structures : list
    policies : list

    Notes
    -----
    this generator is essentially three nested loops: for each model structure,
    for each policy, for each experiment, run the experiment. This means 
    that designs should not be a generator because this will be exhausted after
    the running the first policy on the first model. 
    
    '''
    
    job_counter = itertools.count()
    
    for msi in model_structures:
        debug("generating designs for model %s" % (msi.name))
        msi_uncs = {unc.name for unc in msi.uncertainties}
        
        for policy in policies:
            debug("generating designs for policy %s" % (policy['name']))
            
            for design in designs:
                # from the design only get the uncertainties that 
                # are valid for the current msi
                keys = set(design.keys()).intersection(msi_uncs)
                experiment = {unc:design[unc] for unc in keys}
                
                # complete the design by adding the policy, model name
                # and experiment id to it
                experiment['policy'] = policy
                experiment['model'] = msi.name
                experiment['experiment id'] = six.next(job_counter)
                yield experiment