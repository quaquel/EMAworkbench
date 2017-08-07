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

import itertools
import numbers
import os
import six
import warnings
import numpy as np

from .callbacks import DefaultCallback
from .ema_parallel import MultiprocessingPool
from .experiment_runner import ExperimentRunner
from .model import AbstractModel
from .parameters import Policy, Experiment
from .samplers import LHSSampler, sample_uncertainties, from_experiments
from .util import determine_objects, NamedObjectMap

from ..util import info, debug, EMAError
from ema_workbench.em_framework import samplers

# Created on 23 dec. 2010
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['ModelEnsemble']

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
        self._model_structures = NamedObjectMap(AbstractModel)
        self._policies = NamedObjectMap(Policy)
        self.sampler = sampler

    @property
    def model_structure(self):
        warnings.warn('deprecated, use ensemble.model_structures')
        return self.model_structures

    @model_structure.setter
    def model_structure(self, value):
        warnings.warn('deprecated, use ensemble.model_structures')
        self.model_structures = value

    @property
    def policies(self):
        return self._policies
   
    @policies.setter
    def policies(self, policies):
        self.policies.clear() 
        self.policies.extend(policies)
   
    @property
    def model_structures(self):
        return self._model_structures
    
    @model_structures.setter
    def model_structures(self, msis):
        self._model_structures.extend(msis)
    
    def perform_experiments(self, 
                           cases,
                           callback=DefaultCallback,
                           reporting_interval=None,
                           uncertainty_union=False,
                           outcome_union=False,
                           **kwargs):
        """
        Method responsible for running the experiments on a structure. In case 
        of multiple model structures, the outcomes are set to the intersection 
        of the sets of outcomes of the various models.     
        
        Parameters
        ----------    
        cases : int
                In case of Latin Hypercube sampling and Monte Carlo 
                sampling, cases specifies the number of cases to
                generate. In case of Full Factorial sampling,
                cases specifies the resolution to use for sampling
                continuous uncertainties. 
        callback : callback, optional
                   callable that will be called after finishing a 
                   single experiment (default is :class:`~callbacks.DefaultCallback`)
        reporting_interval : int, optional
                             parameter for specifying the frequency with
                             which the callback reports the progress.
                             If none is provided, it defaults to 1/10 of 
                             the total number of scenarios.
        uncertainty_union : bool, optional
                              keyword argument for controlling whether,
                              in case of multiple model structure 
                              interfaces, the intersection or the union
                              of uncertainties should be used. 
        outcome_union : bool, optional
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
        if not self.policies:
            self.policies = Policy('none')
            levers = []        
        else:
            levers = determine_objects(self.model_structures, 'levers', 
                                       union=True)
            attributes = {key for p in self.policies for key in p.keys()}
            levers = [lever for lever in levers if lever.name in attributes]

        outcomes = determine_objects(self.model_structures, 'outcomes', 
                                     union=outcome_union)

        if isinstance(cases, numbers.Integral):
            res = sample_uncertainties(self.model_structures, cases, 
                                       uncertainty_union, sampler=self.sampler)
            scenarios = res
            uncertainties = res.parameters
            nr_of_scenarios = res.n
        elif isinstance(cases, np.ndarray):
            res = from_experiments(self.model_structures, cases)
            scenarios = res
            uncertainties = res.parameters
            nr_of_scenarios = res.n
        else:
            scenarios = cases
            nr_of_scenarios = len(scenarios)
            uncertainties = samplers.determine_parameters(self.model_structures, 
                                                          'uncertainties')
            names = set()
            for case in cases:
                names = names.union(case.keys())
                
#             uncertainties = [u for u in uncertainties if u.name in names]
            
        
        experiments = experiment_generator(scenarios, self.model_structures, 
                                           self.policies)
        nr_of_exp = nr_of_scenarios * len(self.model_structures) * len(self.policies)
        
        info(str(nr_of_exp) + " experiment will be executed")

        if reporting_interval is None:
            reporting_interval = max(1, int(round(nr_of_exp / 10))) 

        #initialize the callback object
        callback = callback(uncertainties, 
                            levers,
                            outcomes, 
                            nr_of_exp,
                            reporting_interval=reporting_interval,
                            **kwargs)

        if self.parallel:
            info("preparing to perform experiment in parallel")
            
            if not self.pool:
                self.pool = MultiprocessingPool(self.model_structures,
                                                nr_processes=self.processes)
            info("starting to perform experiments in parallel")

            self.pool.perform_experiments(callback, experiments)
        else:
            info("starting to perform experiments sequentially")
            
            cwd = os.getcwd() 
            runner = ExperimentRunner(self.model_structures)
            for experiment in experiments:
                result = runner.run_experiment(experiment)
                callback(experiment, result)
            runner.cleanup()
            os.chdir(cwd)
        
        if callback.i != nr_of_exp:
            raise EMAError(('some fatal error has occurred while '
                            'running the experiments, not all runs have ' 
                            'completed. expected {} '.format(nr_of_exp),
                            'got {}'.format(callback.i),
                            '{}'.format(type(callback))))
       
        results = callback.get_results()
        info("experiments finished")
        
        return results


def experiment_generator(scenarios, model_structures, policies):
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
    jobs = itertools.product(model_structures, policies, scenarios)
    
    for i, job in enumerate(jobs):
        msi, policy, scenario = job
        name = '{} {} {}'.format(msi.name, policy.name, i)
        experiment = Experiment(name, msi, policy, scenario, i)
        yield experiment
