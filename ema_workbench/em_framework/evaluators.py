'''


'''
from __future__ import (unicode_literals, print_function, absolute_import,
                                        division)

import multiprocessing
import numbers 
import os
import random
import shutil
import string
import threading
import warnings

warnings.simplefilter("once", ImportWarning)

from .callbacks import DefaultCallback
from .ema_multiprocessing import LogQueueReader, initializer, add_tasks
from .ema_ipyparallel import (start_logwatcher, set_engine_logger, 
                              initialize_engines, cleanup, _run_experiment)
from .experiment_runner import ExperimentRunner
from .model import AbstractModel
from .optimization import (evaluate_robust, evaluate, EpsNSGAII, 
                           to_dataframe, to_problem, to_robust_problem,
                           process_levers, process_uncertainties, 
                           process_robust)
from .outcomes import ScalarOutcome, AbstractOutcome
from .parameters import (experiment_generator, Scenario, Policy)
from .samplers import (MonteCarloSampler, FullFactorialSampler, LHSSampler, 
                       PartialFactorialSampler, sample_levers, 
                       sample_uncertainties)
from .salib_samplers import (SobolSampler, MorrisSampler, FASTSampler) # TODO:: should become optional import
from .util import NamedObjectMap, determine_objects
from ..util import ema_logging, EMAError


# Created on 5 Mar 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

LHS = 'lhs'
MC = 'mc'
FF = 'ff'
PFF = 'pff'
SOBOL = 'sobol'
MORRIS = 'morris'
FAST = 'fast'

#TODO:: better name, samplers lower case conflicts with module name
SAMPLERS = {LHS:LHSSampler,
            MC:MonteCarloSampler,
            FF:FullFactorialSampler,
            PFF:PartialFactorialSampler,
            SOBOL:SobolSampler,
            MORRIS:MorrisSampler,
            FAST:FASTSampler}

__all__ = ['MultiprocessingEvaluator', 'IpyparallelEvaluator', 
           'optimize', 'perform_experiments', 'SequentialEvaluator']

class BaseEvaluator(object):
    '''evaluator for experiments using a multiprocessing pool
    
    Parameters
    ----------
    msis : collection of models
    searchover : {None, 'levers', 'uncertainties'}, optional
                  to be used in combination with platypus
    union : {None, True, False}, optional
            to be used in combination with platypus, indicates whether
            you want to optimize over the union or the intersection of
            search_over
    
    Raises
    ------
    ValueError
    
    '''
    
    def __init__(self, msis, union=None):
        super(BaseEvaluator, self).__init__()
        
        if isinstance(msis, AbstractModel):
            msis = [msis]
        
        self._msis = msis
        
#         if searchover:
#             if searchover not in {'levers', 'uncertainties'}:
#                 raise ValueError(("search_over must be one of 'levers'"
#                               "or 'uncertainties' not {}".format(searchover)))
#             
#             self.searchover = searchover
#             
#             self.parameters = determine_objects(msis, searchover, union=union)
#             self.parameter_names = [p.name for p in self.parameters]
#             
#             outcomes = determine_objects(msis, "outcomes", union=union)
#             self.outcomes = [o for o in outcomes if
#                              o.kind != AbstractOutcome.INFO]
#             self.outcome_names = [o.name for o in self.outcomes]

    def __enter__(self):
        
        self.initialize()
        
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        
        self.finalize()
        
        if exc_type is not None:
            return False
        
    def initialize(self):
        ''' initialize the evaluator'''
        
        raise NotImplementedError
    
    def finalize(self):
        ''' finalize the evaluator'''
        
        raise NotImplementedError
        
    def evaluate_experiments(self, scenarios, policies, callback):
        '''used by ema_workbench'''
        
        raise NotImplementedError

    
    def evaluate_all(self, jobs, **kwargs):
        '''makes ema_workbench evaluators compatible with Platypus evaluators
        as used by platypus algorithms'''
        
        problem = jobs[0].solution.problem
        searchover = problem.searchover
        
        if searchover == 'levers':
            scenarios, policies = process_levers(jobs)
            jobs_collection = zip(policies, jobs)
        elif searchover == 'uncertainties':
            scenarios, policies = process_uncertainties(jobs)
            jobs_collection = zip(scenarios, jobs)
        elif searchover == 'robust':
            scenarios, policies = process_robust(jobs)
            jobs_collection = zip(policies, jobs)
        else:
            raise NotImplementedError()
        
        experiments, outcomes = perform_experiments(self._msis, 
                                        scenarios=scenarios, policies=policies, 
                                        evaluator=self)


        if searchover in ('levers', 'uncertainties'):
            evaluate(jobs_collection, experiments, outcomes, 
                      problem)
        else:
            evaluate_robust(jobs_collection, experiments, outcomes, 
                             problem)
            
        return jobs

    def perform_experiments(self, scenarios=0, policies=0, evaluator=None, 
                        reporting_interval=None, uncertainty_union=False, 
                        lever_union=False, outcome_union=False, 
                        uncertainty_sampling=LHS, levers_sampling=LHS):
        '''convenience method for performing experiments.
        
        is forwarded to :func:perform_experiments, with evaluator and models
        arguments added in.
        
        '''
        
        return perform_experiments(self._msis, scenarios=scenarios, 
                    policies=policies, evaluator=self, 
                    reporting_interval=reporting_interval, 
                    uncertainty_union=uncertainty_union, lever_union=lever_union, 
                    outcome_union=outcome_union, 
                    uncertainty_sampling=uncertainty_sampling, 
                    levers_sampling=levers_sampling)


    def optimize(self, algorithm=EpsNSGAII, nfe=10000, searchover='levers',
                 reference=None, **kwargs):
        '''convenience method for outcome optimization.
        
        is forwarded to :func:optimize, with evaluator and models
        arguments added in.
        
        '''
        
        return optimize(self._msis, algorithm=algorithm, nfe=nfe, 
                        searchover=searchover, evaluator=self, 
                        reference=reference,**kwargs)

        
    def robust_optimize(self, robustness_functions, scenarios, 
                        algorithm=EpsNSGAII, nfe=10000, searchover='levers',
                        **kwargs):
        '''convenience method for robust optimization.
        
        is forwarded to :func:robust_optimize, with evaluator and models
        arguments added in.
        
        '''
        return robust_optimize(self._msis, robustness_functions, scenarios,
                               self, algorithm=algorithm, nfe=nfe, **kwargs)


class SequentialEvaluator(BaseEvaluator):
    def __init__(self, models, **kwargs):
        super(SequentialEvaluator, self).__init__(models, **kwargs)
    
    def initialize(self):
        pass
    
    def finalize(self):
        pass
    
    def evaluate_experiments(self, scenarios, policies, callback):
        ema_logging.info("performing experiments sequentially")
        
        ex_gen = experiment_generator(scenarios, self._msis, policies)
        
        models = NamedObjectMap(AbstractModel)
        models.extend(self._msis)
        
        cwd = os.getcwd() 
        runner = ExperimentRunner(models)
        for experiment in ex_gen:
            result = runner.run_experiment(experiment)
            callback(experiment, result)
        runner.cleanup()
        os.chdir(cwd)
    

class MultiprocessingEvaluator(BaseEvaluator):
    '''evaluator for experiments using a multiprocessing pool
    
    Parameters
    ----------
    msis : collection of models
    n_processes : int (optional)
    
    
    '''
    
    def __init__(self, msis, n_processes=None, **kwargs):
        super(MultiprocessingEvaluator, self).__init__(msis, **kwargs)
        
        self._pool = None
        self.n_processes = n_processes

    def initialize(self):
        log_queue = multiprocessing.Queue()
    
        log_queue_reader = LogQueueReader(log_queue)
        log_queue_reader.start()
    
        try:
            loglevel = ema_logging._logger.getEffectiveLevel()
        except AttributeError:
            loglevel=30
            
            
        random_part = [random.choice(string.ascii_letters + string.digits) 
                     for _ in range(5)]
        random_part = ''.join(random_part)
        self.root_dir = os.path.abspath("tmp"+random_part)
        os.makedirs(self.root_dir)
    
        self._pool = multiprocessing.Pool(self.n_processes , initializer, 
                                  (self._msis, log_queue, loglevel, self.root_dir))
        ema_logging.info("pool started")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        ema_logging.info("terminating pool")
        
        if exc_type is not None:
            # When an exception is thrown stop accepting new jobs
            # and abort pending jobs without waiting.
            self._pool.terminate()
            return False
        
        super(MultiprocessingEvaluator, self).__exit__(exc_type, exc_value, traceback)

    def finalize(self):
        # Stop accepting new jobs and wait for pending jobs to finish.
        self._pool.close()
        self._pool.join()
        
        shutil.rmtree(self.root_dir)
        
    def evaluate_experiments(self, scenarios, policies, callback):
        ex_gen = experiment_generator(scenarios, self._msis, policies)
        
        add_tasks(self._pool, ex_gen, callback)


class IpyparallelEvaluator(BaseEvaluator):
    '''evaluator for using an ipypparallel pool'''
    

    def __init__(self,  msis, client, **kwargs):
        super(IpyparallelEvaluator, self).__init__(msis, **kwargs)
        self.client = client
        
    def initialize(self):
        import ipyparallel
        
        ema_logging.debug("starting ipyparallel pool")

        try:
            TIMEOUT_MAX = threading.TIMEOUT_MAX
        except AttributeError:
            TIMEOUT_MAX = 1e10  # noqa        
        ipyparallel.client.asyncresult._FOREVER = TIMEOUT_MAX
        # update loggers on all engines
        self.client[:].apply_sync(set_engine_logger)
        
        ema_logging.debug("initializing engines")
        initialize_engines(self.client, self._msis, 
                                                os.getcwd())
        
        self.logwatcher, self.logwatcher_thread = start_logwatcher()
        
        ema_logging.debug("successfully started ipyparallel pool")
        
        ema_logging.info("performing experiments using ipyparallel")
        
        return self


    def finalize(self):
        self.logwatcher.stop()
        cleanup(self.client)
        
        
    def evaluate_experiments(self, scenarios, policies, callback):
        ex_gen = experiment_generator(scenarios, self._msis, policies)
        
        lb_view = self.client.load_balanced_view()
        
        results = lb_view.map(_run_experiment, 
                              ex_gen, ordered=False, block=False)

        for entry in results:
            callback(*entry)
        


def perform_experiments(models, scenarios=0, policies=0, evaluator=None, 
                        reporting_interval=None, uncertainty_union=False, 
                        lever_union=False, outcome_union=False, 
                        uncertainty_sampling=LHS, levers_sampling=LHS):
    '''sample uncertainties and levers, and perform the resulting experiments
    on each of the models
    
    Parameters
    ----------
    models : one or more AbstractModel instances
    scenarios : int or collection of Scenario instances, optional
    policies :  int or collection of Policy instances, optional
    evaluator : Evaluator instance, optional
    reporting interval : int, optional
    uncertainty_union : boolean, optional
    lever_union : boolean, optional
    uncertainty_sampling : {LHS, MC, FF, PFF, SOBOL, MORRIS, FAST}, optional
    lever_sampling : {LHS, MC, FF, PFF, SOBOL, MORRIS, FAST}, optional
    
    
    '''
    if not scenarios and not policies:
        raise EMAError(('no experiments possible since both ' 
                        'scenarios and policies are 0'))
    
    if not scenarios:
        scenarios = [Scenario("None", **{})]
        uncertainties = []
        n_scenarios = 1
    elif(isinstance(scenarios, numbers.Integral)):
        scenarios = sample_uncertainties(models, scenarios, 
             union=uncertainty_union, sampler=SAMPLERS[uncertainty_sampling]())
        uncertainties = scenarios.parameters
        n_scenarios = scenarios.n
    else:
        try:
            uncertainties = scenarios.parameters
            n_scenarios = scenarios.n
        except AttributeError:
            uncertainties = determine_objects(models, "uncertainties", 
                                              union=True)
            if isinstance(scenarios, Scenario):
                scenarios = [scenarios]
            
            uncertainties = [u for u in uncertainties if u.name in 
                             scenarios[0]]
            n_scenarios = len(scenarios)
        
    
    if not policies:
        policies = [Policy("None", **{})]
        levers = []
        n_policies = 1
    elif(isinstance(policies, numbers.Integral)):    
        policies = sample_levers(models, policies, union=lever_union, 
                                 sampler=SAMPLERS[levers_sampling]())
        levers = policies.parameters
        n_policies = policies.n
    else:
        try:
            levers = policies.parameters
            n_policies = policies.n
        except AttributeError:
            levers = determine_objects(models, "levers", union=True)
            if isinstance(policies, Policy):
                policies = [policies]
            
            levers = [l for l in levers if l.name in policies[0]]
            n_policies = len(policies)
    try:
        n_models = len(models)
    except TypeError:
        n_models = 1

    outcomes = determine_objects(models, 'outcomes', union=outcome_union)
    nr_of_exp = n_models * n_scenarios * n_policies 
    
    ema_logging.info("performing {} scenarios * {} policies * {} model(s) = {} experiments".format(n_scenarios, n_policies, n_models, nr_of_exp))
    
    callback = DefaultCallback(uncertainties,
                               levers,
                               outcomes,
                               nr_of_exp,
                               reporting_interval=reporting_interval)
    
    if not evaluator:
        evaluator = SequentialEvaluator(models)
    
    evaluator.evaluate_experiments(scenarios, policies, callback)
    
    if callback.i != nr_of_exp:
        raise EMAError(('some fatal error has occurred while '
                        'running the experiments, not all runs have ' 
                        'completed. expected {} '.format(nr_of_exp),
                        'got {}'.format(callback.i),
                        '{}'.format(type(callback))))
       
    results = callback.get_results()
    ema_logging.info("experiments finished")
    return results



def optimize(models, algorithm=EpsNSGAII, nfe=10000, 
             searchover='levers', evaluator=None, reference=None,
             **kwargs):
    '''optimize the model
    
    Parameters
    ----------
    models : 1 or more Model instances
    algorithm : a valid Platypus optimization algorithm
    nfe : int
    searchover : {'uncertainties', 'levers'}
    kwargs : aditional argumenst to pass on to algorithm
    
    Returns
    -------
    pandas DataFrame
    
    
    Raises
    ------
    EMAError if searchover is not one of 'uncertainties' or 'levers'
    NotImplementedError if len(models) > 1
    
    TODO:: constrains are not yet supported
    TODO:: add an optional single policy/scenario to overwrite the defaults
    
    
    '''
    if searchover not in ('levers', 'uncertainties'):
        raise EMAError(("searchover should be one of 'levers' or"
                        "'uncertainties' not {}".format(searchover)))
        
    try:
        if len(models)==1:
            models = models[0]
        else:
            raise NotImplementedError(("optimization over multiple" 
                                      "models yet supported"))
    except TypeError:
        pass
        
    
    problem = to_problem(models, searchover, reference=reference)
    
    # solve the optimization problem
    if not evaluator:
        evaluator = SequentialEvaluator(models)

    optimizer = algorithm(problem, evaluator=evaluator, **kwargs)
    optimizer.run(nfe)
    results = to_dataframe(optimizer, problem.parameter_names, 
                           problem.outcome_names)

    message = "optimization completed, found {} solutions"
    ema_logging.info(message.format(len(optimizer.algorithm.archive)))

    return results


    

def robust_optimize(model, robustness_functions, scenarios, 
                    evaluator=None, algorithm=EpsNSGAII, nfe=10000, 
                    **kwargs):
    '''
    
    Parameters
    ----------
    model : 
    robustness_functions : collection of ScalarOutcomes
    scenarios : 
    evaluator : Evaluator instance
    algorithm : platypus Algorithm instance
    nfe : int
    kwargs : any additional arguments will be passed on to algorithm
    
    
    Raises
    ------
    AssertionError if robustness_function is not a ScalarOutcome,
    if robustness_funcion.kind is INFO, or 
    if robustness_function.function is None
    
    
    robustness functions are scalar outcomes, kind should be MINIMIZE or
    MAXIMIZE, function is the robustness function you want to use.
    
    
    
    '''
    
    for rf in robustness_functions:
        assert(isinstance(rf, ScalarOutcome))
        assert(rf.kind != AbstractOutcome.INFO)
        assert(rf.function != None)
    
    problem = to_robust_problem(model, scenarios, robustness_functions)

    # solve the optimization problem
    if not evaluator:
        evaluator = SequentialEvaluator(model)
        
    optimizer = algorithm(problem, evaluator=evaluator, **kwargs)
    optimizer.run(nfe)
    
    results = to_dataframe(optimizer, problem.parameter_names, 
                           problem.outcome_names)

    message = "optimization completed, found {} solutions"
    ema_logging.info(message.format(len(optimizer.algorithm.archive)))

    return results
    
