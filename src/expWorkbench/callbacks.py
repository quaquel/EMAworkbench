'''
Created on 22 Jan 2013

@author: jhkwakkel
'''
from __future__ import division
import numpy as np
from expWorkbench.ema_exceptions import EMAError
np = np

from threading import Lock


from expWorkbench import ema_logging

from expWorkbench.ema_logging import info, debug
from uncertainties import CategoricalUncertainty,\
                                       ParameterUncertainty,\
                                       INTEGER

__all__ = ['AbstractCallback',
           'DefaultCallback'
          ]

class AbstractCallback(object):
    '''
    Base class from which different call back classes can be derived.
    Callback is responsible for storing the results of the runs.
    
    '''
    
    i = 0
    reporting_interval = 100
    results = []
    
    def __init__(self, 
                 uncertainties, 
                 outcomes,
                 nr_experiments,
                 reporting_interval=100):
        '''
        
        :param uncs: a list of the uncertainties over which the experiments 
                     are being run.
        :param outcomes: a list of outcomes
        :param nr_experiments: the total number of experiments to be executed
        :param reporting_interval: the interval at which to provide
                                   progress information via logging.
        
                
        '''
        self.reporting_interval = reporting_interval
            
    
    def __call__(self, case_id, case, policy, name, result):
        '''
        Method responsible for storing results. The implementation in this
        class only keeps track of how many runs have been completed and 
        logging this. 
        
        :param case_id: the job id
        :param case: the case to be stored
        :param policy: the name of the policy being used
        :param name: the name of the model being used
        :param result: the result dict
        
        '''
        
        self.i+=1
        debug(str(self.i)+" cases completed")
        
        if self.i % self.reporting_interval == 0:
            info(str(self.i)+" cases completed")

    def get_results(self):
        """
        method for retrieving the results. Called after all experiments have 
        been completed
        """
        self.results
        
class DefaultCallback(AbstractCallback):
    """ 
    default callback system
    callback can be used in performExperiments as a means for specifying 
    the way in which the results should be handled. If no callback is 
    specified, this default implementation is used. This one can be 
    overwritten or replaced with a callback of your own design. For 
    example if you prefer to store the result in a database or write 
    them to a text file
    """
    
    i = 0
    cases = None
    policies = None
    names = None   
    results = {}
    
    shape_error_msg = "can only save up to 2d arrays, this array is {}d"
    
    def __init__(self, 
                 uncs, 
                 outcomes, 
                 nr_experiments, 
                 reporting_interval=100):
        '''
        
        
        :param uncs: a list of the uncertainties over which the experiments 
                     are being run.
        :param outcomes: a list of outcomes
        :param nr_experiments: the total number of experiments to be executed
        :param reporting_interval: the interval at which to provide
                                   progress information via logging.
        
        '''
        
        
        super(DefaultCallback, self).__init__(uncs, 
                                              outcomes, 
                                              nr_experiments, 
                                              reporting_interval)
        self.i = 0
        self.cases = None
        self.policies = None
        self.names = None   
        self.results = {}
        self.lock = Lock()
        
        self.outcomes = outcomes

        #determine data types of uncertainties
        self.dtypes = []
        self.uncertainties = []
        
        for uncertainty in uncs:
            name = uncertainty.name
            self.uncertainties.append(name)
            dataType = float
            
            if isinstance(uncertainty, CategoricalUncertainty):
                dataType = object
            elif isinstance(uncertainty, ParameterUncertainty) and\
                          uncertainty.dist==INTEGER:
                dataType = int
            self.dtypes.append((name, dataType))
        self.dtypes.append(('model', object))
        self.dtypes.append(('policy', object))
        
        self.cases = np.empty((nr_experiments,), dtype=self.dtypes)
        self.cases[:] = np.NAN
        self.nr_experiments = nr_experiments
        

    def _store_case(self, case_id, case, model, policy):
        case = [case.get(key) for key in self.uncertainties]
        case.append(model)
        case.append(policy)
        case = tuple(case)
        self.cases[case_id] = case
            
    def _store_result(self, case_id, result):
        for outcome in self.outcomes:
            try:
                debug("storing {}".format(outcome))
            except ValueError:
                print "what"
            
            try:
                outcome_res = result[outcome]
            except KeyError:
                ema_logging.debug("%s not in msi" % outcome)
            else:
                try:
                    self.results[outcome][case_id, ] = outcome_res
                except KeyError: 
                    shape = np.asarray(outcome_res).shape
                    
                    if len(shape)>2:
                        raise EMAError(self.shape_error_msg.format(len(shape)))
                    
                    shape = list(shape)
                    shape.insert(0, self.nr_experiments)
                    self.results[outcome] = np.empty(shape)
                    self.results[outcome][:] = np.NAN
                    self.results[outcome][case_id, ] = outcome_res
    
    def __call__(self, case_id, case, policy, name, result ):
        '''
        Method responsible for storing results. This method calls 
        :meth:`super` first, thus utilizing the logging provided there
        
        :param case_id: the id of the case
        :param case: the case to be stored
        :param policy: the name of the policy being used
        :param name: the name of the model being used
        :param result: the result dict. This implementation assumes that
                       the values in this dict can be cast to numpy arrays. 
                       Any shape is supported. The code takes the shape of the
                       array and adds the nr_experiments to it as first 
                       dimension.
        :return: a tuple with the cases structured array and the dict of 
                 result arrays. 
        
        '''
        super(DefaultCallback, self).__call__(case_id, case, policy, name, result)

        self.lock.acquire()
                           
        #store the case
        self._store_case(case_id, case, name, policy.get('name'), )
        
        #store results
        self._store_result(case_id, result)
        
        self.lock.release()
        
        
    def get_results(self):
        return self.cases, self.results