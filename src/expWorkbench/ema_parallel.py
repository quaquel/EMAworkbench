'''
Module provides the high level interface to working with either a 
multiprocessing pool or ipython parallel pool. 

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
import abc

from .ema_parallel_ipython import (_run_experiment, 
                                   initialize_engines, set_engine_logger)
from .ema_parallel_multiprocessing import CalculatorPool

# Created on Jul 22, 2015
# 
# .. codeauthor:: jhkwakkel@tudelft.net

__all__ = ['AbstractPool',
           'MultiprocessingPool',
           'IpyparallelPool']


class AbstractPool(object):
    '''
    Abstract base class for a pool of workers. 
    
    Parameters
    ----------
    msis : iterable
           iterable of model structure interface instances
    model_kwargs : dict
    
    '''
    
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def __init__(self, msis, model_kwargs={}):
        pass
        
    @abc.abstractmethod
    def perform_experiments(self, callback, experiments):
        ''' 
        perform the experiments using the pool
        
        
        Parameters
        ----------
        callback : a Callback instance
        experiments : collection of dicts
        
        '''
    

class MultiprocessingPool(AbstractPool):
    '''
    Extension of AbstractPool which wraps the multiprocessing pool.
    
    
    '''

    def __init__(self, msis, model_kwargs={}, nr_processes=None):
        self._pool = CalculatorPool(msis, processes=nr_processes, 
                                    kwargs=model_kwargs)
    
    def perform_experiments(self, callback, experiments):
        self._pool.run_experiments(experiments, callback)

class IpyparallelPool(AbstractPool):
    '''
    Extension of AbstractPool which wraps a ipyparallel cluster.
    
    Parameters
    ----------
    msis : iterable
           iterable of model structure interface instances
    cliet : IPython.parallel.client instance
    model_kwargs : dict
        
    
    '''
    
    def __init__(self, msis, client, model_kwargs={}):
        self.client = client
        
        # update loggers on all engines
        client[:].apply_sync(set_engine_logger)
        
        initialize_engines(self.client, msis, model_kwargs)
    
    def perform_experiments(self, callback, experiments):
        lb_view = self.client.load_balanced_view()
        
        results = lb_view.map(_run_experiment, experiments, ordered=False)

        # we can also get the results
        # as they arrive
        for entry in results:
            experiment_id, case, policy, model_name, result = entry
            callback(experiment_id, case, policy, model_name, result)
            
