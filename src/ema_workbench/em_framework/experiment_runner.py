'''
helper module for running experiments and keeping track of which model
has been initialized with which policy. 
'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import copy

from ..util import ema_logging, EMAError, CaseError
from pickle import PicklingError

# Created on Aug 11, 2015
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ["ExperimentRunner"]

class ExperimentRunner(object):
    '''Helper class for running the experiments
    
    This class contains the logic for initializing models properly,
    running the experiment, getting the results, and cleaning up afterwards.
    
    Parameters
    ----------
    msis : dict
    model_kwargs : dict
    
    Attributes
    ----------
    msi_initializiation : dict
                          keeps track of which model is initialized with
                          which policy. 
    msis : dict
           models indexed by name
    model_kwargs : dict
                   keyword arguments for model_init
    
    '''
    
    def __init__ (self, msis, model_kwargs):
        self.msi_initialization = {}
        self.msis = msis
        self.model_kwargs = model_kwargs
    
    def cleanup(self):
        for msi in self.msis:
            msi.cleanup()
        self.msis = None
    
    def run_experiment(self, experiment):
        '''The logic for running a single experiment. This code makes
        sure that model(s) are initialized correctly.
        
        Parameters
        ----------
        experiment : dict
        
        Returns
        -------
        experiment_id: int
        case : dict
        policy : str
        model_name : str
        result : dict
        
        Raises
        ------
        EMAError
            if the model instance raises an EMA error, these are reraised.
        Exception
            Catch all for all other exceptions being raised by the model. 
            These are reraised.
        
        '''
        
        policy_name = experiment.policy.name
        model_name = experiment.model_name
        msi = self.msis[model_name]
        policy = experiment.policy
        experiment_id = experiment.experiment_id
        
        ema_logging.debug("running policy {} for experiment {}".format(policy_name, 
                                                           experiment_id))
        
        # check whether we already initialized the model for this 
        # policy
        if not (policy_name, model_name) in self.msi_initialization.keys():
            policy = copy.deepcopy(policy)
            model_kwargs = copy.deepcopy(self.model_kwargs)
            
            
            try:
                msi.model_init(policy, model_kwargs)
            except EMAError as inst:
                ema_logging.exception(inst)
                self.cleanup()
                raise inst
            except Exception as inst:
                ema_logging.exception("some exception occurred when invoking the init")
                self.cleanup()
                raise inst
                
            ema_logging.debug("initialized model %s with policy %s" % (model_name, 
                                                           policy_name))

            self.msi_initialization = {(policy_name, model_name):self.msis[model_name]}
        

        case = copy.deepcopy(experiment.scenario)
        try:
            msi.run_model(case)
        except CaseError as e:
            ema_logging.warning(str(e))
        except Exception as e:
            raise EMAError('some exception has been raised by run_model '+str(e))
            
        output = msi.output
        msi.reset_model()
        
        return output      