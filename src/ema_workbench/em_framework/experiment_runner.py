'''
helper module for running experiments and keeping track of which model
has been initialized with which policy. 
'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import sys
import traceback

from ..util import ema_logging, EMAError, CaseError

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
    
    def __init__ (self, msis):
        self.msis = msis
        self.log_message = ('running scenario {scenario_id} for policy '
                            '{policy_name} on model {model_name}')
    
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
        model = self.msis[model_name]
        policy = experiment.policy.copy()
        scenario_id = experiment.scenario.name
        
        ema_logging.debug(self.log_message.format(scenario_id=scenario_id,
                                                  policy_name=policy_name,
                                                  model_name = model_name))
        scenario = experiment.scenario.copy()
        try:
            model.run_model(scenario, policy)
        except CaseError as e:
            ema_logging.warning(str(e))
        except Exception as e:
            try:
                self.cleanup()
            except Exception:
                raise e
            
            exception = traceback.print_exc()
            if exception:
                sys.stderr.write(exception)
                sys.stderr.write("\n")
            
            raise EMAError(("exception in run_model"
                   "\nCaused by: {}: {}".format(type(e).__name__, str(e))))
            
        output = model.output
        model.reset_model()
        
        return output      